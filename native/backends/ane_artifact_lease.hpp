#pragma once

// Private compiled-graph snapshots, without Core ML or model dependencies.
// An exclusive kernel lock outlives every graph reader. A reaper must acquire
// that SAME lock inode before deleting a completed lease. PID reuse, clock
// changes, and a second TurboCider process therefore cannot evict a live graph.
// Old/unmarked directories are deliberately not inferred to be ours.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace tc::ane::detail {

class ArtifactLease {
  public:
    using Reporter = std::function<void(const std::string &)>;
    struct ReapResult { size_t removed = 0, live = 0, ignored = 0, failed = 0; };
    std::filesystem::path root;

    explicit ArtifactLease(const std::filesystem::path &temporary = std::filesystem::temp_directory_path(),
                           Reporter reporter = {}) : reporter_(std::move(reporter)) {
        auto opened = open_temporary(temporary);
        temporary_ = std::move(opened.first);
        parent_ = std::move(opened.second);
        reap_at(parent_.get(), temporary_, reporter_);
        // mkdirat is exclusive and relative to the verified parent descriptor.
        // A random name is not an ownership credential; the marker below is.
        for (int attempt = 0; attempt < 16; ++attempt) {
            unsigned long long nonce = 0;
            require(::getentropy(&nonce, sizeof(nonce)) == 0, "cannot generate snapshot name");
            char name[64];
            std::snprintf(name, sizeof(name), "turbocider-runtime-ane-%016llx", nonce);
            if (::mkdirat(parent_.get(), name, 0700) == 0) { name_ = name; break; }
            require(errno == EEXIST, "cannot create private snapshot directory");
        }
        require(!name_.empty(), "cannot allocate unique snapshot directory");
        root = temporary_ / name_;
        try {
            directory_ = open_directory(parent_.get(), name_.c_str());
            identity_ = status(directory_.get());
            require(private_directory(identity_), "snapshot directory ownership or permissions changed");
            create_marker(directory_.get(), identity_, marker_);
            // The lock is acquired before the marker becomes recognizable;
            // graph.mlmodelc is written only after this constructor returns.
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~ArtifactLease() { cleanup(); }
    ArtifactLease(const ArtifactLease &) = delete;
    ArtifactLease &operator=(const ArtifactLease &) = delete;

    // Call only AFTER the worker and Core ML readers have gone. Failures are
    // observable. A final rmdir failure restores a newly locked marker when
    // ownership is still provable; a crash or restoration failure can still
    // leave an unrecognized empty directory. Do not claim zero crash residue.
    bool cleanup() noexcept {
        if (root.empty()) return true;
        try {
            require(directory_.get() >= 0 && marker_.get() >= 0,
                    "incomplete snapshot ownership; directory retained");
            erase(parent_.get(), name_, directory_.get(), identity_, marker_);
            root.clear();
            marker_ = Fd();
            directory_ = Fd();
            return true;
        } catch (const std::exception &error) {
            report(reporter_, root, error.what());
            return false;
        } catch (...) {
            report(reporter_, root, "unknown snapshot cleanup failure");
            return false;
        }
    }

    // Bounded metadata-only recovery; no model loads, timers, or background
    // thread. A crash before marker completion, or between final marker/root
    // removal, can leave an empty/unrecognized directory. It is intentionally
    // never auto-deleted; a completed graph has a marker before it is written.
    static ReapResult reap(const std::filesystem::path &temporary, Reporter reporter = {}) {
        auto opened = open_temporary(temporary);
        return reap_at(opened.second.get(), opened.first, reporter);
    }

  private:
    static constexpr const char *marker_name = ".runtime-ane-lease-v1";
    static constexpr size_t max_entries = 4096, max_reaped = 16, max_depth = 32;
    class Fd {
        int value_ = -1;
      public:
        explicit Fd(int value = -1) : value_(value) {}
        ~Fd() { if (value_ >= 0) ::close(value_); }
        Fd(const Fd &) = delete;
        Fd &operator=(const Fd &) = delete;
        Fd(Fd &&other) noexcept : value_(std::exchange(other.value_, -1)) {}
        Fd &operator=(Fd &&other) noexcept {
            if (this != &other) {
                if (value_ >= 0) ::close(value_);
                value_ = std::exchange(other.value_, -1);
            }
            return *this;
        }
        int get() const { return value_; }
        int release() { return std::exchange(value_, -1); }
    };
    Reporter reporter_;
    std::filesystem::path temporary_;
    std::string name_;
    Fd parent_, directory_, marker_;
    struct stat identity_{};

    static void require(bool good, const char *reason) {
        if (!good) throw std::runtime_error(std::string("runtime ANE artifact lease: ") + reason);
    }
    static struct stat status(int fd) {
        struct stat result{};
        require(::fstat(fd, &result) == 0, "cannot inspect open snapshot entry");
        return result;
    }
    static struct stat entry_status(int parent, const char *name) {
        struct stat result{};
        require(::fstatat(parent, name, &result, AT_SYMLINK_NOFOLLOW) == 0,
                "cannot inspect snapshot directory entry");
        return result;
    }
    static bool same(const struct stat &a, const struct stat &b) {
        return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
               a.st_mode == b.st_mode && a.st_uid == b.st_uid &&
               (!S_ISREG(a.st_mode) || a.st_nlink == b.st_nlink);
    }
    static bool private_directory(const struct stat &s) {
        return S_ISDIR(s.st_mode) && s.st_uid == ::geteuid() && (s.st_mode & 07777) == 0700;
    }
    static bool private_marker(const struct stat &s) {
        return S_ISREG(s.st_mode) && s.st_uid == ::geteuid() && s.st_nlink == 1 &&
               (s.st_mode & 07777) == 0600;
    }
    static Fd open_directory(int parent, const char *name) {
        Fd result(::openat(parent, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        require(result.get() >= 0, "cannot open snapshot directory without following links");
        require(same(status(result.get()), entry_status(parent, name)), "snapshot directory inode changed");
        return result;
    }
    static std::pair<std::filesystem::path, Fd> open_temporary(const std::filesystem::path &path) {
        // macOS's standard /var and /tmp aliases are resolved once. Every
        // component of the resulting absolute path is then opened NOFOLLOW.
        const auto canonical = std::filesystem::canonical(path);
        Fd fd(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        require(fd.get() >= 0, "cannot open filesystem root");
        for (const auto &part : canonical.relative_path()) fd = open_directory(fd.get(), part.c_str());
        const auto s = status(fd.get());
        require((s.st_uid == ::geteuid() && !(s.st_mode & 0022)) ||
                    (s.st_uid == 0 && (s.st_mode & S_ISVTX)),
                "temporary parent must be private or root-owned sticky");
        return {canonical, std::move(fd)};
    }
    static std::string marker_content(const struct stat &s, const struct stat &marker) {
        return "turbocider-runtime-ane-private-snapshot-v1\n" +
            std::to_string(static_cast<unsigned long long>(s.st_uid)) + " " +
            std::to_string(static_cast<unsigned long long>(s.st_dev)) + " " +
            std::to_string(static_cast<unsigned long long>(s.st_ino)) + " " +
            std::to_string(static_cast<unsigned long long>(marker.st_dev)) + " " +
            std::to_string(static_cast<unsigned long long>(marker.st_ino)) + "\n";
    }
    static void create_marker(int directory, const struct stat &identity, Fd &marker) {
        Fd created(::openat(directory, marker_name,
                            O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
        require(created.get() >= 0, "cannot create snapshot lease marker");
        require(private_marker(status(created.get())), "invalid snapshot lease marker");
        require(::flock(created.get(), LOCK_EX | LOCK_NB) == 0, "cannot lock snapshot lease");
        // Publish ownership before writing, so a write failure can still be
        // cleaned/retried by this owner. Reapers never accept a partial body.
        marker = std::move(created);
        const auto content = marker_content(identity, status(marker.get()));
        size_t offset = 0;
        while (offset < content.size()) {
            const auto written = ::write(marker.get(), content.data() + offset, content.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            require(written > 0, "cannot write snapshot lease marker");
            offset += size_t(written);
        }
    }
    static bool valid_marker(int fd, const struct stat &directory) {
        const auto s = status(fd);
        const auto expected = marker_content(directory, s);
        if (!private_marker(s) || s.st_dev != directory.st_dev || s.st_size != off_t(expected.size())) return false;
        std::string actual(expected.size(), '\0');
        size_t offset = 0;
        while (offset < actual.size()) {
            const auto read = ::pread(fd, actual.data() + offset, actual.size() - offset, off_t(offset));
            if (read < 0 && errno == EINTR) continue;
            if (read <= 0) return false;
            offset += size_t(read);
        }
        return actual == expected;
    }
    static bool lease_name(const std::string &name) {
        const std::string prefix = "turbocider-runtime-ane-";
        return name.size() == prefix.size() + 16 && name.starts_with(prefix) &&
               name.find_first_not_of("0123456789abcdef", prefix.size()) == std::string::npos;
    }
    static std::vector<std::string> entries(int fd, size_t limit = max_entries) {
        Fd duplicate(::openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        require(duplicate.get() >= 0, "cannot enumerate snapshot directory");
        std::unique_ptr<DIR, decltype(&::closedir)> directory(::fdopendir(duplicate.get()), &::closedir);
        require(bool(directory), "cannot enumerate snapshot directory");
        duplicate.release();
        std::vector<std::string> result;
        for (;;) {
            errno = 0;
            const auto *entry = ::readdir(directory.get());
            if (!entry) { require(errno == 0, "snapshot directory enumeration failed"); break; }
            std::string name(entry->d_name);
            if (name == "." || name == "..") continue;
            require(result.size() < limit, "snapshot enumeration safety limit exceeded");
            result.push_back(std::move(name));
        }
        return result;
    }
    static void remove_graph_contents(int fd, dev_t device, size_t depth, size_t &remaining) {
        require(depth < max_depth, "snapshot tree depth limit exceeded");
        for (const auto &name : entries(fd)) {
            require(remaining > 0, "snapshot tree entry limit exceeded");
            --remaining;
            const auto before = entry_status(fd, name.c_str());
            require(before.st_uid == ::geteuid() && before.st_dev == device && !(before.st_mode & 0022),
                    "snapshot tree ownership, device or permissions changed");
            if (S_ISDIR(before.st_mode)) {
                auto child = open_directory(fd, name.c_str());
                require(same(before, status(child.get())), "snapshot subtree inode changed");
                remove_graph_contents(child.get(), device, depth + 1, remaining);
                require(same(before, entry_status(fd, name.c_str())), "snapshot subtree inode changed before removal");
                require(::unlinkat(fd, name.c_str(), AT_REMOVEDIR) == 0, "cannot remove snapshot subtree");
            } else {
                require(S_ISREG(before.st_mode) && before.st_nlink == 1,
                        "snapshot contains symlink, hardlink or nonregular file; retained");
                require(same(before, entry_status(fd, name.c_str())), "snapshot file inode changed before removal");
                require(::unlinkat(fd, name.c_str(), 0) == 0, "cannot remove snapshot file");
            }
        }
    }
    static void erase(int parent, const std::string &name, int directory,
                      const struct stat &identity, Fd &marker) {
        require(private_directory(status(directory)) && same(identity, status(directory)) &&
                    same(identity, entry_status(parent, name.c_str())), "snapshot root identity changed; retained");
        const auto lock_identity = status(marker.get());
        require(private_marker(lock_identity) &&
                    same(lock_identity, entry_status(directory, marker_name)), "snapshot lock identity changed; retained");
        const auto children = entries(directory);
        for (const auto &child : children)
            require(child == marker_name || child == "graph.mlmodelc", "unrecognized snapshot root entry; retained");
        for (const auto &child : children) if (child == "graph.mlmodelc") {
            const auto before = entry_status(directory, child.c_str());
            require(S_ISDIR(before.st_mode) && before.st_uid == ::geteuid() &&
                        before.st_dev == identity.st_dev && !(before.st_mode & 0022),
                    "compiled snapshot root changed; retained");
            auto graph = open_directory(directory, child.c_str());
            require(same(before, status(graph.get())), "compiled snapshot inode changed; retained");
            size_t remaining = max_entries;
            remove_graph_contents(graph.get(), identity.st_dev, 0, remaining);
            require(same(before, entry_status(directory, child.c_str())), "compiled snapshot inode changed before removal");
            require(::unlinkat(directory, child.c_str(), AT_REMOVEDIR) == 0, "cannot remove compiled snapshot");
        }
        // Marker last: interrupted graph cleanup remains recognizable. Hold
        // its lock until the containing directory has actually been removed.
        require(same(identity, entry_status(parent, name.c_str())) &&
                    same(lock_identity, entry_status(directory, marker_name)), "snapshot identity changed before final removal");
        require(::unlinkat(directory, marker_name, 0) == 0, "cannot remove snapshot lease marker");
        if (::unlinkat(parent, name.c_str(), AT_REMOVEDIR) != 0) {
            const int removal_error = errno;
            // Preserve recoverability if only the parent permission changed,
            // for example. Never overwrite a substituted marker or recreate
            // one in a directory whose parent/name/inode identity changed.
            require(private_directory(status(directory)) && same(identity, status(directory)) &&
                        same(identity, entry_status(parent, name.c_str())),
                    "snapshot root changed after final removal failure; marker not restored");
            create_marker(directory, identity, marker);
            throw std::runtime_error("runtime ANE artifact lease: cannot remove private snapshot directory: " +
                std::string(std::strerror(removal_error)) + "; locked marker restored for retry");
        }
    }
    static void report(const Reporter &reporter, const std::filesystem::path &path, const char *reason) noexcept {
        try {
            const std::string message = "runtime ANE artifact cleanup: " + path.string() + ": " + reason;
            if (reporter) reporter(message);
            else std::fprintf(stderr, "%s\n", message.c_str());
        } catch (...) { std::fputs("runtime ANE artifact cleanup failed (diagnostic unavailable)\n", stderr); }
    }
    static ReapResult reap_at(int parent, const std::filesystem::path &path, const Reporter &reporter) {
        ReapResult result;
        std::vector<std::string> candidates;
        try { candidates = entries(parent); }
        catch (const std::exception &error) { report(reporter, path, error.what()); return result; }
        for (const auto &name : candidates) {
            if (!lease_name(name)) continue;
            if (result.removed + result.failed >= max_reaped) break;
            try {
                auto directory = open_directory(parent, name.c_str());
                const auto identity = status(directory.get());
                if (!private_directory(identity) || identity.st_dev != status(parent).st_dev) { ++result.ignored; continue; }
                Fd marker(::openat(directory.get(), marker_name, O_RDWR | O_CLOEXEC | O_NOFOLLOW));
                if (marker.get() < 0 || !valid_marker(marker.get(), identity)) { ++result.ignored; continue; }
                if (::flock(marker.get(), LOCK_EX | LOCK_NB) != 0) {
                    if (errno == EWOULDBLOCK || errno == EAGAIN) { ++result.live; continue; }
                    require(false, "cannot acquire abandoned snapshot lock");
                }
                // Recheck after acquiring the lock; another reaper may have
                // deleted/replaced a name between our open and flock calls.
                require(valid_marker(marker.get(), identity), "snapshot lease marker changed after locking");
                erase(parent, name, directory.get(), identity, marker);
                ++result.removed;
            } catch (const std::exception &error) {
                ++result.failed;
                report(reporter, path / name, error.what());
            }
        }
        return result;
    }
};

} // namespace tc::ane::detail
