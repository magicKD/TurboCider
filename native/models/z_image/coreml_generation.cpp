#include "coreml_generation.hpp"
#include "../../core/common.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

namespace tc::z_image {
namespace {
namespace fs = std::filesystem;
using streaming::OwnedSourceFd;
struct Directory {
    uint64_t device, inode;
    int64_t mtime_s, mtime_ns, ctime_s, ctime_ns;
    bool operator==(const Directory &) const = default;
};
Directory directory(const fs::path &path) {
    struct stat s{};
    require(lstat(path.c_str(), &s) == 0 && S_ISDIR(s.st_mode), "Core ML generation: invalid directory");
    return {uint64_t(s.st_dev), uint64_t(s.st_ino), s.st_mtimespec.tv_sec,
            s.st_mtimespec.tv_nsec, s.st_ctimespec.tv_sec, s.st_ctimespec.tv_nsec};
}
void cancelled(const std::atomic<bool> *flag) {
    if (flag && flag->load()) throw Cancelled();
}
void no_link_ancestors(const fs::path &path) {
    fs::path current;
    for (const auto &part : path) {
        current /= part;
        require(!fs::is_symlink(fs::symlink_status(current)), "Core ML generation: symlink ancestor");
    }
}
struct Inventory {
    std::map<std::string, Directory> directories;
    std::vector<streaming::SourceFileIdentity> files;
};
Inventory inventory(const fs::path &root, const std::atomic<bool> *flag) {
    no_link_ancestors(root);
    Inventory out;
    out.directories.emplace("", directory(root));
    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        cancelled(flag);
        auto relative = entry.path().lexically_relative(root).generic_string();
        auto status = entry.symlink_status();
        require(!fs::is_symlink(status), "Core ML generation: symlink entry");
        if (fs::is_directory(status)) out.directories.emplace(relative, directory(entry.path()));
        else {
            require(fs::is_regular_file(status), "Core ML generation: nonregular entry");
            streaming::SourceFileIdentity file;
            file.logical_id = relative; file.path = entry.path();
            out.files.push_back(std::move(file));
        }
    }
    std::sort(out.files.begin(), out.files.end(), [](const auto &a, const auto &b) { return a.logical_id < b.logical_id; });
    require(!out.files.empty(), "Core ML generation: empty tree");
    return out;
}
void same_inventory(const Inventory &a, const Inventory &b) {
    require(a.directories == b.directories && a.files.size() == b.files.size(), "Core ML generation: tree changed");
    for (size_t i = 0; i < a.files.size(); ++i)
        require(a.files[i].logical_id == b.files[i].logical_id, "Core ML generation: entry set changed");
}
}
struct CoreMLGeneration::State {
    fs::path root;
    Directory created{};
    Inventory snapshot;
    std::shared_ptr<const streaming::SourceLease> lease;
    uint64_t copied = 0;
    ~State() {
        if (root.empty()) return;
        try {
            auto current = directory(root);
            // Never recursively remove a replacement installed at our name.
            if (current.device != created.device || current.inode != created.inode) return;
            fs::permissions(root, fs::perms::owner_all);
            for (auto &entry : fs::recursive_directory_iterator(root))
                if (fs::is_directory(entry.symlink_status())) fs::permissions(entry.path(), fs::perms::owner_all);
            fs::remove_all(root);
        } catch (...) { /* Destruction cannot throw on external tampering. */ }
    }
};
CoreMLGeneration::CoreMLGeneration(std::unique_ptr<State> s) : state_(std::move(s)) {}
CoreMLGeneration::~CoreMLGeneration() = default;
const fs::path &CoreMLGeneration::root() const noexcept { return state_->root; }
std::string_view CoreMLGeneration::content_digest() const noexcept { return state_->lease->artifact_digest(); }
uint64_t CoreMLGeneration::copied_bytes() const noexcept { return state_->copied; }
void CoreMLGeneration::revalidate() const {
    same_inventory(state_->snapshot, inventory(state_->root, nullptr));
    state_->lease->revalidate_after_drain();
}
std::shared_ptr<const CoreMLGeneration> CoreMLGeneration::import_tree(
        const fs::path &source, const fs::path &private_parent, const std::atomic<bool> *flag) {
    cancelled(flag);
    auto src = fs::absolute(source).lexically_normal();
    auto parent = fs::absolute(private_parent).lexically_normal();
    no_link_ancestors(parent);
    directory(parent);
    auto initial = inventory(src, flag);
    auto source_lease = streaming::SourceLease::capture_verified(initial.files, flag);
    same_inventory(initial, inventory(src, flag));
    auto state = std::make_unique<State>();
    std::string pattern = (parent / "tc-coreml-XXXXXX").string();
    require(mkdtemp(pattern.data()) != nullptr, "Core ML generation: cannot create private directory");
    state->root = pattern;
    state->created = directory(state->root);
    for (const auto &[name, ignored] : initial.directories) {
        (void)ignored;
        if (!name.empty()) fs::create_directories(state->root / name);
    }
    std::vector<char> buffer(1 << 20);
    std::vector<streaming::SourceFileIdentity> targets;
    for (const auto &file : source_lease->descriptor().files) {
        cancelled(flag);
        auto input = source_lease->duplicate_fd(file.logical_id);
        auto target = state->root / file.logical_id;
        OwnedSourceFd output(open(target.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600));
        require(bool(output), "Core ML generation: cannot create file");
        uint64_t offset = 0;
        while (offset < file.bytes) {
            cancelled(flag);
            ssize_t got = pread(input.get(), buffer.data(), std::min<uint64_t>(buffer.size(), file.bytes - offset), off_t(offset));
            if (got < 0 && errno == EINTR) continue;
            require(got > 0, "Core ML generation: source read failed");
            ssize_t done = 0;
            while (done < got) {
                cancelled(flag);
                auto count = write(output.get(), buffer.data() + done, size_t(got - done));
                if (count < 0 && errno == EINTR) continue;
                require(count > 0, "Core ML generation: target write failed");
                done += count; state->copied += uint64_t(count);
            }
            offset += uint64_t(got);
        }
        require(fchmod(output.get(), 0400) == 0, "Core ML generation: cannot seal file");
        streaming::SourceFileIdentity imported;
        imported.logical_id = file.logical_id; imported.path = target;
        imported.content_digest = file.content_digest;
        targets.push_back(std::move(imported));
    }
    source_lease->revalidate_after_drain();
    same_inventory(initial, inventory(src, flag));
    // No open writer remains when independent native content verification starts.
    state->lease = streaming::SourceLease::capture_verified(std::move(targets), flag);
    require(state->lease->artifact_digest() == source_lease->artifact_digest(), "Core ML generation: copied content mismatch");
    for (const auto &[name, ignored] : initial.directories) {
        (void)ignored;
        fs::permissions(state->root / name, fs::perms::owner_read | fs::perms::owner_exec);
    }
    state->snapshot = inventory(state->root, flag);
    cancelled(flag);
    auto result = std::shared_ptr<const CoreMLGeneration>(new CoreMLGeneration(std::move(state)));
    result->revalidate();
    return result;
}
} // namespace tc::z_image
