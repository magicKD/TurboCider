#pragma once

// Internal implementation, included only by source_lease.cpp. Registration
// receipts are not caller-supplied content proofs or portable manifests.
#include "canonical_encoding.hpp"
#include "source_lease.hpp"
#include "../memory_manifest.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <pwd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tc::streaming::detail {

inline std::string sha256_binary(std::string_view bytes) {
    const auto hex = memory_sha256_hex(bytes);
    auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    std::string result(32, '\0');
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = char((nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    return result;
}

inline std::string registration_hmac(std::string_view key, std::string_view bytes) {
    // HMAC-SHA256; the local key is exactly 32 random bytes. Never emit it.
    std::string inner(64, '\x36'), outer(64, '\x5c');
    for (size_t i = 0; i < key.size(); ++i) {
        inner[i] = char(static_cast<unsigned char>(inner[i]) ^ static_cast<unsigned char>(key[i]));
        outer[i] = char(static_cast<unsigned char>(outer[i]) ^ static_cast<unsigned char>(key[i]));
    }
    inner.append(bytes);
    outer += sha256_binary(inner);
    return memory_sha256_hex(outer);
}

inline bool registration_equal(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    unsigned difference = 0;
    for (size_t i = 0; i < a.size(); ++i)
        difference |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    return difference == 0;
}

class SourceRegistrationStore final {
    OwnedSourceFd directory_, lock_;
    std::string key_;
    static constexpr std::string_view prefix = "tc-source-registration-v1\n";
    static constexpr size_t receipt_bytes = prefix.size() + 3 * 65;

    static bool private_file(int fd, uint64_t bytes, bool allow_unlinked = false) {
        struct stat st{};
        return ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == ::geteuid() &&
            (st.st_mode & 0777) == 0600 && (st.st_nlink == 1 || (allow_unlinked && st.st_nlink == 0)) && st.st_size >= 0 &&
            static_cast<uint64_t>(st.st_size) == bytes;
    }
    static bool read_exact(int fd, std::string &bytes) {
        size_t offset = 0;
        while (offset < bytes.size()) {
            const auto n = ::pread(fd, bytes.data() + offset, bytes.size() - offset, off_t(offset));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            offset += size_t(n);
        }
        return true;
    }
    static std::string random_bytes() {
        OwnedSourceFd random(::open("/dev/urandom", O_RDONLY | O_CLOEXEC));
        std::string bytes(32, '\0');
        size_t offset = 0;
        if (!random) return {};
        while (offset < bytes.size()) {
            const auto n = ::read(random.get(), bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return {};
            offset += size_t(n);
        }
        return bytes;
    }
    bool acquire_lock() const {
        while (::flock(lock_.get(), LOCK_EX) != 0) {
            if (errno != EINTR) return false;
        }
        return true;
    }
    struct Unlock {
        int fd;
        ~Unlock() { ::flock(fd, LOCK_UN); }
    };
    bool atomic_write(const std::string &name, std::string_view bytes) const {
        const auto random = random_bytes();
        if (random.empty()) return false;
        const auto temporary = ".pending-" + memory_sha256_hex(random);
        OwnedSourceFd fd(::openat(directory_.get(), temporary.c_str(),
            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (!fd) return false;
        bool complete = true;
        size_t offset = 0;
        while (offset < bytes.size()) {
            const auto n = ::write(fd.get(), bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { complete = false; break; }
            offset += size_t(n);
        }
        complete = complete && private_file(fd.get(), bytes.size()) && ::fsync(fd.get()) == 0;
        if (complete) complete = ::renameat(directory_.get(), temporary.c_str(), directory_.get(), name.c_str()) == 0;
        if (!complete) ::unlinkat(directory_.get(), temporary.c_str(), 0);
        else ::fsync(directory_.get());
        return complete;
    }
    static std::filesystem::path cache_path() {
        if (const auto *override_path = std::getenv("TURBOCIDER_SOURCE_REGISTRATION_CACHE"))
            return *override_path ? std::filesystem::path(override_path) : std::filesystem::path{};
        // Do not accept a caller's HOME as a content-trust root. The effective
        // user's own cache is the trust boundary (not a hostile same-uid user).
        std::array<char, 16384> buffer{};
        struct passwd entry{}, *found = nullptr;
        if (::getpwuid_r(::geteuid(), &entry, buffer.data(), buffer.size(), &found) ||
            !found || !entry.pw_dir || !*entry.pw_dir) return {};
#if defined(__APPLE__)
        return std::filesystem::path(entry.pw_dir) / "Library/Caches/TurboCider/source-registrations-v1";
#else
        return std::filesystem::path(entry.pw_dir) / ".cache/TurboCider/source-registrations-v1";
#endif
    }
    static std::string generation_identity(const SourceFileIdentity &file) {
        CanonicalEncoder encoder("tc-source-registration-generation-v1");
        encoder.string_field("named_path", file.path.string());
        encoder.string_field("canonical_path", file.canonical_target_path.string());
        encoder.unsigned_field("device", file.device);
        encoder.unsigned_field("inode", file.inode);
        encoder.unsigned_field("bytes", file.bytes);
        encoder.unsigned_field("mtime_ns", static_cast<uint64_t>(file.mtime_ns));
        encoder.unsigned_field("ctime_ns", static_cast<uint64_t>(file.ctime_ns));
        return encoder.sha256();
    }
    static std::string filename(const SourceFileIdentity &file) {
        // One receipt per named path; changed generations replace old records
        // atomically rather than growing an unbounded history of file hashes.
        return memory_sha256_hex(file.path.string()) + ".receipt";
    }

  public:
    SourceRegistrationStore() {
        try {
            const auto path = cache_path();
            if (path.empty() || !path.is_absolute()) return;
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            if (error || (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST)) return;
            directory_ = OwnedSourceFd(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
            struct stat st{};
            if (!directory_ || ::fstat(directory_.get(), &st) || !S_ISDIR(st.st_mode) ||
                st.st_uid != ::geteuid() || (st.st_mode & 0777) != 0700) return;
            // Explicit create-or-open avoids Darwin's transient ENOENT when
            // concurrent O_CREAT|O_NOFOLLOW opens race on a missing lock name.
            int raw_lock = ::openat(directory_.get(), ".lock-v1",
                O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0600);
            if (raw_lock < 0 && errno == EEXIST)
                raw_lock = ::openat(directory_.get(), ".lock-v1", O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
            lock_ = OwnedSourceFd(raw_lock);
            if (!lock_ || !private_file(lock_.get(), 0)) return;
            // Use NONBLOCK only for opening/validating an untrusted cache
            // entry. The verified regular lock descriptor must block normally.
            const int flags = ::fcntl(lock_.get(), F_GETFL);
            if (flags < 0 || ::fcntl(lock_.get(), F_SETFL, flags & ~O_NONBLOCK) != 0) return;
            const bool locked = acquire_lock();
            if (!locked) return;
            Unlock unlock{lock_.get()};
            OwnedSourceFd key(::openat(directory_.get(), "key-v1", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
            if (!key && errno == ENOENT) {
                const auto value = random_bytes();
                if (value.empty() || !atomic_write("key-v1", value)) return;
                key = OwnedSourceFd(::openat(directory_.get(), "key-v1", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
            }
            std::string value(32, '\0');
            if (key && private_file(key.get(), value.size()) && read_exact(key.get(), value)) key_ = std::move(value);
        } catch (...) { key_.clear(); } // Unavailable cache safely falls back to full verification.
    }
    bool available() const noexcept { return !key_.empty(); }
    std::string lookup(const SourceFileIdentity &file) const {
        if (!available() || !acquire_lock()) return {};
        Unlock unlock{lock_.get()};
        OwnedSourceFd fd(::openat(directory_.get(), filename(file).c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
        // A competing publisher may atomically replace this receipt AFTER
        // openat. Its now-unlinked held inode is still safe to authenticate;
        // rejecting nlink==0 would spuriously rewrite otherwise valid records.
        if (!fd || !private_file(fd.get(), receipt_bytes, true)) return {};
        std::string record(receipt_bytes, '\0');
        if (!read_exact(fd.get(), record)) return {};
        const auto body = std::string(prefix) + generation_identity(file) + "\n";
        if (!record.starts_with(body)) return {};
        const auto digest = record.substr(body.size(), 64);
        if (digest.find_first_not_of("0123456789abcdef") != std::string::npos ||
            record[body.size() + 64] != '\n' || record.back() != '\n') return {};
        const auto mac = record.substr(body.size() + 65, 64);
        if (!registration_equal(mac, registration_hmac(key_, record.substr(0, body.size() + 65)))) return {};
        return digest;
    }
    bool publish(const SourceFileIdentity &file) const {
        if (!available() || !acquire_lock()) return false;
        Unlock unlock{lock_.get()};
        const auto body = std::string(prefix) + generation_identity(file) + "\n" + file.content_digest + "\n";
        return atomic_write(filename(file), body + registration_hmac(key_, body) + "\n");
    }
};

} // namespace tc::streaming::detail
