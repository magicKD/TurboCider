#include "source_lease.hpp"

#include "canonical_encoding.hpp"
#include "../memory_manifest.hpp"

#include "../../core/common.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

namespace tc::streaming {
namespace {

using Stat = struct stat;

int64_t mtime_ns(const Stat &status) noexcept {
#if defined(__APPLE__)
    return static_cast<int64_t>(status.st_mtimespec.tv_sec) * 1'000'000'000ll +
        static_cast<int64_t>(status.st_mtimespec.tv_nsec);
#else
    return static_cast<int64_t>(status.st_mtim.tv_sec) * 1'000'000'000ll +
        static_cast<int64_t>(status.st_mtim.tv_nsec);
#endif
}

int64_t ctime_ns(const Stat &status) noexcept {
#if defined(__APPLE__)
    return static_cast<int64_t>(status.st_ctimespec.tv_sec) * 1'000'000'000ll +
        static_cast<int64_t>(status.st_ctimespec.tv_nsec);
#else
    return static_cast<int64_t>(status.st_ctim.tv_sec) * 1'000'000'000ll +
        static_cast<int64_t>(status.st_ctim.tv_nsec);
#endif
}

std::filesystem::path normalized_named_path(
        const std::filesystem::path &path, const char *reason) {
    std::error_code error;
    const auto result = std::filesystem::absolute(path, error);
    require(!error && !result.empty(), reason);
    return result.lexically_normal();
}

std::filesystem::path canonical_path(const std::filesystem::path &path,
                                     const char *reason) {
    std::error_code error;
    const auto result = std::filesystem::canonical(path, error);
    require(!error && !result.empty(), reason);
    return result;
}

void lease_require(bool value, const std::string &reason) {
    if (!value)
        throw std::invalid_argument("streaming_source_lease: " + reason);
}

void check_regular(const Stat &status, const char *reason) {
    lease_require(S_ISREG(status.st_mode), reason);
}

void check_expected(const SourceFileIdentity &expected, const Stat &actual,
                    const char *reason) {
    check_regular(actual, reason);
    lease_require(expected.stat_identity_complete,
                  "source stat identity is incomplete");
    lease_require(actual.st_size > 0, "source is empty");
    lease_require(static_cast<uint64_t>(actual.st_dev) == expected.device,
                  reason);
    lease_require(static_cast<uint64_t>(actual.st_ino) == expected.inode,
                  reason);
    lease_require(static_cast<uint64_t>(actual.st_size) == expected.bytes,
                  reason);
    lease_require(mtime_ns(actual) == expected.mtime_ns, reason);
    lease_require(ctime_ns(actual) == expected.ctime_ns, reason);
}

void check_same_open_file(const Stat &opened, const Stat &path_status,
                          const char *reason) {
    check_regular(path_status, reason);
    lease_require(opened.st_dev == path_status.st_dev &&
                      opened.st_ino == path_status.st_ino &&
                      opened.st_size == path_status.st_size &&
                      mtime_ns(opened) == mtime_ns(path_status) &&
                      ctime_ns(opened) == ctime_ns(path_status),
                  reason);
}

void capture_stat_identity(SourceFileIdentity &file, const Stat &status) {
    check_regular(status, "source is not a regular file");
    lease_require(status.st_size > 0, "source is empty");
    file.device = static_cast<uint64_t>(status.st_dev);
    file.inode = static_cast<uint64_t>(status.st_ino);
    file.bytes = static_cast<uint64_t>(status.st_size);
    file.mtime_ns = mtime_ns(status);
    file.ctime_ns = ctime_ns(status);
    file.stat_identity_complete = true;
}

std::string observed_digest(const std::vector<SourceFileIdentity> &files) {
    CanonicalEncoder encoder("tc-streaming-source-lease-v1");
    encoder.begin_list("files", files.size());
    for (const auto &file : files) {
        encoder.string_field("logical_id", file.logical_id);
        encoder.unsigned_field("device", file.device);
        encoder.unsigned_field("inode", file.inode);
        encoder.unsigned_field("bytes", file.bytes);
        encoder.unsigned_field("mtime_ns", static_cast<uint64_t>(file.mtime_ns));
        encoder.unsigned_field("ctime_ns", static_cast<uint64_t>(file.ctime_ns));
        encoder.boolean_field("stat_identity_complete",
                              file.stat_identity_complete);
        encoder.string_field("header_digest", file.header_digest);
        encoder.string_field("manifest_digest", file.manifest_digest);
        encoder.string_field("content_digest", file.content_digest);
    }
    return encoder.sha256();
}

void validate_order_and_uniqueness(
        const std::vector<SourceFileIdentity> &files) {
    lease_require(!files.empty(), "descriptor has no files");
    std::string previous;
    for (const auto &file : files) {
        lease_require(!file.logical_id.empty(), "logical id is empty");
        lease_require(file.logical_id > previous,
                      "logical ids must be unique and sorted");
        previous = file.logical_id;
        lease_require(!file.path.empty(), "artifact path is empty");
        lease_require(!file.canonical_target_path.empty(),
                      "canonical target path is empty");
        lease_require(file.stat_identity_complete,
                      "source stat identity is incomplete");
        lease_require(file.bytes > 0, "source is empty");
    }
}

std::atomic<uint64_t> next_generation{1};

// This cache is never populated from serialized/user-supplied digests. ctime
// and mtime both participate, so restoring mtime alone cannot reuse a hash.
using ContentKey = std::tuple<uint64_t, uint64_t, uint64_t, int64_t, int64_t>;
std::mutex content_cache_mutex;
std::map<ContentKey, std::string> content_cache;

ContentKey content_key(const SourceFileIdentity &file) {
    return {file.device, file.inode, file.bytes, file.mtime_ns, file.ctime_ns};
}

void check_cancelled(const std::atomic<bool> *cancelled) {
    if (cancelled && cancelled->load(std::memory_order_relaxed))
        throw Cancelled();
}

uint64_t allocate_generation() {
    auto value = next_generation.fetch_add(1, std::memory_order_relaxed);
    if (value == 0 || value == std::numeric_limits<uint64_t>::max()) {
        next_generation.store(1, std::memory_order_relaxed);
        value = next_generation.fetch_add(1, std::memory_order_relaxed);
    }
    return value ? value : 1;
}

} // namespace

OwnedSourceFd &OwnedSourceFd::operator=(OwnedSourceFd &&other) noexcept {
    if (this == &other) return *this;
    if (fd_ >= 0) ::close(fd_);
    fd_ = other.release();
    return *this;
}

OwnedSourceFd::~OwnedSourceFd() {
    if (fd_ >= 0) ::close(fd_);
}

struct SourceLease::State {
    struct OpenFile {
        SourceFileIdentity identity;
        OwnedSourceFd fd;
    };
    SourceLeaseDescriptor descriptor;
    std::vector<OpenFile> files;
    std::map<std::string, size_t> indexes;
    uint64_t generation = 0;
    std::string artifact_digest;
    uint64_t verification_bytes_read = 0;
    size_t verification_cache_hits = 0;
};

SourceLeaseDescriptor capture_source_lease_descriptor(
        std::vector<SourceFileIdentity> files) {
    const auto lease = SourceLease::capture(std::move(files));
    return lease->descriptor();
}

std::shared_ptr<const SourceLease> SourceLease::capture(
        std::vector<SourceFileIdentity> files) {
    return capture_impl(std::move(files), false, false, nullptr);
}

std::shared_ptr<const SourceLease> SourceLease::capture_verified(
        std::vector<SourceFileIdentity> files,
        const std::atomic<bool> *cancelled) {
    return capture_impl(std::move(files), true, true, cancelled);
}

std::shared_ptr<const SourceLease> SourceLease::capture_preverified(
        std::vector<SourceFileIdentity> files,
        const std::atomic<bool> *cancelled) {
    return capture_impl(std::move(files), true, false, cancelled);
}

std::shared_ptr<const SourceLease> SourceLease::capture_impl(
        std::vector<SourceFileIdentity> files, bool verify_content, bool allow_hash,
        const std::atomic<bool> *cancelled) {
    check_cancelled(cancelled);
    for (auto &file : files)
        file.path = normalized_named_path(
            file.path, "source path is unavailable");
    std::sort(files.begin(), files.end(),
              [](const auto &left, const auto &right) {
                  return left.logical_id < right.logical_id;
              });
    lease_require(!files.empty(), "descriptor has no files");
    std::string previous;
    for (const auto &file : files) {
        lease_require(!file.logical_id.empty(), "logical id is empty");
        lease_require(file.logical_id > previous,
                      "logical ids must be unique and sorted");
        previous = file.logical_id;
        lease_require(!file.path.empty(), "artifact path is empty");
    }

    auto state = std::make_unique<State>();
    state->files.reserve(files.size());
    for (auto file : files) {
        check_cancelled(cancelled);
        const int raw = ::open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
        lease_require(raw >= 0, "source open failed");
        OwnedSourceFd fd(raw);
        Stat opened{};
        lease_require(::fstat(fd.get(), &opened) == 0,
                      "source fstat failed");
        capture_stat_identity(file, opened);
        file.canonical_target_path = canonical_path(
            file.path, "source path is unavailable");

        Stat named_status{}, target_status{};
        lease_require(::stat(file.path.c_str(), &named_status) == 0,
                      "source named path stat failed");
        lease_require(::stat(file.canonical_target_path.c_str(),
                             &target_status) == 0,
                      "source target path stat failed");
        check_same_open_file(opened, named_status,
                             "source named path changed during capture");
        check_same_open_file(opened, target_status,
                             "source target changed during capture");

        const size_t index = state->files.size();
        state->indexes.emplace(file.logical_id, index);
        state->files.push_back({std::move(file), std::move(fd)});
    }
    if (verify_content) {
        CanonicalEncoder artifact("tc-streaming-artifact-content-v1");
        artifact.begin_list("files", state->files.size());
        for (auto &entry : state->files) {
            check_cancelled(cancelled);
            std::string verified;
            {
                std::lock_guard lock(content_cache_mutex);
                const auto cached = content_cache.find(content_key(entry.identity));
                if (cached != content_cache.end()) verified = cached->second;
            }
            if (verified.empty()) {
                lease_require(allow_hash, "artifact_verification_required");
                verified = memory_sha256_fd(entry.fd.get(), entry.identity.bytes,
                                             cancelled);
                state->verification_bytes_read += entry.identity.bytes;
            } else {
                ++state->verification_cache_hits;
            }
            lease_require(entry.identity.content_digest.empty() ||
                          entry.identity.content_digest == verified,
                          "source content digest mismatch");
            entry.identity.content_digest = verified;
            artifact.string_field("logical_id", entry.identity.logical_id);
            artifact.unsigned_field("bytes", entry.identity.bytes);
            artifact.string_field("sha256", verified);
        }
        state->artifact_digest = artifact.sha256();
    }
    state->descriptor.files.reserve(state->files.size());
    for (const auto &file : state->files)
        state->descriptor.files.push_back(file.identity);
    state->descriptor.source_snapshot_digest =
        observed_digest(state->descriptor.files);
    state->generation = allocate_generation();
    auto result = std::shared_ptr<const SourceLease>(
        new SourceLease(std::move(state)));
    // Includes earlier files in a multi-artifact capture. Never publish hashes
    // if any held fd or named path changed while another artifact was read.
    result->revalidate_after_drain();
    check_cancelled(cancelled);
    if (verify_content) {
        std::lock_guard lock(content_cache_mutex);
        for (const auto &entry : result->state_->files) {
            if (content_cache.size() >= 256) content_cache.clear();
            content_cache[content_key(entry.identity)] = entry.identity.content_digest;
        }
    }
    return result;
}

std::shared_ptr<const SourceLease> SourceLease::open_and_verify(
        const SourceLeaseDescriptor &descriptor) {
    validate_order_and_uniqueness(descriptor.files);
    auto state = std::make_unique<State>();
    state->descriptor = descriptor;
    state->files.reserve(descriptor.files.size());
    for (const auto &expected : descriptor.files) {
        const auto named_path = normalized_named_path(
            expected.path, "source path is unavailable");
        lease_require(named_path == expected.path,
                      "source named path is not normalized");
        const auto target_path = canonical_path(
            named_path, "source path is unavailable");
        lease_require(target_path == expected.canonical_target_path,
                      "source canonical target changed");
        const int raw = ::open(named_path.c_str(), O_RDONLY | O_CLOEXEC);
        lease_require(raw >= 0, "source open failed");
        OwnedSourceFd fd(raw);
        Stat status{};
        if (::fstat(fd.get(), &status) != 0) {
            throw std::invalid_argument(
                "streaming_source_lease: source fstat failed");
        }
        check_expected(expected, status, "source metadata changed");
        Stat named_status{}, target_status{};
        lease_require(::stat(named_path.c_str(), &named_status) == 0,
                      "source named path stat failed");
        lease_require(::stat(target_path.c_str(), &target_status) == 0,
                      "source target path stat failed");
        check_same_open_file(status, named_status,
                             "source named path changed during open");
        check_same_open_file(status, target_status,
                             "source target changed during open");
        SourceFileIdentity observed = expected;
        observed.path = named_path;
        observed.canonical_target_path = target_path;
        observed.device = static_cast<uint64_t>(status.st_dev);
        observed.inode = static_cast<uint64_t>(status.st_ino);
        observed.bytes = static_cast<uint64_t>(status.st_size);
        observed.mtime_ns = mtime_ns(status);
        observed.ctime_ns = ctime_ns(status);
        observed.stat_identity_complete = true;
        const size_t index = state->files.size();
        state->indexes.emplace(observed.logical_id, index);
        state->files.push_back({std::move(observed), std::move(fd)});
    }
    state->descriptor.files.clear();
    state->descriptor.files.reserve(state->files.size());
    for (const auto &file : state->files)
        state->descriptor.files.push_back(file.identity);
    const std::string digest = observed_digest(state->descriptor.files);
    if (!descriptor.source_snapshot_digest.empty())
        lease_require(descriptor.source_snapshot_digest == digest,
                      "source snapshot digest mismatch");
    state->descriptor.source_snapshot_digest = digest;
    state->generation = allocate_generation();
    return std::shared_ptr<const SourceLease>(
        new SourceLease(std::move(state)));
}

SourceLease::SourceLease(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

SourceLease::~SourceLease() = default;

const SourceLeaseDescriptor &SourceLease::descriptor() const noexcept {
    return state_->descriptor;
}

const SourceFileIdentity &SourceLease::file(
        std::string_view logical_id) const {
    lease_require(state_ != nullptr, "lease state is unavailable");
    const auto found = state_->indexes.find(std::string(logical_id));
    lease_require(found != state_->indexes.end(), "unknown logical id");
    return state_->files[found->second].identity;
}

OwnedSourceFd SourceLease::duplicate_fd(std::string_view logical_id) const {
    lease_require(state_ != nullptr, "lease state is unavailable");
    const auto found = state_->indexes.find(std::string(logical_id));
    lease_require(found != state_->indexes.end(), "unknown logical id");
    const int duplicate = ::fcntl(state_->files[found->second].fd.get(),
                                  F_DUPFD_CLOEXEC, 0);
    lease_require(duplicate >= 0, "source fd duplication failed");
    return OwnedSourceFd(duplicate);
}

void SourceLease::revalidate_paths() const {
    lease_require(state_ != nullptr, "lease state is unavailable");
    for (const auto &entry : state_->files) {
        const auto named_path = normalized_named_path(
            entry.identity.path, "source path is unavailable");
        lease_require(named_path == entry.identity.path,
                      "source named path changed");
        const auto target_path = canonical_path(
            named_path, "source path is unavailable");
        lease_require(target_path == entry.identity.canonical_target_path,
                      "source path canonical target changed");
        Stat named_status{}, target_status{};
        lease_require(::stat(named_path.c_str(), &named_status) == 0,
                      "source named path stat failed");
        lease_require(::stat(target_path.c_str(), &target_status) == 0,
                      "source target path stat failed");
        check_expected(entry.identity, named_status,
                       "source path was replaced");
        check_expected(entry.identity, target_status,
                       "source path target was replaced");
    }
}

void SourceLease::revalidate_open_files() const {
    lease_require(state_ != nullptr, "lease state is unavailable");
    for (const auto &entry : state_->files) {
        Stat status{};
        lease_require(::fstat(entry.fd.get(), &status) == 0,
                      "source fd is unavailable");
        check_expected(entry.identity, status, "source fd changed");
    }
}

void SourceLease::revalidate_after_drain() const {
    revalidate_open_files();
    revalidate_paths();
}

uint64_t SourceLease::generation() const noexcept {
    return state_ ? state_->generation : 0;
}

std::string_view SourceLease::digest() const noexcept {
    return state_ ? state_->descriptor.source_snapshot_digest
                  : std::string_view{};
}

size_t SourceLease::file_count() const noexcept {
    return state_ ? state_->files.size() : 0;
}

bool SourceLease::has_verified_content() const noexcept {
    return state_ && !state_->artifact_digest.empty();
}

std::string_view SourceLease::artifact_digest() const {
    lease_require(has_verified_content(), "source content was not verified");
    return state_->artifact_digest;
}

uint64_t SourceLease::verification_bytes_read() const noexcept {
    return state_ ? state_->verification_bytes_read : 0;
}

size_t SourceLease::verification_cache_hits() const noexcept {
    return state_ ? state_->verification_cache_hits : 0;
}

} // namespace tc::streaming
