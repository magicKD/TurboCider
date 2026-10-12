#include "source_lease.hpp"

#include "canonical_encoding.hpp"
#include "source_registration_store.hpp"
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
#include <thread>
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
    size_t registration_cache_hits = 0;
    size_t registration_records_written = 0;
    double verification_seconds = 0;
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

std::shared_ptr<const SourceLease> SourceLease::capture_registered(
        std::vector<SourceFileIdentity> files, const std::atomic<bool> *cancelled) {
    return capture_impl(std::move(files), true, true, cancelled, true);
}

std::shared_ptr<const SourceLease> SourceLease::capture_for_query(
        std::vector<SourceFileIdentity> files, bool &require_verified) {
    if (require_verified) return capture_preverified(std::move(files));
    try {
        auto lease = capture_preverified(files);
        require_verified = true;
        return lease;
    } catch (const ArtifactVerificationRequired &) {
        return capture(std::move(files));
    }
}

std::shared_ptr<const SourceLease> SourceLease::capture_impl(
        std::vector<SourceFileIdentity> files, bool verify_content, bool allow_hash,
        const std::atomic<bool> *cancelled, bool registered) {
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
    return seal_state(std::move(state), verify_content, allow_hash, cancelled, nullptr, registered);
}

std::shared_ptr<const SourceLease> SourceLease::verify_held_capture(
        const SourceLease &captured, const std::atomic<bool> *cancelled,
        const std::atomic<bool> *stopped, bool registered) {
    check_cancelled(cancelled);check_cancelled(stopped);
    captured.revalidate_after_drain();
    auto state=std::make_unique<State>();
    state->generation=captured.generation();
    for(const auto &entry:captured.state_->files) {
        state->indexes.emplace(entry.identity.logical_id,state->files.size());
        state->files.push_back({entry.identity,captured.duplicate_fd(entry.identity.logical_id)});
    }
    return seal_state(std::move(state),true,true,cancelled,stopped,registered);
}

std::shared_ptr<const SourceLease> SourceLease::seal_state(
        std::unique_ptr<State> state, bool verify_content, bool allow_hash,
        const std::atomic<bool> *cancelled, const std::atomic<bool> *stopped, bool registered) {
    const auto start = std::chrono::steady_clock::now();
    check_cancelled(cancelled);check_cancelled(stopped);
    std::unique_ptr<detail::SourceRegistrationStore> registration;
    if (verify_content && registered) registration = std::make_unique<detail::SourceRegistrationStore>();
    std::vector<bool> registered_files(state->files.size(), false);
    if (verify_content) {
        CanonicalEncoder artifact("tc-streaming-artifact-content-v1");
        artifact.begin_list("files", state->files.size());
        size_t index = 0;
        for (auto &entry : state->files) {
            check_cancelled(cancelled);check_cancelled(stopped);
            std::string verified;
            {
                std::lock_guard lock(content_cache_mutex);
                const auto cached = content_cache.find(content_key(entry.identity));
                if (cached != content_cache.end()) verified = cached->second;
            }
            if (registration) {
                const auto stored = registration->lookup(entry.identity);
                registered_files[index] = !stored.empty() && (verified.empty() || stored == verified);
                if (verified.empty() && !stored.empty()) {
                    verified = stored;
                    ++state->registration_cache_hits;
                }
            }
            if (verified.empty()) {
                if (!allow_hash) throw ArtifactVerificationRequired();
                verified = memory_sha256_fd(entry.fd.get(), entry.identity.bytes,
                                             cancelled,stopped);
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
            ++index;
        }
        state->artifact_digest = artifact.sha256();
    }
    state->descriptor.files.reserve(state->files.size());
    for (const auto &file : state->files)
        state->descriptor.files.push_back(file.identity);
    state->descriptor.source_snapshot_digest =
        observed_digest(state->descriptor.files);
    if(!state->generation)state->generation = allocate_generation();
    auto result = std::shared_ptr<const SourceLease>(
        new SourceLease(std::move(state)));
    // Includes earlier files in a multi-artifact capture. Never publish hashes
    // if any held fd or named path changed while another artifact was read.
    result->revalidate_after_drain();
    check_cancelled(cancelled);check_cancelled(stopped);
    if (verify_content) {
        {
            std::lock_guard lock(content_cache_mutex);
            for (const auto &entry : result->state_->files) {
                if (content_cache.size() >= 256) content_cache.clear();
                content_cache[content_key(entry.identity)] = entry.identity.content_digest;
            }
        }
        // Only publish after ALL source generations/expectations and cancellation
        // checks pass. Never persist caller metadata or a partly verified set.
        if (registration) {
            for (size_t i = 0; i < result->state_->files.size(); ++i) {
                check_cancelled(cancelled);check_cancelled(stopped);
                if (!registered_files[i] && registration->publish(result->state_->files[i].identity))
                    ++result->state_->registration_records_written;
            }
            result->revalidate_after_drain();
            check_cancelled(cancelled);check_cancelled(stopped);
        }
        result->state_->verification_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    return result;
}

struct PendingSourceVerification::State {
    std::shared_ptr<const SourceLease> metadata,verified;
    const std::atomic<bool> *cancelled;
    std::atomic<bool> stopped{false};
    std::thread worker;
    std::exception_ptr error;
    double seconds=0;
    const std::thread::id owner=std::this_thread::get_id();
    explicit State(const std::atomic<bool> *cancel):cancelled(cancel) {}
    void owner_check() const {
        lease_require(owner==std::this_thread::get_id(),"pending verification owner violation");
    }
};
PendingSourceVerification::PendingSourceVerification(std::vector<SourceFileIdentity> files,
        const std::atomic<bool> *cancelled, bool registered):state_(std::make_unique<State>(cancelled)) {
    check_cancelled(cancelled);
    state_->metadata=SourceLease::capture(std::move(files));
    auto *state=state_.get();
    state->worker=std::thread([state,registered] {
        const auto start=std::chrono::steady_clock::now();
        try {state->verified=SourceLease::verify_held_capture(*state->metadata,state->cancelled,&state->stopped,registered);}
        catch(...) {state->error=std::current_exception();}
        state->seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    });
}
PendingSourceVerification::~PendingSourceVerification() {cancel_and_drain();}
std::shared_ptr<const SourceLease> PendingSourceVerification::metadata() const {
    state_->owner_check();return state_->metadata;
}
std::shared_ptr<const SourceLease> PendingSourceVerification::finish() {
    state_->owner_check();
    if(state_->worker.joinable())state_->worker.join();
    check_cancelled(state_->cancelled);check_cancelled(&state_->stopped);
    if(state_->error)std::rethrow_exception(state_->error);
    lease_require(state_->verified && state_->verified->has_verified_content(),"pending verification produced no native proof");
    state_->verified->revalidate_after_drain();
    return state_->verified;
}
void PendingSourceVerification::cancel_and_drain() noexcept {
    if(!state_)return;
    state_->stopped.store(true,std::memory_order_release);
    if(state_->worker.joinable())state_->worker.join();
}
double PendingSourceVerification::verification_seconds() const {
    state_->owner_check();
    lease_require(!state_->worker.joinable(),"pending verification timing before drain");
    return state_->seconds;
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

size_t SourceLease::registration_cache_hits() const noexcept {
    return state_ ? state_->registration_cache_hits : 0;
}

size_t SourceLease::registration_records_written() const noexcept {
    return state_ ? state_->registration_records_written : 0;
}

double SourceLease::verification_seconds() const noexcept {
    return state_ ? state_->verification_seconds : 0;
}

} // namespace tc::streaming
