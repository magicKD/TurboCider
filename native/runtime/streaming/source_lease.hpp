#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tc::streaming {

class ArtifactVerificationRequired final : public std::invalid_argument {
  public:
    ArtifactVerificationRequired()
        : std::invalid_argument("streaming_source_lease: artifact_verification_required") {}
};

// Metadata identity for one logical model artifact.  The stat fields identify
// the file generation that was opened for this request; they are not a
// substitute for an import-time content digest.
struct SourceFileIdentity final {
    std::string logical_id;
    // `path` is the normalized absolute name supplied by the model manifest.
    // It may be a symlink. `canonical_target_path` is the target resolved when
    // the request-scoped descriptor was opened. Readers use the held fd, while
    // revalidation checks that the named path still resolves to this target.
    std::filesystem::path path;
    std::filesystem::path canonical_target_path;
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t bytes = 0;
    int64_t mtime_ns = 0;
    int64_t ctime_ns = 0;
    bool stat_identity_complete = false;
    std::string header_digest;
    std::string manifest_digest;
    std::string content_digest;

    bool operator==(const SourceFileIdentity &) const = default;
};

struct SourceLeaseDescriptor final {
    std::string source_snapshot_digest;
    std::vector<SourceFileIdentity> files;

    bool operator==(const SourceLeaseDescriptor &) const = default;
};

// Move-only ownership wrapper used when a reader needs a duplicate of the
// request lease.  The public runtime never serializes this object.
class OwnedSourceFd final {
  public:
    explicit OwnedSourceFd(int fd = -1) noexcept : fd_(fd) {}
    OwnedSourceFd(const OwnedSourceFd &) = delete;
    OwnedSourceFd &operator=(const OwnedSourceFd &) = delete;
    OwnedSourceFd(OwnedSourceFd &&other) noexcept : fd_(other.release()) {}
    OwnedSourceFd &operator=(OwnedSourceFd &&other) noexcept;
    ~OwnedSourceFd();

    int get() const noexcept { return fd_; }
    int release() noexcept {
        const int value = fd_;
        fd_ = -1;
        return value;
    }
    explicit operator bool() const noexcept { return fd_ >= 0; }

  private:
    int fd_ = -1;
};

// Capture current stat identity for a set of logical artifacts.  This is a
// metadata-only operation and closes all temporary descriptors before return.
// Callers should add model/index header or import-time content digests before
// passing the descriptor to open_and_verify().
SourceLeaseDescriptor capture_source_lease_descriptor(
    std::vector<SourceFileIdentity> files);

class SourceLease final {
  public:
    // Production capture path. Opens each named artifact exactly once and
    // builds the descriptor from those same held descriptors. Metadata parsers
    // and readers subsequently use duplicate_fd(), eliminating a capture/open
    // TOCTOU gap.
    static std::shared_ptr<const SourceLease> capture(
        std::vector<SourceFileIdentity> files);

    // Explicit content verification from the same held descriptors. Reuses
    // only native, process-local hashes of unchanged file generations; caller
    // digests are expectations to verify, never proof. First capture reads all
    // bytes and is cancellable. No serialized descriptor grants this trust.
    static std::shared_ptr<const SourceLease> capture_verified(
        std::vector<SourceFileIdentity> files,
        const std::atomic<bool> *cancelled = nullptr);

    // Request-time capture: consumes native generation-bound proofs without
    // reading model contents. Missing/evicted/changed proofs fail explicitly.
    // The current cache is process-local; this is not a persistent import API.
    static std::shared_ptr<const SourceLease> capture_preverified(
        std::vector<SourceFileIdentity> files,
        const std::atomic<bool> *cancelled = nullptr);

    // Metadata-only query admission. Adopt complete native process proofs when
    // available. Once adopted/requested, never fall back to legacy snapshots.
    // The engine owns and serializes require_verified; no payload is hashed.
    static std::shared_ptr<const SourceLease> capture_for_query(
        std::vector<SourceFileIdentity> files, bool &require_verified);

    // Fixture/replay path. Reopens a previously captured descriptor and
    // verifies it exactly. Real public adapters use capture().
    static std::shared_ptr<const SourceLease> open_and_verify(
        const SourceLeaseDescriptor &descriptor);

    SourceLease(const SourceLease &) = delete;
    SourceLease &operator=(const SourceLease &) = delete;
    ~SourceLease();

    const SourceLeaseDescriptor &descriptor() const noexcept;
    const SourceFileIdentity &file(std::string_view logical_id) const;
    OwnedSourceFd duplicate_fd(std::string_view logical_id) const;

    // Named-path check catches replace/rename.  Open-fd check catches changes
    // to a descriptor that remain reachable after a path mutation.
    void revalidate_paths() const;
    void revalidate_open_files() const;
    void revalidate_after_drain() const;

    uint64_t generation() const noexcept;
    std::string_view digest() const noexcept;
    bool has_verified_content() const noexcept;
    // Stable across copies: logical ids, sizes and verified SHA-256 only.
    // Throws for metadata-only capture/replay. digest() remains request binding.
    std::string_view artifact_digest() const;
    uint64_t verification_bytes_read() const noexcept;
    size_t verification_cache_hits() const noexcept;
    size_t file_count() const noexcept;

  private:
    struct State;
    static std::shared_ptr<const SourceLease> capture_impl(
        std::vector<SourceFileIdentity> files, bool verify_content, bool allow_hash,
        const std::atomic<bool> *cancelled);
    explicit SourceLease(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

} // namespace tc::streaming
