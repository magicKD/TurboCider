#pragma once

#include "../../backends/mlx.hpp"
#include "../../runtime/session.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include <array>
#include <future>

namespace tc {

// Explicit BF16 layer streaming. The worker only preads into buffers allocated
// on the inference thread; it never calls MLX's thread-unsafe default stream.
class ZImageWeightStream {
    struct Record {
        std::string name;
        mx::Shape shape;
        uint64_t offset = 0, bytes = 0;
        bool packed = false;
    };
    struct Read {
        uint64_t offset, bytes;
        char *destination;
        int fd = -1;
    };
    struct ReadResult { uint64_t bytes = 0; double seconds = 0; };
    struct Slot {
        std::vector<Tensor> arrays;
        std::vector<char *> pointers;
        std::future<ReadResult> pending;
        int block = -1;
    };
    int fd_ = -1;
    int packed_fd_ = -1;
    std::shared_ptr<const streaming::SourceLease> lease_;
    uint64_t file_bytes_ = 0;
    int64_t modified_seconds_ = 0, modified_nanos_ = 0;
    std::vector<Record> fixed_records_;
    std::array<std::vector<Record>, 30> blocks_;
    std::vector<Weights> pinned_;
    std::array<Slot, 2> slots_;
    // Exact public layouts may use one slot for the lowest memory tier or two
    // slots for the normal double-buffered path.  Keep the backing container
    // fixed-size so the legacy path remains allocation-free, but only expose
    // the compiled number of exact slots to the executor.
    uint32_t exact_slot_count_ = 2;
    std::atomic<bool> &cancelled_;
    BlockResidencyMetrics metrics_;
    int expected_block_ = 0;
    bool exact_layout_ = false;
    bool exact_pool_live_ = false;

    void index(const std::filesystem::path &,
               streaming::OwnedSourceFd source_fd =
                   streaming::OwnedSourceFd());
    void pack_suffix(int prefix_channels, const Event &);
    void check_source() const;
    void allocate(Slot &, const std::vector<Record> &);
    ReadResult read(const std::vector<Read> &,
                    const std::atomic<bool> *worker_cancel = nullptr) const;
    ReadResult fill(Slot &, const std::vector<Record> &,
                    const std::atomic<bool> *worker_cancel = nullptr) const;
    void record(ReadResult);
    Weights bind(const Slot &, int) const;
    void prefetch(int);
    void load_fixed_and_prefix(unsigned, Weights &, const Event &);
    void configure_exact(unsigned pinned_blocks, uint32_t slot_count,
                         uint64_t budget,
                         uint64_t activation_reserve, Weights &fixed,
                         const Event &event);

  public:
    ZImageWeightStream(const std::filesystem::path &, uint64_t budget,
                       uint64_t activation_reserve, Weights &fixed,
                       const Event &, std::atomic<bool> &,
                       int prefix_channels = 0);
    // Exact-layout construction preserves the user-selected prefix. It loads
    // only fixed/prefix weights; the generic StageExecutor remains the sole
    // owner of worker creation, suffix fill dispatch and slot state.
    ZImageWeightStream(const std::filesystem::path &, unsigned pinned_blocks,
                       uint64_t budget, uint64_t activation_reserve,
                       Weights &fixed, const Event &, std::atomic<bool> &);
    ZImageWeightStream(const std::filesystem::path &, unsigned pinned_blocks,
                       uint32_t slot_count, uint64_t budget,
                       uint64_t activation_reserve,
                       Weights &fixed, const Event &, std::atomic<bool> &);
    ZImageWeightStream(std::shared_ptr<const streaming::SourceLease>,
                       unsigned pinned_blocks, uint64_t budget,
                       uint64_t activation_reserve, Weights &fixed,
                       const Event &, std::atomic<bool> &);
    ZImageWeightStream(std::shared_ptr<const streaming::SourceLease>,
                       unsigned pinned_blocks, uint32_t slot_count,
                       uint64_t budget,
                       uint64_t activation_reserve, Weights &fixed,
                       const Event &, std::atomic<bool> &);
    ~ZImageWeightStream();
    ZImageWeightStream(const ZImageWeightStream &) = delete;
    void reset_metrics();
    void begin_pass();
    // Call only after evaluating the preceding block's output. Reusing a slot
    // before GPU completion would silently overwrite weights still in flight.
    Weights acquire(int block);
    void check_unchanged() const;
    void create_exact_pool(uint32_t slots, uint64_t capacity_bytes);
    void destroy_exact_pool() noexcept;
    uint64_t fill_exact(uint32_t slot, uint32_t block,
                        const std::atomic<bool> *worker_cancel);
    Weights bind_exact(uint32_t slot, uint32_t block) const;
    const Weights &prefix_weights(uint32_t block) const;
    const BlockResidencyMetrics &metrics() const { return metrics_; }
};

} // namespace tc
