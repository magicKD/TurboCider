#pragma once

#include "../../backends/mlx.hpp"
#include "../../runtime/session.hpp"
#include <array>
#include <future>

namespace tc {

// Explicit BF16 / ConvRot Q8 streaming. Workers read and convert into reusable
// buffers allocated on the inference thread; they never call MLX operations.
class ZImageWeightStream {
    enum class Conversion { none, signed_q8, bf16, scales, biases };
    struct Record {
        std::string name;
        mx::Shape shape;
        uint64_t offset = 0, bytes = 0;
        bool packed = false;
        mx::Dtype dtype = mx::bfloat16;
        uint64_t source_bytes = 0;
        Conversion conversion = Conversion::none;
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
        std::vector<float> scratch;
        std::future<ReadResult> pending;
        int block = -1;
    };
    int fd_ = -1;
    int packed_fd_ = -1;
    uint64_t file_bytes_ = 0;
    int64_t modified_seconds_ = 0, modified_nanos_ = 0;
    std::vector<Record> fixed_records_;
    std::array<std::vector<Record>, 30> blocks_;
    std::vector<Weights> pinned_;
    std::vector<Slot> slots_;
    bool convrot_ = false;
    unsigned prefetch_layers_ = 1;
    std::atomic<bool> &cancelled_;
    BlockResidencyMetrics metrics_;
    int expected_block_ = 0;

    void index(const std::filesystem::path &);
    void pack_suffix(int prefix_channels, const Event &);
    void prepare_convrot(std::vector<Record> &);
    void check_source() const;
    void allocate(Slot &, const std::vector<Record> &);
    ReadResult read(const std::vector<Read> &) const;
    ReadResult fill(Slot &, const std::vector<Record> &) const;
    void record(ReadResult);
    Weights bind(const Slot &, int) const;
    void prefetch(int);

  public:
    ZImageWeightStream(const std::filesystem::path &, uint64_t budget,
                       uint64_t activation_reserve, Weights &fixed,
                       const Event &, std::atomic<bool> &, int prefix_channels = 0,
                       unsigned prefetch_layers = 1);
    ~ZImageWeightStream();
    ZImageWeightStream(const ZImageWeightStream &) = delete;
    void reset_metrics();
    void begin_pass();
    // Call only after evaluating the preceding block's output. Reusing a slot
    // before GPU completion would silently overwrite weights still in flight.
    Weights acquire(int block);
    const BlockResidencyMetrics &metrics() const { return metrics_; }
};

} // namespace tc
