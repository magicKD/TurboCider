#include "gguf_packed_bank.hpp"
#include "gguf_storage.hpp"
#include "canonical_encoding.hpp"
#include "../../core/gguf_affine.hpp"

#include <chrono>
#include <fcntl.h>
#include <set>
#include <thread>

namespace tc::streaming {
namespace {
using Clock = std::chrono::steady_clock;
void cancelled(const std::atomic<bool> *cancel) {
    if (cancel && cancel->load(std::memory_order_acquire)) throw Cancelled();
}
struct Field {
    std::string key;
    mx::Shape shape;
    mx::Dtype dtype;
    uint64_t bytes = 0, capacity = 0;
    std::optional<gguf::AffinePart> part;
};
struct Task { size_t tensor = 0; uint64_t row_bytes = 0; std::vector<Field> fields; };
mx::Shape shape(const std::vector<uint64_t> &dimensions) {
    mx::Shape result;
    for (auto dim : dimensions) {
        require(dim && dim <= INT32_MAX, "gguf_packed_bank: shape exceeds MLX");
        result.push_back(int(dim));
    }
    return result;
}
uint64_t actual(const Tensor &tensor) {
    return mx::allocator::allocator().size(tensor.data_shared_ptr()->buffer);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
}

struct GgufPackedBank::State {
    std::shared_ptr<const SourceLease> lease;
    OwnedSourceFd fd;
    gguf::Directory directory;
    std::vector<Task> tasks;
    MemoryLedger &ledger;
    GgufPackedBankMetrics metrics;
    uint64_t read_bytes;
    bool fused_affine;
    bool used = false, loaded = false;
    std::thread::id owner = std::this_thread::get_id();
    State(std::shared_ptr<const SourceLease> source, MemoryLedger &memory, uint64_t bytes,bool fused)
        : lease(std::move(source)), ledger(memory), read_bytes(bytes),fused_affine(fused) {}
    void owner_check() const {
        require(owner == std::this_thread::get_id(), "gguf_packed_bank: owner violation");
    }
};

GgufPackedBank::GgufPackedBank(std::shared_ptr<const SourceLease> lease, std::string logical,
                              MemoryLedger &ledger, uint64_t read_bytes,bool fused_affine)
    : state_(std::make_unique<State>(std::move(lease), ledger, read_bytes,fused_affine)) {
    auto &s = *state_;
    require(s.lease && s.lease->has_verified_content(), "gguf_packed_bank: native content proof required");
    require(read_bytes && read_bytes <= INT32_MAX && read_bytes % gguf_storage::alignment == 0,
            "gguf_packed_bank: invalid fixed read buffer");
    const auto &file = s.lease->file(logical);
    s.fd = s.lease->duplicate_fd(logical);
#ifdef F_NOCACHE
    require(::fcntl(s.fd.get(), F_NOCACHE, 1) == 0, "gguf_packed_bank: cannot disable source file cache");
#endif
    s.directory = gguf::read_directory(s.fd.get(), file.bytes, {}, true);
    s.metrics.source_sha256 = file.content_digest;
    s.metrics.verification_bytes = s.lease->verification_bytes_read();
    s.metrics.affine_packing_recipe=fused_affine ? "fused-affine-one-pass-v1" : "legacy-affine-three-pass-v1";
    s.metrics.float_import_recipe=fused_affine ? "direct-bf16-read-inplace-f16-v1" : "buffered-row-rne-v1";
#if defined(__aarch64__)
    s.metrics.affine_packing_backend=fused_affine ? "arm_neon" : "legacy_scalar";
#else
    s.metrics.affine_packing_backend=fused_affine ? "scalar" : "legacy_scalar";
#endif
    CanonicalEncoder encoding("gguf-mlx-compat-affine-packed-bank-v1");
    encoding.string_field("affine_packing",s.metrics.affine_packing_recipe);
    encoding.string_field("float_import",s.metrics.float_import_recipe);
    encoding.string_field("affine_packing_backend",s.metrics.affine_packing_backend);
    encoding.string_field("source", file.content_digest);
    encoding.string_field("logical_id", logical);
    encoding.unsigned_field("read_bytes", read_bytes);
    encoding.begin_list("tensors", s.directory.tensors.size());
    std::set<std::string> keys;
    for (size_t index = 0; index < s.directory.tensors.size(); ++index) {
        const auto &tensor = s.directory.tensors[index];
        const auto &type = gguf::type_info(tensor.type);
        Task task; task.tensor = index;
        task.row_bytes = gguf::checked_mul(tensor.columns() / type.elements, type.bytes);
        require(task.row_bytes <= read_bytes, "qe_budget_floor: packed row exceeds fixed import buffer");
        auto field = [&](std::string key, mx::Shape dimensions, mx::Dtype dtype, uint64_t bytes,
                         std::optional<gguf::AffinePart> part = {}) {
            require(keys.insert(key).second, "gguf_packed_bank: duplicate output binding");
            task.fields.push_back({std::move(key),std::move(dimensions),dtype,bytes,gguf_storage::capacity_upper(bytes),part});
        };
        if (type.elements == 1) {
            const auto dtype = tensor.type == 0 ? mx::float32 : mx::float16;
            field(tensor.name, shape(tensor.logical_shape()), dtype,
                  gguf::checked_mul(tensor.elements, dtype == mx::float32 ? 4 : 2));
        } else {
            require(tensor.dimensions.size() == 2 && tensor.name.ends_with(".weight"),
                    "gguf_packed_bank: quantized matrix requires rank2 weight binding");
            const auto prefix = tensor.name.substr(0,tensor.name.size()-7);
            const uint32_t bits = tensor.type == 8 ? 8 : 4;
            const uint64_t groups = tensor.elements / 32;
            field(tensor.name, shape({tensor.rows(),tensor.columns()*bits/32}), mx::uint32,
                  gguf::checked_mul(groups,bits*4), gguf::AffinePart::codes);
            field(prefix+".scales",shape({tensor.rows(),tensor.columns()/32}),mx::float16,
                  gguf::checked_mul(groups,2),gguf::AffinePart::scales);
            field(prefix+".biases",shape({tensor.rows(),tensor.columns()/32}),mx::float16,
                  gguf::checked_mul(groups,2),gguf::AffinePart::biases);
        }
        encoding.string_field("tensor",tensor.name); encoding.unsigned_field("type",tensor.type);
        encoding.unsigned_field("offset",tensor.file_offset); encoding.unsigned_field("bytes",tensor.bytes);
        encoding.begin_list("fields",task.fields.size());
        for (const auto &f : task.fields) {
            encoding.string_field("key",f.key);
            encoding.string_field("dtype",f.dtype==mx::uint32 ? "U32" : f.dtype==mx::float32 ? "F32" : "F16");
            encoding.unsigned_field("bytes",f.bytes); encoding.unsigned_field("capacity",f.capacity);
            encoding.begin_list("shape",f.shape.size());
            for (auto dim : f.shape) encoding.unsigned_field("dim",uint64_t(dim));
            s.metrics.planned_packed_capacity_bytes = gguf::checked_add(s.metrics.planned_packed_capacity_bytes,f.capacity);
            s.metrics.output_bytes = gguf::checked_add(s.metrics.output_bytes,f.bytes);
            ++s.metrics.field_count;
        }
        s.metrics.logical_source_bytes = gguf::checked_add(s.metrics.logical_source_bytes,tensor.bytes);
        s.tasks.push_back(std::move(task));
    }
    s.metrics.plan_digest = encoding.sha256();
    s.metrics.tensor_count = uint32_t(s.tasks.size());
    check_unchanged();
}
GgufPackedBank::~GgufPackedBank() = default;
const gguf::Directory &GgufPackedBank::directory() const { state_->owner_check(); return state_->directory; }
void GgufPackedBank::check_unchanged() const { state_->owner_check(); state_->lease->revalidate_after_drain(); }
GgufPackedBankMetrics GgufPackedBank::metrics() const { state_->owner_check(); return state_->metrics; }

void GgufPackedBank::load(Weights &output, const std::atomic<bool> *cancel, const Event &event) {
    auto &s = *state_; s.owner_check();
    require(!s.used && output.sorted_keys().empty(), "gguf_packed_bank: load requires fresh bank and empty output");
    s.used = true; cancelled(cancel); check_unchanged();
    const auto snapshot = s.ledger.snapshot();
    const uint64_t upper = gguf::checked_add(s.metrics.planned_packed_capacity_bytes,s.read_bytes);
    require(snapshot.committed_bytes <= snapshot.budget_bytes && snapshot.reserved_bytes <= snapshot.budget_bytes-snapshot.committed_bytes &&
            upper <= snapshot.budget_bytes-snapshot.committed_bytes-snapshot.reserved_bytes,
            "qe_budget_floor: prepared packed bank and read buffer exceed managed weight ceiling");
    const auto start = Clock::now();
    // No output binding is published until ALL tensors and source generation
    // checks succeed. Fail/cancel only leaves local arrays, which are released.
    auto buffer = gguf_storage::allocate(s.ledger,s.read_bytes,s.read_bytes,{int(s.read_bytes)},mx::uint8,
                                        MemoryClass::ConversionScratch,s.lease->generation());
    auto *read = buffer.data<std::byte>();
    s.metrics.read_buffer_capacity_bytes = actual(buffer);
    std::vector<std::string> keys; std::vector<Tensor> arrays;
    keys.reserve(s.metrics.field_count); arrays.reserve(s.metrics.field_count);
    for (const auto &task : s.tasks) {
        cancelled(cancel); check_unchanged();
        if (event) event("load_gguf_direct_tensor",int(task.tensor),int(s.tasks.size()));
        cancelled(cancel);
        const auto &tensor = s.directory.tensors[task.tensor];
        const size_t first = arrays.size();
        std::vector<std::byte *> pointers;
        for (const auto &f : task.fields) {
            arrays.push_back(gguf_storage::allocate(s.ledger,f.bytes,f.capacity,f.shape,f.dtype,
                                                   MemoryClass::Weights,s.lease->generation()));
            pointers.push_back(arrays.back().data<std::byte>()); keys.push_back(f.key);
            s.metrics.packed_capacity_bytes = gguf::checked_add(s.metrics.packed_capacity_bytes,actual(arrays.back()));
        }
        const uint64_t rows_per_chunk = s.read_bytes/task.row_bytes;
        const bool bf16_alias=s.fused_affine && tensor.type==30 && task.fields.size()==1 &&
            !task.fields[0].part && task.fields[0].dtype==mx::float16;
        for (uint64_t row = 0; row < tensor.rows();) {
            cancelled(cancel);
            const uint64_t rows = std::min(rows_per_chunk,tensor.rows()-row), bytes = rows*task.row_bytes;
            auto read_start = Clock::now(); uint64_t done = 0;
            // Same-width source-float alias: read into the already-admitted
            // final backing, convert in place, and publish only after the
            // complete bank succeeds. No extra raw/dense floating checkpoint.
            auto *destination=bf16_alias ? pointers[0]+row*task.row_bytes : read;
            while (done < bytes) {
                cancelled(cancel);
                const auto n = ::pread(s.fd.get(),destination+done,size_t(bytes-done),off_t(tensor.file_offset+row*task.row_bytes+done));
                if (n < 0 && errno == EINTR) continue;
                require(n > 0,"gguf_packed_bank: payload read failed or source changed");
                done += uint64_t(n); s.metrics.source_read_bytes = gguf::checked_add(s.metrics.source_read_bytes,uint64_t(n));
            }
            s.metrics.read_seconds += seconds(read_start);
            const gguf::PackedMatrix source{{read,size_t(bytes)},tensor.type,rows,tensor.columns()};
            const auto decode_start = Clock::now();
            if (bf16_alias) gguf::bf16_to_fp16_inplace({destination,size_t(bytes)},cancel);
            else if (s.fused_affine && task.fields.size()==3 && task.fields[0].part==gguf::AffinePart::codes &&
                task.fields[1].part==gguf::AffinePart::scales && task.fields[2].part==gguf::AffinePart::biases) {
                std::array<std::span<std::byte>,3> target;
                for(size_t i=0;i<3;++i) {
                    const auto stride=task.fields[i].bytes/tensor.rows();
                    target[i]={pointers[i]+row*stride,size_t(rows*stride)};
                }
                gguf::pack_native_affine_all(source,target,cancel);
            } else for (size_t i = 0; i < task.fields.size(); ++i) {
                const auto &f = task.fields[i]; const auto stride = f.bytes/tensor.rows();
                auto target = std::span<std::byte>(pointers[i]+row*stride,size_t(rows*stride));
                if (f.part) gguf::pack_native_affine(source,*f.part,target,cancel);
                else gguf::decode_cpu_into(source,{0,rows,0,tensor.columns()},
                    {target,f.dtype==mx::float32 ? gguf::DecodeDType::f32 : gguf::DecodeDType::f16,stride,f.dtype==mx::float32 ? 4u : 2u},cancel);
            }
            const auto elapsed=seconds(decode_start);s.metrics.decode_seconds+=elapsed;
            if (task.fields[0].part) s.metrics.affine_decode_seconds+=elapsed;
            else s.metrics.float_decode_seconds+=elapsed;
            row += rows;
        }
        require(arrays.size()==first+task.fields.size(),"gguf_packed_bank: output field accounting mismatch");
        cancelled(cancel); check_unchanged();
    }
    cancelled(cancel); check_unchanged();
    Weights candidate; candidate.bind_arrays(keys,arrays);
    if (event) event("load_gguf_direct_tensor",int(s.tasks.size()),int(s.tasks.size()));
    cancelled(cancel); check_unchanged();
    output = std::move(candidate); s.loaded = true;
    s.metrics.managed_peak_bytes = s.ledger.snapshot().peak_committed_bytes;
    s.metrics.load_seconds = seconds(start);
}

} // namespace tc::streaming
