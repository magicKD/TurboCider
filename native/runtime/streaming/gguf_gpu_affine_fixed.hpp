#pragma once
#include "gguf_storage.hpp"
#include <array>

namespace tc::streaming {
struct GgufGpuAffineFixedTarget {
    uint32_t type;
    uint64_t rows,columns;
    std::array<Tensor,4> arrays;
};
inline GgufGpuAffineFixedTarget allocate_gguf_gpu_affine_fixed(uint32_t type,uint64_t rows,uint64_t columns,
        MemoryLedger &ledger,uint64_t generation) {
    require((type==2 || type==3 || type==8) && rows && columns && columns%32==0 && rows<=INT32_MAX && columns<=INT32_MAX,
            "invalid fixed GPU affine geometry/type");
    const uint64_t groups=gguf::checked_mul(rows,columns/32),bits=type==8 ? 8 : 4;
    require(groups<=INT32_MAX,"fixed GPU affine groups exceed shader range");
    const uint64_t code_bytes=gguf::checked_mul(groups,bits*4),meta_bytes=gguf::checked_mul(groups,2);
    auto allocate=[&](uint64_t bytes,const mx::Shape &shape,mx::Dtype dtype) {
        return gguf_storage::allocate(ledger,bytes,gguf_storage::capacity_upper(bytes),shape,dtype,
                                     MemoryClass::ConversionScratch,generation);
    };
    return {type,rows,columns,{allocate(code_bytes,{int(rows),int(columns*bits/32)},mx::uint32),
        allocate(meta_bytes,{int(rows),int(columns/32)},mx::float16),
        allocate(meta_bytes,{int(rows),int(columns/32)},mx::float16),allocate(4,{1},mx::uint32)}};
}

// Owner-only synchronous GPU fill. The caller proves prior bank readers are
// retired and holds raw/targets until completion. Failure NEVER publishes.
void fill_gguf_gpu_affine_fixed(const std::vector<Tensor> &raw,
                              const std::vector<GgufGpuAffineFixedTarget *> &targets);

// Dependency-ready, NOT decoded/validated Ready. No dispatch or host wait
// here: the returned multi-output primitive is an explicit input dependency
// of every QMM consumer. All four outputs alias the admitted fixed bank.
// The caller must hold the ticket until compute completion AND validation;
// a failed status prevents publication of this layer's result.
std::vector<Tensor> prepare_gguf_gpu_affine_fixed_dependency(const std::vector<Tensor> &raw,
                              const std::vector<GgufGpuAffineFixedTarget *> &targets);
void finish_gguf_gpu_affine_fixed_dependency(std::vector<Tensor> &outputs);

inline Tensor gguf_gpu_affine_fixed_view(const Tensor &owner) {
    auto data=owner.data_shared_ptr();
    // New argument identity for each content ticket, one physical backing
    // claim. An escaped alias keeps the original Data/StorageLease alive.
    return Tensor(data->buffer,owner.shape(),owner.dtype(),[data](mx::allocator::Buffer) {});
}
} // namespace tc::streaming
