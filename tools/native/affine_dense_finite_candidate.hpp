#pragma once

#include "runtime/streaming/gguf_storage.hpp"
#include <mlx/fast.h>
#include <array>
#include <limits>

// Standalone research candidate. No model routing, raw-GGUF conversion,
// inverse ConvRot, cache hits, async publication or extra persistent bank.
namespace tc::research::affine_finite {
using Packed=std::array<Tensor,3>;
struct Geometry {
    int rows,columns,words,threads,flags;
    uint64_t dense_bytes,dense_upper,status_bytes,status_upper;
};
inline Geometry geometry(const Packed &source,int bits,int group,int batch=1) {
    const auto &[words,scales,biases]=source;
    require((bits==4 || bits==8) && (group==32 || group==64 || group==128) && (batch==1 || batch==4) &&
        words.ndim()==2 && words.dtype()==mx::uint32 && words.shape(0)>0 && words.shape(1)>0 &&
        scales.ndim()==2 && scales.shape(0)==words.shape(0) && scales.shape(1)>0 && biases.shape()==scales.shape() &&
        (scales.dtype()==mx::float16 || scales.dtype()==mx::bfloat16) && biases.dtype()==scales.dtype() &&
        words.flags().row_contiguous && scales.flags().row_contiguous && biases.flags().row_contiguous,
        "fused affine prepare needs original contiguous typed Q4/Q8 metadata and bounded recipe");
    const uint64_t k=uint64_t(words.shape(1))*(32/bits),count=gguf::checked_mul(uint64_t(words.shape(0)),uint64_t(words.shape(1)));
    const uint64_t bytes=gguf::checked_mul(gguf::checked_mul(uint64_t(words.shape(0)),k),2);
    require(k==uint64_t(scales.shape(1))*group && k<=std::numeric_limits<int>::max() && count<=std::numeric_limits<int>::max() && bytes<=uint64_t(256)<<20,
        "fused affine prepare physical geometry/capacity exceeds bounded target");
    const uint64_t threads=(count+batch-1)/batch,flags=(threads+31)/32,status=flags*sizeof(uint32_t);
    return {words.shape(0),int(k),int(count),int(threads),int(flags),bytes,
        streaming::gguf_storage::capacity_upper(bytes),status,streaming::gguf_storage::capacity_upper(status)};
}

inline Tensor prepare(const Packed &source,int bits,int group,MemoryLedger &ledger,uint64_t generation,
                      int batch=1,int threadgroup=128) {
    require(generation && (threadgroup==128 || threadgroup==256),"fused affine prepare needs generation and bounded threadgroup");
    const auto g=geometry(source,bits,group,batch);
    auto dense_reservation=ledger.try_reserve(MemoryClass::ConversionScratch,g.dense_upper,"affine-finite-dense-v1");
    require(dense_reservation.has_value(),"fused affine dense admission denied including escaped readers");
    auto status_reservation=ledger.try_reserve(MemoryClass::ConversionScratch,g.status_upper,"affine-finite-status-v1");
    require(status_reservation.has_value(),"fused affine finite-status admission denied");
    streaming::gguf_storage::ExactCapacityCacheScope exact;
    static auto kernel=mx::fast::metal_kernel("tc_affine_decode_finite_word_candidate",
        {"words","scales","biases"},{"dense","status"},R"metal(
        constexpr uint PACK=32/BITS;
        uint id=thread_position_in_grid.x;
        bool bad=false;
        #pragma unroll
        for(uint b=0;b<BATCH;++b) {
            uint index=id*BATCH+b;
            if(index<WORDS) {
                uint word=words[index],meta=(index*PACK)/GROUP;
                T scale=scales[meta],bias=biases[meta];
                #pragma unroll
                for(uint j=0;j<PACK;++j) {
                    uchar code=uchar((word>>(j*BITS))&((1u<<BITS)-1));
                    // Original MLX affine-dequantize T expression and final
                    // T storage. Never widen metadata before multiply/add.
                    T value=scale*code+bias;
                    dense[index*PACK+j]=value;
                    bad=bad || !isfinite(float(value));
                }
            }
        }
        // All lanes participate, including padded lanes. Every in-range
        // flag is written exactly once; inactive SIMDs never write a flag.
        uint failures=simd_sum(uint(bad));
        if(thread_index_in_simdgroup==0 && id/32<FLAGS)status[id/32]=failures;
    )metal","",false,false,mx::CompileOptions{});
    auto outputs=kernel({source[0],source[1],source[2]},{{g.rows,g.columns},{g.flags}},
        {source[1].dtype(),mx::uint32},{((g.threads+threadgroup-1)/threadgroup)*threadgroup,1,1},
        {threadgroup,1,1},{{"T",source[1].dtype()},{"BITS",bits},{"GROUP",group},{"BATCH",batch},{"WORDS",g.words},{"FLAGS",g.flags}},
        {},false,{});
    auto maximum_status=mx::max(outputs[1]);
    mx::eval({outputs[0],outputs[1],maximum_status});mx::synchronize();
    require(maximum_status.item<uint32_t>()==0,"fused affine prepare rejects nonfinite decoded coefficients");
    const std::array<uint64_t,2> bytes{g.dense_bytes,g.status_bytes},upper{g.dense_upper,g.status_upper};
    std::array<uint64_t,2> actual{};
    for(size_t i=0;i<outputs.size();++i) {
        actual[i]=mx::allocator::allocator().size(outputs[i].data_shared_ptr()->buffer);
        require(actual[i]>=bytes[i] && actual[i]<=upper[i],"fused affine output backing exceeds admitted upper");
    }
    // Both capacities and finite status are checked before any publication.
    // Claims remain on Data, not this owner; escaped views/lazy readers count.
    for(size_t i=0;i<outputs.size();++i) {
        outputs[i].detach();outputs[i].set_siblings({},0);
        auto data=outputs[i].data_shared_ptr();auto &reservation=i?status_reservation:dense_reservation;
        auto claim=std::make_shared<StorageLease>(reservation->commit({0x544346494e495445ull,
            uint64_t(reinterpret_cast<uintptr_t>(data->buffer.ptr())),actual[i],generation}));
        auto prior=data->d;data->d=[prior,claim](mx::allocator::Buffer buffer){prior(buffer);};
    }
    return outputs[0];
}
} // namespace tc::research::affine_finite
