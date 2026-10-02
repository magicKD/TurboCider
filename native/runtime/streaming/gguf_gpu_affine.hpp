#pragma once

#include "gguf_storage.hpp"
#include <mlx/fast.h>
#include <array>

namespace tc::streaming {
// CPU publishes RAW source bytes only. GPU packing is an owner-thread
// consumer of that raw Ready ticket, not a fake CPU-decoded Ready event.
// All output reservations precede dispatch; their claims follow array Data.
struct GgufGpuAffinePending {
    std::array<uint64_t,4> bytes;
    std::vector<MemoryReservation> reservations;
    std::vector<Tensor> output;
    uint64_t generation=0;
};
inline GgufGpuAffinePending prepare_gguf_gpu_affine(const Tensor &raw,uint32_t type,uint64_t rows,
        uint64_t columns,MemoryLedger &ledger,uint64_t generation) {
    const uint64_t block_bytes=type==8 ? 34 : type==2 ? 18 : type==3 ? 20 : 0;
    require(block_bytes && rows && columns && columns%32==0 && rows<=INT32_MAX && columns<=INT32_MAX &&
        columns/32*block_bytes<=INT32_MAX && raw.nbytes()<=UINT32_MAX,"gguf_gpu_affine: extent exceeds shader range");
    require(raw.dtype()==mx::uint8 && raw.flags().row_contiguous &&
        raw.shape()==mx::Shape{int(rows),int(columns/32*block_bytes)},"gguf_gpu_affine: raw geometry/type mismatch");
    const uint64_t groups=gguf::checked_mul(rows,columns/32);
    require(groups<=INT32_MAX && rows<=INT32_MAX && columns<=INT32_MAX,"gguf_gpu_affine: extent exceeds shader range");
    const int bits=type==8 ? 8 : 4;
    const std::array<uint64_t,4> bytes{gguf::checked_mul(groups,bits*4),gguf::checked_mul(groups,2),
        gguf::checked_mul(groups,2),groups};
    std::vector<MemoryReservation> reservations;
    for (const auto n:bytes) {
        auto reserved=ledger.try_reserve(MemoryClass::ConversionScratch,gguf_storage::capacity_upper(n),"gguf-gpu-affine-current-v1");
        require(reserved.has_value(),"qe_budget_floor: GPU affine output admission denied");
        reservations.push_back(std::move(*reserved));
    }
    static auto kernel=mx::fast::metal_kernel("tc_gguf_raw_affine_v1",{"raw"},{"codes","scales","biases","valid"},R"metal(
        uint group=thread_position_in_grid.x;
        if (group>=GROUPS) return;
        uint begin=group*BLOCK_BYTES;
        ushort sb=ushort(raw[begin]) | (ushort(raw[begin+1])<<8);
        half scale=as_type<half>(sb);
        half bias;
        if (TYPE==3) {
            ushort bb=ushort(raw[begin+2]) | (ushort(raw[begin+3])<<8);
            bias=as_type<half>(bb);
        } else bias=half(float(scale)*(TYPE==8 ? -128.f : -8.f));
        scales[group]=scale;biases[group]=bias;
        float lo=float(bias),hi=float((1u<<BITS)-1u)*float(scale)+lo;
        valid[group]=uchar(isfinite(float(scale)) && isfinite(float(bias)) &&
            max(abs(lo),abs(hi))<=65504.f);
        for (uint word=0;word<BITS;++word) {
            uint value=0;
            for (uint i=0;i<32/BITS;++i) {
                uint k=word*(32/BITS)+i;
                uint code;
                if (TYPE==8) code=uint(raw[begin+2+k])^0x80u;
                else {
                    uint q=uint(raw[begin+(TYPE==3 ? 4 : 2)+(k%16)]);
                    code=k<16 ? (q&15u) : (q>>4);
                }
                value|=code<<(i*BITS);
            }
            codes[group*BITS+word]=value;
        }
    )metal","",false,false,{});
    auto output=kernel({raw},{{int(rows),int(columns*bits/32)},{int(rows),int(columns/32)},
        {int(rows),int(columns/32)},{int(rows),int(columns/32)}},
        {mx::uint32,mx::float16,mx::float16,mx::uint8},{int(groups),1,1},{256,1,1},
        {{"TYPE",int(type)},{"BITS",bits},{"BLOCK_BYTES",int(block_bytes)},{"GROUPS",int(groups)}},{},false,{});
    return {bytes,std::move(reservations),std::move(output),generation};
}
inline std::vector<std::array<Tensor,3>> finish_gguf_gpu_affine(std::vector<GgufGpuAffinePending> &pending) {
    std::vector<Tensor> validations;
    for (auto &task:pending) validations.push_back(mx::all(task.output[3]));
    require(!validations.empty(),"empty GPU affine batch");
    gguf_storage::ExactCapacityCacheScope exact;
    mx::eval(validations);mx::synchronize();
    bool valid=true;
    for (const auto &flag:validations) valid &= flag.item<bool>();
    for (auto &valid:validations) valid.detach();
    validations.clear();mx::synchronize();
    for (auto &task:pending) {
        for (auto &part:task.output) {part.detach();part.set_siblings({},0);}
        task.output[3]=Tensor(uint8_t(0),mx::uint8);task.reservations[3].cancel();
    }
    require(valid,"qe_decode_invalid: GPU affine scale/bias nonfinite or FP16 consumer overflow");
    std::vector<std::array<Tensor,3>> result;
    for (auto &task:pending) {
    auto &output=task.output;auto &reservations=task.reservations;const auto &bytes=task.bytes;
    // Multi-output MLX nodes retain sibling arrays even after evaluation.
    // All writers are complete now; detach each result so one escaped codes
    // view cannot retain metadata/status or the raw slot through that graph.
    for (size_t i=0;i<3;++i) {
        auto data=output[i].data_shared_ptr();
        const uint64_t actual=mx::allocator::allocator().size(data->buffer);
        require(actual>=bytes[i] && actual<=gguf_storage::capacity_upper(bytes[i]),"GPU affine allocation exceeded admitted capacity");
        auto claim=std::make_shared<StorageLease>(reservations[i].commit({0x544347475546ull,
            uint64_t(reinterpret_cast<uintptr_t>(data->buffer.ptr())),actual,task.generation}));
        auto prior=data->d;
        mx::Deleter owned=[prior,claim](mx::allocator::Buffer buffer) {prior(buffer);};
        data->d=std::move(owned);
    }
    result.push_back({std::move(output[0]),std::move(output[1]),std::move(output[2])});
    }
    return result;
}
inline std::array<Tensor,3> gguf_gpu_affine(const Tensor &raw,uint32_t type,uint64_t rows,
        uint64_t columns,MemoryLedger &ledger,uint64_t generation) {
    std::vector<GgufGpuAffinePending> pending;
    pending.push_back(prepare_gguf_gpu_affine(raw,type,rows,columns,ledger,generation));
    auto result=finish_gguf_gpu_affine(pending);return std::move(result.front());
}
} // namespace tc::streaming
