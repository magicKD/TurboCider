#pragma once
// Research-only fixed runtime-weight QKV MatMul. No checkpoint constants.
// Reuses native GPU staging/transfer and Private Program ownership contracts.
#include "../../../native/backends/private/ane_program.hpp"
#include "../../../native/backends/private/ane_mil.hpp"
#include "../../../native/backends/mlx.hpp"
#include <mlx/allocator.h>
#include <array>
#include <sstream>

namespace tc::qwen21::base_qkv_lab {
namespace pa = ane::private_api;
constexpr int hidden=4096, outputs=12288, rows=1024;

inline std::string tiled_mil(int chunk) {
    require(chunk==256 || chunk==512,"fixed QKV chunk must be256 or512");
    auto shape=[](int r,int c){return "[1, 1, "+std::to_string(r)+", "+std::to_string(c)+"]";};
    auto tensor=[&](const char *type,int r,int c){return std::string("tensor<")+type+", "+shape(r,c)+">";};
    auto buffer=[&](const char *type,int r,int c){const auto plane=std::to_string(uint64_t(r)*c);
        return std::string("tensor_buffer<")+type+", shape="+shape(r,c)+", strides=["+plane+", "+plane+", "+
            std::to_string(c)+", 1], interleave_factors=[1, 1, 1, 1]>";};
    std::ostringstream s;
    s<<"program(1.3)\n{\n    func main_ane<ios18>("<<buffer("int8",hidden,chunk)<<" x, "
     <<buffer("int8",outputs,hidden)<<" w) {\n";
    auto line=[&](const std::string &v){s<<"        "<<v<<";\n";};
    line(tensor("int8",hidden,chunk)+" xq = tensor_buffer_to_tensor<ios17>(input = x)");
    line(tensor("int8",outputs,hidden)+" wq = tensor_buffer_to_tensor<ios17>(input = w)");
    line(tensor("fp16",hidden,chunk)+" xd = dequantize(input = xq, scale = fp16(0x1p-7))");
    std::string pieces;
    for(int n=0;n<outputs;n+=512) {
        const auto id=std::to_string(n);
        line(tensor("int8",512,hidden)+" wq"+id+" = slice_by_size(x = wq, begin = tensor<int32, [4]>([0, 0, "+id+
             ", 0]), size = tensor<int32, [4]>("+shape(512,hidden)+"))");
        line(tensor("fp16",512,hidden)+" wd"+id+" = dequantize(input = wq"+id+", scale = fp16(0x1p-7))");
        line(tensor("fp16",512,chunk)+" p"+id+" = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wd"+id+", y = xd)");
        if(!pieces.empty())pieces+=", ";pieces+="p"+id;
    }
    line(tensor("fp16",outputs,chunk)+" yn = concat(values = ("+pieces+"), axis = int32(2), interleave = bool(false))");
    const auto plane=std::to_string(uint64_t(outputs)*chunk);
    line(buffer("fp16",outputs,chunk)+" y = tensor_to_tensor_buffer<ios17>(input = yn, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(["+
         plane+", "+plane+", "+std::to_string(chunk)+", 1]))");
    s<<"    } -> (y);\n}\n";return s.str();
}

inline ane::DeviceMatrixView device_view(const Tensor &a,int r,int c,int begin=0) {
    require(a.is_available() && a.flags().row_contiguous && a.dtype()==mx::bfloat16 && a.buffer().ptr() &&
        begin>=0 && r>0 && size_t(begin+r)*c<=a.size(),"ready BF16 device view required");
    return {const_cast<void*>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset())+size_t(begin)*c*2,
        r,c,size_t(c)*2,ane::DType::BF16,std::make_shared<Tensor>(a),a.data_shared_ptr()};
}

struct Projection {
    Tensor value;
    double stage_seconds=0,activation_seconds=0,submit_seconds=0,join_seconds=0;
};
class FixedMatmul {
    pa::Device device_{true,true,0};
    pa::Surface x_,tx_,w_,sw_,y_;
    std::unique_ptr<pa::Program> program_;
    int chunk_;
    uint64_t timeline_=0;
  public:
    uint64_t predictions=0,stages=0;
    double setup_seconds=0;
    bool compiled_now=false;
    std::string mil,cache_key;
    size_t slot_bytes=0;
    FixedMatmul(int chunk,bool tiled,const std::filesystem::path &cache)
      :x_(device_,hidden,chunk,pa::Element::I8),tx_(device_,1,chunk,pa::Element::FP16),
       w_(device_,outputs,hidden,pa::Element::I8),sw_(device_,outputs,1,pa::Element::FP16),
       y_(device_,outputs,chunk,pa::Element::FP16),chunk_(chunk) {
        require(chunk==256||chunk==512,"fixed QKV chunk must be256 or512");
        const auto started=Clock::now();
        mil=tiled?tiled_mil(chunk):pa::w8_matmul_program({ane::Kind::Matmul,chunk,hidden,outputs,4096,512,false});
        program_=std::make_unique<pa::Program>(device_,mil,std::span<const uint8_t>{},cache);
        auto names=program_->inputs();std::sort(names.begin(),names.end());
        require(names==std::vector<std::string>{"w","x"} && program_->outputs()==std::vector<std::string>{"y"},"fixed QKV ABI differs");
        compiled_now=program_->compiled_now();cache_key=program_->cache_key();
        slot_bytes=x_.bytes()+tx_.bytes()+w_.bytes()+sw_.bytes()+y_.bytes();
        setup_seconds=seconds(started);
    }
    static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
    void stage(const std::array<Tensor,3> &weights) {
        std::vector<pa::QuantStage> tickets;
        try {
            for(int part=0;part<3;++part) {
                const auto &a=weights[part];require(a.shape()==mx::Shape{hidden,hidden},"original Q/K/V geometry required");
                auto v=device_view(a,hidden,hidden);
                ane::DeviceWeightView source{v.buffer,v.buffer_bytes,v.offset_bytes,v.row_stride_bytes,hidden,hidden,
                    ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},v.owner,v.allocation_identity};
                source.immutable_generation=true;
                tickets.push_back(device_.stage_w8(source,{0,hidden,0,hidden,128,20260930,false},
                    w_.slice_rows(part*hidden,hidden),sw_.slice_rows(part*hidden,hidden)));
            }
        } catch(...) {for(auto &t:tickets)try{(void)t.finish();}catch(...){}throw;}
        bool valid=true;std::exception_ptr failure;
        for(auto &t:tickets)try{const auto c=t.finish();valid=valid&&c.ok&&!c.timed_out&&t.validation_flags()==0;}
            catch(...){if(!failure)failure=std::current_exception();}
        if(failure)std::rethrow_exception(failure);
        require(valid,"QKV weight staging failed; no slots reused");++stages;
    }
    template<class GPU>
    Projection run(const Tensor &input,const std::array<Tensor,3> &weights,GPU &&gpu,bool restage) {
        require(input.shape()==mx::Shape{1,rows,hidden},"complete1024-row base QKV input required");
        Projection result{Tensor(0.f)};
        auto start=Clock::now();if(restage)stage(weights);result.stage_seconds=seconds(start);
        start=Clock::now();auto v=device_view(input,chunk_,hidden,rows-chunk_);
        ane::DeviceWeightView source{v.buffer,v.buffer_bytes,v.offset_bytes,v.row_stride_bytes,chunk_,hidden,
            ane::DeviceWeightEncoding::Dense,ane::DType::BF16,32,{},{},v.owner,v.allocation_identity};
        auto activation=device_.stage_w8(source,{0,chunk_,0,hidden,128,20260930,true},x_,tx_);
        const auto ready_input=activation.finish();
        require(ready_input.ok&&!ready_input.timed_out&&activation.validation_flags()==0,"QKV A8 staging failed");
        result.activation_seconds=seconds(start);
        auto allocation=mx::allocator::malloc(size_t(chunk_)*outputs*2);require(allocation.ptr(),"owned QKV output allocation failed");
        Tensor tail(allocation,{1,chunk_,outputs},mx::bfloat16);auto out=device_view(tail,chunk_,outputs);
        const uint64_t ready=timeline_+1,done=timeline_+2;timeline_=done;
        auto transfer=device_.prepare_transfer({},{{y_,out,0,ane::DType::BF16,1.f,sw_,tx_}},ready,done);
        std::pair<std::string,pa::Surface> inputs[]{{"x",x_},{"w",w_}},outputs_view[]{{"y",y_}};
        start=Clock::now();auto ticket=program_->enqueue(inputs,outputs_view,ready,done,transfer.failure_callback());
        std::optional<Tensor> head;
        try {
            transfer.submit();head=gpu(slice_axis(input,1,0,rows-chunk_));
            require(!head->is_available(),"GPU head unexpectedly reused completed output");mx::async_eval(*head);
        } catch(...) {try{(void)ticket.finish();(void)transfer.finish();}catch(...){}throw;}
        result.submit_seconds=seconds(start);start=Clock::now();
        const auto evaluated=ticket.finish();const auto restored=transfer.finish();
        require(evaluated.ok&&evaluated.physical_complete&&!evaluated.retired&&!evaluated.timed_out&&
            restored.ok&&!restored.timed_out&&transfer.validation_flags()==0,"QKV predict/restore failed; no output publication");
        ++predictions;result.join_seconds=seconds(start);
        result.value=mx::concatenate({*head,tail},1);return result;
    }
};
} // namespace tc::qwen21::base_qkv_lab
