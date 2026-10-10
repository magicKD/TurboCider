// Generated small-component qualification only. The default host contract
// never creates a Metal device. --tiny-device is explicit and has no timing
// loop, model checkpoint, ANE call, or full-image generation.
#include "../../native/backends/mlx.hpp"
#include "../../native/backends/metal_graph/native_lora.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace mg=tc::metal_graph;
namespace mx=tc::mx;
using tc::Tensor;

namespace {
void check(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
}
mg::MatrixView matrix(mg::Backend& backend,const Tensor& value) {
    const auto v=mg::import_mlx_matrix(backend,value);
    return {v.storage,v.rows,v.columns,v.row_stride_bytes};
}
Tensor tensor(const std::vector<float>& values,int rows,int columns,mx::Dtype dtype) {
    return mx::astype(Tensor(values.data(),{rows,columns},mx::float32),dtype);
}
float bf16(uint16_t value) {return std::bit_cast<float>(uint32_t(value)<<16);}
uint64_t hash(const mg::MatrixView& value) {
    check(value.storage.contents()!=nullptr,"fixture requires Shared produced storage");
    uint64_t result=14695981039346656037ULL;
    const auto* begin=static_cast<const unsigned char*>(value.storage.contents());
    for(size_t i=0;i<value.storage.size_bytes();++i){result^=begin[i];result*=1099511628211ULL;}
    return result;
}
void equal(const mg::MatrixView& actual,const mg::MatrixView& reference,const char* name) {
    check(actual.rows==reference.rows && actual.columns==reference.columns,"oracle geometry mismatch");
    size_t differences=0;
    for(uint32_t r=0;r<actual.rows;++r)for(uint32_t c=0;c<actual.columns;++c) {
        uint16_t a,b;
        std::memcpy(&a,static_cast<const std::byte*>(actual.storage.contents())+r*actual.row_stride_bytes+c*2,2);
        std::memcpy(&b,static_cast<const std::byte*>(reference.storage.contents())+r*reference.row_stride_bytes+c*2,2);
        check(std::isfinite(bf16(a)) && std::isfinite(bf16(b)),"nonfinite component output");
        differences+=a!=b;
    }
    std::cout<<"case="<<name<<" elements="<<size_t(actual.rows)*actual.columns
             <<" bf16_bit_differences="<<differences<<'\n';
    check(differences==0,"generated dyadic component differs from actual Weights GPU oracle");
}
void host_contract() {
    using namespace mg::native_lora;
    const Geometry g{65,64,128};
    const Layout x{3,65,130,390,0,mg::ScalarType::BF16,1};
    const Layout out{3,44,88,264,0,mg::ScalarType::BF16,2};
    const Layout low{3,128,512,1536,0,mg::ScalarType::FP32,3};
    const TermLayout t{{16,65,130,2080,0,mg::ScalarType::BF16,4},
        {17,16,32,544,0,mg::ScalarType::BF16,5},-.5f,0,17};
    check(validate_bindings(g,x,std::span(&t,1),out,out,low).terms.size()==1,
          "host generated shape rejected");
    std::cout<<"host_contract=PASS device_execution=0 model_loads=0\n";
}
struct SourceGuard {mg::MatrixView source;uint64_t digest;};
struct DeviceFixture {
    mg::Backend backend;
    mg::NativeLoRA operation;
    std::filesystem::path directory;
    std::vector<std::unique_ptr<tc::Weights>> weights;
    std::vector<Tensor> tensors;
    std::vector<SourceGuard> guards;
    mg::Ticket pending;
    bool submission_uncertain=false;
    size_t dispatches=0,submissions=0,temporary_bytes=0;
    DeviceFixture():operation(mg::NativeLoRA::prepare(backend,{65,64,128})) {
        char pattern[]="/private/tmp/tc-native-lora-component.XXXXXX";
        const char* path=mkdtemp(pattern);
        check(path!=nullptr,"cannot create generated fixture directory");directory=path;
    }
    ~DeviceFixture(){std::error_code error;std::filesystem::remove_all(directory,error);}
    mg::MatrixView source(const Tensor& value) {
        tensors.push_back(value);auto result=matrix(backend,value);
        guards.push_back({result,hash(result)});return result;
    }
    mg::MatrixView allocate(uint32_t rows,uint32_t columns,mg::ScalarType type,const char* label,
                            size_t padding=8) {
        const size_t item=mg::scalar_bytes(type),pitch=(size_t(columns)+padding)*item;
        auto backing=backend.allocate((size_t(rows)+2)*pitch,label);
        check(backing.contents()!=nullptr,"generated allocation is not Shared");
        std::memset(backing.contents(),0x5a,backing.size_bytes());
        return {backing.view(pitch,size_t(rows)*pitch,type),rows,columns,pitch};
    }
    void execute(mg::CommandGraph graph) {
        dispatches+=graph.steps().size();++submissions;
        submission_uncertain=true;
        pending=backend.submit(std::move(graph));
        const auto result=pending.wait(std::chrono::seconds(10));
        if(result.physically_complete)submission_uncertain=false;
        check(result.physically_complete && result.ok,"native component did not physically finish: "+result.error);
    }
    void padding(const mg::MatrixView& view,uint32_t written_columns) {
        const size_t item=mg::scalar_bytes(view.storage.scalar_type());
        const auto* bytes=static_cast<const unsigned char*>(view.storage.buffer().contents());
        check(bytes!=nullptr,"padding inspection needs physical completion");
        for(size_t i=0;i<view.storage.buffer().size_bytes();++i) {
            const bool logical=i>=view.storage.offset_bytes() &&
                i<view.storage.offset_bytes()+size_t(view.rows)*view.row_stride_bytes &&
                (i-view.storage.offset_bytes())%view.row_stride_bytes<size_t(written_columns)*item;
            if(!logical)check(bytes[i]==0x5a,"output/rank workspace padding changed");
        }
    }
    void projection(bool fp32) {
        const int m=fp32?9:3,k=65,full=fp32?64:128,first=fp32?7:47,last=fp32?40:91;
        const std::string prefix=fp32?"transformer_blocks.0.attn.to_q":"transformer_blocks.0.img_mlp.gate_up";
        auto owner=std::make_unique<tc::Weights>();
        // Sparse dyadic inputs permit a strict byte oracle despite GEMM
        // reduction order. FP32 case deliberately uses a non-BF16 factor.
        std::vector<float> xv(size_t(m)*k),wv(size_t(full)*k),biasv(size_t(full),0.f);
        for(int r=0;r<m;++r){xv[size_t(r)*k]=1; xv[size_t(r)*k+k-1]=float(r%3-1)*.5f;}
        for(int n=0;n<full;++n){wv[size_t(n)*k]=fp32?-256.f:float(n%5-2)*.125f;biasv[n]=float(n%3-1)*.00390625f;}
        auto x=tensor(xv,m,k,mx::bfloat16),w=tensor(wv,full,k,mx::bfloat16);
        auto bias=mx::reshape(tensor(biasv,1,full,mx::bfloat16),{full});
        owner->bind_arrays({prefix+".weight",prefix+".bias"},{w,bias});
        std::vector<tc::LoRAAsset> assets;
        const int files=fp32?1:2;
        for(int ordinal=0;ordinal<files;++ordinal) {
            const int rank=fp32?128:(ordinal?3:16),n=fp32?full:64;
            std::unordered_map<std::string,Tensor> file;
            for(int branch=0;branch<(fp32?1:2);++branch) {
                std::vector<float> a(size_t(rank)*k),b(size_t(n)*rank);
                a[0]=fp32?1.00390625f:.125f;
                a[size_t(rank-1)*k+k-1]=fp32?.0078125f:(branch?-.0625f:.0625f);
                for(int row=0;row<n;++row) {
                    b[size_t(row)*rank]=fp32?256.f:float(row%3-1)*.125f;
                    b[size_t(row)*rank+rank-1]=float(row%5-2)*.03125f;
                }
                const auto dtype=fp32?mx::float32:mx::bfloat16;
                const std::string name=fp32?prefix:"transformer_blocks.0.img_mlp."+std::string(branch?"proj":"gate_layer");
                file.emplace(name+".lora_A.weight",tensor(a,rank,k,dtype));
                file.emplace(name+".lora_B.weight",tensor(b,n,rank,dtype));
            }
            const auto path=directory/(std::string(fp32?"fp32-":"bf16-")+std::to_string(ordinal)+".safetensors");
            mx::save_safetensors(path.string(),file);
            temporary_bytes+=std::filesystem::file_size(path);
            check(temporary_bytes<(1u<<20),"generated files exceed 1 MiB bound");
            assets.push_back({path.string(),ordinal?-.5f:1.f,"transformer"});
        }
        std::atomic<bool> cancelled{false};
        check(owner->apply_loras(assets,"transformer",[](const std::string&,int,int){},cancelled,true,true)==size_t(fp32?1:4),
              "actual generated safetensors loader did not resolve expected terms");
        owner->materialize();
        const auto snapshot=owner->runtime_lora_snapshot();
        check(!snapshot.fp16_rank_math && snapshot.bindings.size()==1 && snapshot.bindings[0].prefix==prefix,
              "unexpected actual runtime snapshot");
        auto base=mx::contiguous(owner->project_base_slice(x,prefix,first,last,0,k,false));
        // Retain original Data control blocks before the MLX oracle can donate
        // base/source storage. Native submission is after MLX synchronize.
        mx::eval(x,base);mx::synchronize();
        auto input_view=source(x),base_view=source(base);
        std::vector<mg::OrderedLoRATerm> terms;
        for(const auto& term:snapshot.bindings[0].ordered_terms) {
            const int begin=std::max(first,term.output_start),end=std::min(last,term.output_end);
            if(begin>=end)continue;
            check(term.down_dtype==(fp32?mx::float32:mx::bfloat16) && term.up_dtype==term.down_dtype,
                  "loader factor dtype was narrowed or changed");
            auto b=tc::slice_axis(term.up,0,begin-term.output_start,end-term.output_start);
            mx::eval(b);mx::synchronize();
            terms.push_back({source(term.down),source(b),term.scale,uint32_t(begin-first),uint32_t(end-begin),term.source_adapter_ordinal});
        }
        auto reference=mx::contiguous(owner->apply_runtime_lora_slice(base,x,prefix,first,last,0,k,true));
        auto bias_slice=mx::reshape(tc::slice_axis(bias,0,first,last),{1,last-first});
        mx::eval(reference,bias_slice);mx::synchronize();
        auto reference_view=source(reference),bias_view=source(bias_slice);
        auto output=fp32?base_view:allocate(uint32_t(m),uint32_t(last-first),mg::ScalarType::BF16,"component.lora.output");
        auto low=allocate(uint32_t(m),128,mg::ScalarType::FP32,"component.lora.low");
        mg::CommandGraph graph;
        operation.encode_ordered_corrections(graph,input_view,terms,base_view,output,low,bias_view);
        weights.push_back(std::move(owner));
        execute(std::move(graph));
        equal(output,reference_view,fp32?"r128-fp32-inplace-rank-tail-input-tail":"ordered-bf16-r16-r3-fused-range-bias-last");
        padding(low,fp32?128:16);
        if(!fp32)padding(output,uint32_t(last-first));
        // Exact in-place case intentionally modifies base; all other original
        // source factors, bias, input, and oracle output must remain immutable.
        for(const auto& guard:guards)if(!(fp32 && guard.source.storage.buffer().native_handle()==base_view.storage.buffer().native_handle()))
            check(hash(guard.source)==guard.digest,"immutable produced source changed");
        guards.clear();
    }
    void silu_up() {
        std::vector<float> g(3*35),u(3*35);
        for(size_t i=0;i<g.size();++i){g[i]=float(int(i%33)-16)*.5f;u[i]=float(int(i%5)-2)*.125f;}
        auto gate=tensor(g,3,35,mx::bfloat16),up=tensor(u,3,35,mx::bfloat16);
        auto expected=mx::contiguous(tc::silu(gate)*up);
        mx::eval(gate,up,expected);mx::synchronize();
        auto gv=source(gate),uv=source(up),oracle=source(expected);
        auto output=allocate(3,35,mg::ScalarType::BF16,"component.lora.silu_up");
        mg::CommandGraph graph;operation.encode_silu_up(graph,gv,uv,output);execute(std::move(graph));
        equal(output,oracle,"bf16-silu-times-up");padding(output,35);
        for(const auto& guard:guards)check(hash(guard.source)==guard.digest,"SiLU immutable source changed");
    }
};
}
int main(int argc,char** argv) {
    std::unique_ptr<DeviceFixture> fixture;
    try {
        if(argc==1 || (argc==2 && std::string_view(argv[1])=="--cpu-contract")){host_contract();return 0;}
        check(argc==2 && std::string_view(argv[1])=="--tiny-device","usage: native-lora-component [--cpu-contract|--tiny-device]");
        tc::configure_streams();
        fixture=std::make_unique<DeviceFixture>();
        std::cout<<"device="<<fixture->backend.device_name()<<" recipe="<<mg::NativeLoRA::precision_recipe()<<'\n';
        fixture->projection(false);fixture->projection(true);fixture->silu_up();
        std::cout<<"component=PASS actual_weights_gpu_oracle=1 model_loads=0 ane_execution=0 submissions="
                 <<fixture->submissions<<" dispatches="<<fixture->dispatches
                 <<" generated_file_bytes="<<fixture->temporary_bytes<<" performance_qualified=0\n";
        return 0;
    } catch(const std::exception& error) {
        if(fixture && fixture->submission_uncertain) {
            (void)fixture.release(); // Entire fixture quarantined until process exit.
            std::cerr<<"unknown_physical_completion=1 fixture_quarantined=1 retry=0\n";
        }
        std::cerr<<"component=FAIL "<<error.what()<<'\n';return 1;
    }
}
