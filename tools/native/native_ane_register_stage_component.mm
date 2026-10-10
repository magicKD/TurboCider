// Generated tiny GPU-only W staging equivalence. No ANE request or model.
// Default entry is CPU-only; --tiny-device is explicitly scheduled by the owner.
#include "../../native/backends/metal_graph/native_ane_stages.hpp"
#include "../../native/backends/metal_graph/kernels/native_ane_stages.hpp"
#include "../../native/backends/ane_w8a8_math.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <optional>

#pragma clang fp contract(off)

namespace mg=tc::metal_graph;
namespace ag=tc::ane::gpu;
namespace nc=mg::native_ane_stage_contract;
namespace {
constexpr nc::Geometry shape{32,128,1024,512};
constexpr size_t guard=64;
constexpr unsigned char canary=0x6d;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
uint16_t narrow(float value) {
    uint32_t bits=std::bit_cast<uint32_t>(value);
    return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float value(const mg::MatrixView& matrix,uint32_t row,uint32_t column) {
    const auto* p=static_cast<const std::byte*>(matrix.storage.contents())+size_t(row)*matrix.row_stride_bytes;
    return std::bit_cast<float>(uint32_t(reinterpret_cast<const uint16_t*>(p)[column])<<16);
}
void put(const mg::MatrixView& matrix,uint32_t row,uint32_t column,uint16_t bits) {
    auto* p=static_cast<std::byte*>(matrix.storage.contents())+size_t(row)*matrix.row_stride_bytes;
    reinterpret_cast<uint16_t*>(p)[column]=bits;
}
mg::MatrixView allocate(mg::Backend& backend,uint32_t rows,uint32_t columns,std::string_view label) {
    const size_t pitch=size_t(columns+16)*2;
    auto buffer=backend.allocate(guard+size_t(rows)*pitch+guard,label);
    std::memset(buffer.contents(),canary,buffer.size_bytes());
    return {buffer.view(guard,size_t(rows-1)*pitch+size_t(columns)*2,mg::ScalarType::BF16),rows,columns,pitch};
}
void fill(const mg::MatrixView& m) {
    for(uint32_t r=0;r<m.rows;++r)for(uint32_t c=0;c<m.columns;++c) {
        // Every selected region contains zero/tiny/random/last-lane impulses.
        const uint32_t row=r%shape.ane_channels;
        float x=row==0 ? 0.f : row==1 ? 1e-20f :
            float(int((r*17+c*31)%257)-128)/4096.f;
        if(row==3 && (c==m.columns-1 || c==31 || c==32))x=c==m.columns-1?16.f:-2.f;
        put(m,r,c,narrow(x));
    }
}
uint64_t hash(const mg::Buffer& buffer) {
    uint64_t h=1469598103934665603ULL;
    const auto* p=static_cast<const unsigned char*>(buffer.contents());
    for(size_t i=0;i<buffer.size_bytes();++i) {h^=p[i];h*=1099511628211ULL;}
    return h;
}
struct Outputs {
    std::vector<ag::Surface> owners;
    mg::AneWeightBankViews bank;
    mg::AneActivationViews activation;
    std::array<mg::ImportedAneSurface,8> views;
    Outputs(mg::Backend& backend,ag::Device& device) {
        owners.reserve(8);
        const auto add=[&](uint32_t rows,uint32_t columns,ag::Element type) {
            owners.emplace_back(device,rows,columns,type);
            return mg::import_ane_surface(backend,owners.back());
        };
        bank.g=add(shape.ane_channels,shape.hidden,ag::Element::I8);
        bank.sg=add(shape.ane_channels,1,ag::Element::FP16);
        bank.u=add(shape.ane_channels,shape.hidden,ag::Element::I8);
        bank.su=add(shape.ane_channels,1,ag::Element::FP16);
        bank.d=add(shape.hidden,shape.ane_channels,ag::Element::I8);
        bank.sd=add(shape.hidden,1,ag::Element::FP16);
        activation.x=add(shape.hidden,shape.rows,ag::Element::I8);
        activation.tx=add(1,shape.rows,ag::Element::FP16);
        views={bank.g,bank.sg,bank.u,bank.su,bank.d,bank.sd,activation.x,activation.tx};
    }
    void reset_before_submission() {
        for(const auto& v:views)std::memset(v.buffer.contents(),canary,v.buffer.size_bytes());
    }
};
void compare_outputs(const Outputs& a,const Outputs& b) {
    for(size_t i=0;i<a.views.size();++i) {
        const auto& x=a.views[i];const auto& y=b.views[i];
        check(x.buffer.size_bytes()==y.buffer.size_bytes() &&
            !std::memcmp(x.buffer.contents(),y.buffer.contents(),x.buffer.size_bytes()),
            "generic/register codes/scales/A8/backing padding differ");
        const size_t row_bytes=size_t(x.lease.cols)*(x.lease.element==ag::Element::FP16?2:1);
        const auto* p=static_cast<const unsigned char*>(x.buffer.contents());
        for(size_t offset=0;offset<x.buffer.size_bytes();++offset) {
            const bool logical=offset/x.lease.row_stride_bytes<x.lease.rows &&
                offset%x.lease.row_stride_bytes<row_bytes;
            if(!logical)check(p[offset]==canary,"staging overwrote surface padding");
        }
    }
}
void oracle(const mg::MatrixView& source,const mg::ImportedAneSurface& codes,
    const mg::ImportedAneSurface& scales,uint32_t first_row,uint32_t first_column,
    uint32_t rows,uint32_t columns,uint32_t block) {
    std::vector<float> rotated(columns);
    for(uint32_t row=0;row<rows;++row) {
        for(uint32_t column=0;column<columns;++column)rotated[column]=value(source,first_row+row,first_column+column);
        for(uint32_t column=0;column<columns;column+=block)
            tc::ane::rotate_block({rotated.data()+column,block},20260930);
        float peak=0;
        for(float x:rotated)peak=std::max(peak,std::abs(x));
        const uint16_t expected=tc::ane::normalized_scale(peak);
        const auto* scale=static_cast<const std::byte*>(scales.view.contents())+size_t(row)*scales.lease.row_stride_bytes;
        check(*reinterpret_cast<const uint16_t*>(scale)==expected,"staging FP16 scale differs from CPU oracle");
        const auto* code=static_cast<const int8_t*>(codes.view.contents())+size_t(row)*codes.lease.row_stride_bytes;
        for(uint32_t column=0;column<columns;++column)
            check(code[column]==tc::ane::quantize_rotated(rotated[column],expected),"staging W code differs from CPU oracle");
    }
}
void graph_contract(const mg::CommandGraph& graph,bool registers) {
    check(graph.steps().size()==8,"weight/A8 stage did not record exactly eight dispatches");
    for(size_t i=0;i<8;++i) {
        const auto& step=graph.steps()[i];
        check(step.kind==mg::GraphStep::Kind::Dispatch,"stage unexpectedly recorded an event");
        nc::W8Parameters p{};
        check(step.compute.parameters.size()==sizeof(p),"staging parameter ABI mismatch");
        std::memcpy(&p,step.compute.parameters.data(),sizeof(p));
        const bool weight=i<6;
        check(p.encoding==0 && p.dtype==1 && p.transpose==uint32_t(!weight),"staging dense BF16/transpose mismatch");
        check(step.compute.threads.x==(weight&&registers?32:p.block) &&
            step.compute.threads.y==1 && step.compute.threads.z==1,"register did not dispatch exactly one SIMD");
        const size_t buffers=weight&&registers?(i%2?7:6):(i%2?8:7);
        check(step.compute.buffers.size()==buffers,"staging native register binding count mismatch");
    }
}
struct Trial {
    mg::Backend backend;
    ag::Device device{false,false};
    mg::PreparedAneStages generic=mg::PreparedAneStages::prepare(backend,shape);
    mg::PreparedAneStages registers=mg::PreparedAneStages::prepare(backend,shape,mg::AneWeightStaging::SylvesterRegister);
    Outputs a{backend,device},b{backend,device};
    mg::MatrixView fused=allocate(backend,2*shape.width,shape.hidden,"register-fixture-fused-weight");
    mg::MatrixView down=allocate(backend,shape.hidden,shape.width,"register-fixture-down-weight");
    mg::MatrixView input=allocate(backend,shape.rows,shape.hidden,"register-fixture-input");
    std::optional<mg::Ticket> pending;
};
uint32_t run(Trial& trial,const mg::PreparedAneStages& stages,Outputs& outputs,
    const mg::MatrixView& gate,const mg::MatrixView& up,uint64_t request) {
    outputs.reset_before_submission();
    auto status=mg::AneStageStatus::producer(trial.backend,{1,1,request,request,request});
    mg::CommandGraph graph;
    stages.encode_weight_bank(graph,gate,up,trial.down,outputs.bank,status);
    stages.encode_activation(graph,trial.input,outputs.activation,status);
    graph_contract(graph,stages.weight_staging()==mg::AneWeightStaging::SylvesterRegister);
    check(status.encoded_mask()==nc::producer_mask && !status.validated(),"fixture claimed an ANE producer permit");
    trial.pending.emplace(trial.backend.submit(std::move(graph)));
    const auto done=trial.pending->wait(std::chrono::seconds(5));
    check(done.ok && done.physically_complete && !done.timed_out && done.submitted_buffers==1 &&
        done.completed_buffers==1,"generated staging ticket did not physically succeed");
    // Byte testing only: the actual captured GPU ticket above is joined, so
    // Shared flags can be observed. This NEVER mints an ANE admission permit.
    uint32_t flags=0;std::memcpy(&flags,status.validation_view().contents(),sizeof(flags));
    trial.pending.reset();
    return flags;
}
void cpu_only() {
    nc::validate_geometry(shape);
    check(mg::AneWeightStaging{}==mg::AneWeightStaging::Generic,"generic staging is not the default");
    for(uint32_t block:{128U,512U}) {
        check(mg::native_ane_stage_kernels::register_source(block).find("constant bool tc_w8_specialize = true;")!=std::string::npos,
            "requested register source is not specialized");
        std::vector<float> values(block);
        for(uint32_t i=0;i<block;++i)values[i]=float(int(i%17)-8)/1024.f;
        tc::ane::rotate_block(values,20260930);
        float peak=0;for(float x:values)peak=std::max(peak,std::abs(x));
        const auto scale=tc::ane::normalized_scale(peak);
        for(float x:values)(void)tc::ane::quantize_rotated(x,scale);
    }
    std::cout<<"native_ane_register_stage_cpu=PASS device_created=0 model_loads=0\n";
}
void tiny_device() {
    auto trial=std::make_unique<Trial>();
    try {
        check(trial->device.native_device()==trial->backend.native_device(),"fixture surfaces are on a different device");
        const size_t extent=size_t(shape.width-1)*trial->fused.row_stride_bytes+shape.hidden*2;
        const auto offset=trial->fused.storage.offset_bytes();
        const auto& source=trial->fused.storage.buffer();
        const mg::MatrixView gate{source.view(offset,extent,mg::ScalarType::BF16),shape.width,shape.hidden,trial->fused.row_stride_bytes};
        const mg::MatrixView up{source.view(offset+shape.width*trial->fused.row_stride_bytes,extent,mg::ScalarType::BF16),
            shape.width,shape.hidden,trial->fused.row_stride_bytes};
        for(uint32_t test=0;test<4;++test) {
            fill(trial->fused);fill(trial->down);fill(trial->input);
            const uint32_t expected=test==0?0:test<3?1:4;
            if(test==1) {
                put(gate,shape.first_channel()+2,17,0x7f80); // H128 source infinity
                put(trial->down,5,shape.first_channel()+79,0x7f80); // H512 source infinity
            }
            if(test==2) {
                put(up,shape.first_channel()+2,17,0x7fc1); // fused up NaN
                put(trial->down,5,shape.first_channel()+79,0x7fc1); // H512 source NaN
            }
            if(test==3)for(uint32_t c=0;c<shape.hidden;++c)
                put(gate,shape.first_channel()+2,c,narrow(float(tc::ane::rotation_sign(20260930,c))*1e5f));
            const std::array<uint64_t,3> before{hash(source),hash(trial->down.storage.buffer()),hash(trial->input.storage.buffer())};
            const auto generic=run(*trial,trial->generic,trial->a,gate,up,test*2+1);
            const auto registers=run(*trial,trial->registers,trial->b,gate,up,test*2+2);
            check(generic==expected && registers==expected,"generic/register finite validation flags changed");
            compare_outputs(trial->a,trial->b);
            if(test==0) {
                oracle(gate,trial->a.bank.g,trial->a.bank.sg,shape.first_channel(),0,shape.ane_channels,shape.hidden,128);
                oracle(up,trial->a.bank.u,trial->a.bank.su,shape.first_channel(),0,shape.ane_channels,shape.hidden,128);
                oracle(trial->down,trial->a.bank.d,trial->a.bank.sd,0,shape.first_channel(),shape.hidden,shape.ane_channels,512);
            }
            check(before==std::array<uint64_t,3>{hash(source),hash(trial->down.storage.buffer()),hash(trial->input.storage.buffer())},
                "staging mutated source/offset/pitch guards");
            std::cout<<"native_ane_register_stage_case=PASS case="<<test<<" flags="<<generic
                <<" codes_scales_padding_exact=1 source_immutable=1 a8_generic=1\n";
        }
        std::cout<<"native_ane_register_stage_device=PASS device="<<trial->backend.device_name()
            <<" geometry=32x128x1024x512 cases=4 paired_submissions=8 dispatches=64"
            <<" ane_requests=0 models=0 benchmarks=0 performance_qualified=0\n";
    } catch(...) {
        if(trial->pending && !trial->pending->status().physically_complete)(void)trial.release();
        throw;
    }
}
} // namespace

int main(int argc,char** argv) {
    if(argc!=2 || (std::string_view(argv[1])!="--cpu-only" && std::string_view(argv[1])!="--tiny-device")) {
        std::cerr<<"usage: native-ane-register-stage-component --cpu-only|--tiny-device\n";return 2;
    }
    try {
        if(std::string_view(argv[1])=="--cpu-only")cpu_only();else tiny_device();
    } catch(const std::exception& error) {
        std::cerr<<"native_ane_register_stage=FAIL "<<error.what()<<'\n';return 1;
    }
}
