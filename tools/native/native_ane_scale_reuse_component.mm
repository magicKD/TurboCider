// Generated tiny GPU-only compact row-scale reuse. No ANE/model/timing loop.
// The default and --cpu-only paths never construct a device. --tiny-device is opt-in.
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
constexpr size_t guard=64, gate_bytes=2*shape.ane_channels, up_bytes=gate_bytes, down_bytes=2*shape.hidden;
constexpr size_t gate_offset=guard, up_offset=gate_offset+gate_bytes+guard;
constexpr size_t down_offset=up_offset+up_bytes+guard, cache_bytes=down_offset+down_bytes+guard;
constexpr unsigned char canary=0x6d;
size_t checks=0;
void check(bool value,const char* message) { ++checks;if(!value)throw std::runtime_error(message); }
uint16_t narrow(float value) {
    const uint32_t bits=std::bit_cast<uint32_t>(value);
    return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float value(const mg::MatrixView& m,uint32_t row,uint32_t column) {
    uint16_t bits=0;
    const auto* p=static_cast<const std::byte*>(m.storage.contents())+size_t(row)*m.row_stride_bytes+column*2;
    std::memcpy(&bits,p,sizeof(bits));return std::bit_cast<float>(uint32_t(bits)<<16);
}
void fill(const mg::MatrixView& m,unsigned variant) {
    for(uint32_t r=0;r<m.rows;++r)for(uint32_t c=0;c<m.columns;++c) {
        const uint32_t row=r%shape.ane_channels;
        float x=row==0?0.f:row==1?1e-20f:float(int((r*17+c*31)%257)-128)/4096.f;
        if(row==3 && (c==m.columns-1 || c==31 || c==32))x=c==m.columns-1?16.f:-2.f;
        if(variant)x*=2.f; // Changes all three nonzero scale sets, not only code signs.
        const uint16_t bits=narrow(x);
        auto* p=static_cast<std::byte*>(m.storage.contents())+size_t(r)*m.row_stride_bytes+c*2;
        std::memcpy(p,&bits,sizeof(bits));
    }
}
mg::MatrixView allocate(mg::Backend& backend,uint32_t rows,uint32_t columns,std::string_view label) {
    const size_t pitch=size_t(columns+16)*2;
    auto buffer=backend.allocate(guard+size_t(rows)*pitch+guard,label);
    check(buffer.contents()!=nullptr,"fixture source is not Shared");
    std::memset(buffer.contents(),canary,buffer.size_bytes());
    return {buffer.view(guard,size_t(rows-1)*pitch+size_t(columns)*2,mg::ScalarType::BF16),rows,columns,pitch};
}
using Bytes=std::vector<std::byte>;
Bytes snapshot(const mg::Buffer& buffer) {
    check(buffer.contents()!=nullptr,"fixture snapshot is not Shared");
    const auto* p=static_cast<const std::byte*>(buffer.contents());return {p,p+buffer.size_bytes()};
}
void unchanged(const mg::Buffer& buffer,const Bytes& before,const char* message) {
    check(buffer.size_bytes()==before.size() && !std::memcmp(buffer.contents(),before.data(),before.size()),message);
}
struct Source {
    mg::MatrixView fused,down,gate,up;
    Source(mg::Backend& b,unsigned variant)
        :fused(allocate(b,2*shape.width,shape.hidden,variant?"scale-reuse-B-fused":"scale-reuse-A-fused")),
         down(allocate(b,shape.hidden,shape.width,variant?"scale-reuse-B-down":"scale-reuse-A-down")) {
        fill(fused,variant);fill(down,variant); // Written once; immutable for this Trial's entire lifetime.
        const size_t extent=size_t(shape.width-1)*fused.row_stride_bytes+shape.hidden*2;
        const auto& buffer=fused.storage.buffer();const auto offset=fused.storage.offset_bytes();
        gate={buffer.view(offset,extent,mg::ScalarType::BF16),shape.width,shape.hidden,fused.row_stride_bytes};
        up={buffer.view(offset+shape.width*fused.row_stride_bytes,extent,mg::ScalarType::BF16),
            shape.width,shape.hidden,fused.row_stride_bytes};
    }
};
struct Compact {
    mg::Buffer backing;
    mg::AneCompactWeightScales views;
    Compact(mg::Backend& b,std::string_view label):backing(b.allocate(cache_bytes,label)) {
        check(backing.contents()!=nullptr,"compact fixture is not Shared");
        std::memset(backing.contents(),canary,backing.size_bytes());
        views={backing.view(gate_offset,gate_bytes,mg::ScalarType::FP16),
            backing.view(up_offset,up_bytes,mg::ScalarType::FP16),
            backing.view(down_offset,down_bytes,mg::ScalarType::FP16)};
    }
    void guards() const {
        const auto* p=static_cast<const unsigned char*>(backing.contents());
        for(size_t i=0;i<backing.size_bytes();++i) {
            const bool logical=(i>=gate_offset&&i<gate_offset+gate_bytes) ||
                (i>=up_offset&&i<up_offset+up_bytes) || (i>=down_offset&&i<down_offset+down_bytes);
            if(!logical)check(p[i]==canary,"compact copy overwrote a cache guard");
        }
    }
};
struct Outputs {
    std::vector<ag::Surface> owners;
    mg::AneWeightBankViews bank;
    mg::AneActivationViews activation;
    std::array<mg::ImportedAneSurface,8> views;
    Outputs(mg::Backend& b,ag::Device& device) {
        owners.reserve(8);
        const auto add=[&](uint32_t rows,uint32_t columns,ag::Element type) {
            owners.emplace_back(device,rows,columns,type);return mg::import_ane_surface(b,owners.back());
        };
        bank.g=add(shape.ane_channels,shape.hidden,ag::Element::I8);bank.sg=add(shape.ane_channels,1,ag::Element::FP16);
        bank.u=add(shape.ane_channels,shape.hidden,ag::Element::I8);bank.su=add(shape.ane_channels,1,ag::Element::FP16);
        bank.d=add(shape.hidden,shape.ane_channels,ag::Element::I8);bank.sd=add(shape.hidden,1,ag::Element::FP16);
        activation.x=add(shape.hidden,shape.rows,ag::Element::I8);activation.tx=add(1,shape.rows,ag::Element::FP16);
        views={bank.g,bank.sg,bank.u,bank.su,bank.d,bank.sd,activation.x,activation.tx};reset();
    }
    void reset() {
        for(const auto& v:views)std::memset(v.buffer.contents(),canary,v.buffer.size_bytes());
    }
    void padding() const {
        for(const auto& v:views) {
            const size_t logical_bytes=size_t(v.lease.cols)*(v.lease.element==ag::Element::FP16?2:1);
            const auto* p=static_cast<const unsigned char*>(v.buffer.contents());
            for(size_t i=0;i<v.buffer.size_bytes();++i) {
                const bool logical=i/v.lease.row_stride_bytes<v.lease.rows && i%v.lease.row_stride_bytes<logical_bytes;
                if(!logical)check(p[i]==canary,"staging overwrote bank/A8 padding");
            }
        }
    }
};
using BankBytes=std::array<Bytes,8>;
BankBytes snapshot(const Outputs& outputs) {
    BankBytes result;for(size_t i=0;i<result.size();++i)result[i]=snapshot(outputs.views[i].buffer);return result;
}
void equal(const Outputs& output,const BankBytes& reference) {
    for(size_t i=0;i<reference.size();++i)unchanged(output.views[i].buffer,reference[i],"cold/hit bank scales/codes/A8 bytes differ");
    output.padding();
}
void gathered(const Compact& compact,const Outputs& outputs) {
    const std::array<const mg::BufferView*,3> cache{&compact.views.gate,&compact.views.up,&compact.views.down};
    const std::array<const mg::ImportedAneSurface*,3> scales{&outputs.bank.sg,&outputs.bank.su,&outputs.bank.sd};
    for(size_t j=0;j<cache.size();++j)for(uint32_t r=0;r<scales[j]->lease.rows;++r)
        check(!std::memcmp(static_cast<const std::byte*>(cache[j]->contents())+r*2,
            static_cast<const std::byte*>(scales[j]->view.contents())+size_t(r)*scales[j]->lease.row_stride_bytes,2),
            "compact metadata is not the logical FP16 row scale");
    compact.guards();
}
void oracle(const mg::MatrixView& source,const mg::ImportedAneSurface& codes,
    const mg::ImportedAneSurface& scales,uint32_t first_row,uint32_t first_column,
    uint32_t rows,uint32_t columns,uint32_t block) {
    std::vector<float> rotated(columns);
    for(uint32_t r=0;r<rows;++r) {
        for(uint32_t c=0;c<columns;++c)rotated[c]=value(source,first_row+r,first_column+c);
        for(uint32_t c=0;c<columns;c+=block)tc::ane::rotate_block({rotated.data()+c,block},20260930);
        float peak=0;for(float x:rotated)peak=std::max(peak,std::abs(x));
        const uint16_t expected=tc::ane::normalized_scale(peak);uint16_t actual=0;
        std::memcpy(&actual,static_cast<const std::byte*>(scales.view.contents())+size_t(r)*scales.lease.row_stride_bytes,2);
        check(actual==expected,"cold source scales differ from CPU oracle");
        const auto* code=static_cast<const int8_t*>(codes.view.contents())+size_t(r)*codes.lease.row_stride_bytes;
        for(uint32_t c=0;c<columns;++c)check(code[c]==tc::ane::quantize_rotated(rotated[c],expected),"cold source codes differ from CPU oracle");
    }
}
void oracle(const Source& s,const Outputs& out) {
    oracle(s.gate,out.bank.g,out.bank.sg,shape.first_channel(),0,shape.ane_channels,shape.hidden,128);
    oracle(s.up,out.bank.u,out.bank.su,shape.first_channel(),0,shape.ane_channels,shape.hidden,128);
    oracle(s.down,out.bank.d,out.bank.sd,0,shape.first_channel(),shape.hidden,shape.ane_channels,512);
}
void graph_contract(const mg::CommandGraph& graph,bool registers,bool reuse) {
    const size_t weight_steps=reuse?6:9;
    check(graph.steps().size()==weight_steps+2,"compact stage did not record cold9W/hit6W plus original2A8");
    const std::array<uint32_t,3> rows{shape.ane_channels,shape.ane_channels,shape.hidden};
    for(size_t i=0;i<graph.steps().size();++i) {
        const auto& d=graph.steps()[i].compute;
        check(graph.steps()[i].kind==mg::GraphStep::Kind::Dispatch,"staging unexpectedly recorded an event");
        const bool weight=i<weight_steps;
        const size_t part=weight?i/(reuse?2:3):0;
        const bool copy=weight && i%(reuse?2:3)==(reuse?0:2);
        if(copy) {
            std::array<uint32_t,4> p{};check(d.parameters.size()==sizeof(p),"compact copy POD is not aligned uint4");
            std::memcpy(p.data(),d.parameters.data(),sizeof(p));
            check(p==std::array<uint32_t,4>{rows[part],reuse?1U:32U,reuse?32U:1U,0},"compact half pitches or rows changed");
            check(d.buffers.size()==2 && d.threads.x==64 && d.threads.y==1 && d.threads.z==1 &&
                d.groups.x==(rows[part]+63)/64 && d.groups.y==1 && d.groups.z==1,"compact copy binding/dispatch mismatch");
            continue;
        }
        nc::W8Parameters p{};check(d.parameters.size()==sizeof(p),"W8/A8 parameter ABI mismatch");
        std::memcpy(&p,d.parameters.data(),sizeof(p));
        const bool codes=weight?(reuse || i%3==1):i==weight_steps+1;
        const uint32_t block=weight&&part==2?512:128;
        check(p.block==block && p.encoding==0 && p.dtype==1 && p.transpose==uint32_t(!weight),"W8/A8 BF16/block/transpose changed");
        check(d.threads.x==(weight&&registers?32:block) && d.threads.y==1 && d.threads.z==1,"W8/A8 threadgroup changed");
        check(d.buffers.size()==(weight&&registers?(codes?7:6):(codes?8:7)),"W8/A8 binding count changed");
    }
}
struct Trial {
    mg::Backend backend;
    ag::Device device{false,false};
    mg::PreparedAneStages generic=mg::PreparedAneStages::prepare(backend,shape,mg::AneWeightStaging::Generic,true);
    mg::PreparedAneStages registers=mg::PreparedAneStages::prepare(backend,shape,mg::AneWeightStaging::SylvesterRegister,true);
    Outputs live{backend,device},reference{backend,device};
    Source a{backend,0},b{backend,1};
    mg::MatrixView input=allocate(backend,shape.rows,shape.hidden,"scale-reuse-input");
    Compact cache_a{backend,"scale-reuse-cache-A"},scratch{backend,"scale-reuse-cache-scratch"};
    std::optional<mg::Ticket> pending;
    Trial() { fill(input,0); }
};
uint32_t run(Trial& t,const mg::PreparedAneStages& stage,Outputs& out,const Source& source,
    uint64_t request,const mg::AneCompactWeightScales* reuse,const mg::AneCompactWeightScales* fill_cache) {
    auto status=mg::AneStageStatus::producer(t.backend,{1,1,request,request,request});mg::CommandGraph graph;
    stage.encode_weight_bank(graph,source.gate,source.up,source.down,out.bank,status,reuse,fill_cache);
    stage.encode_activation(graph,t.input,out.activation,status);
    graph_contract(graph,stage.weight_staging()==mg::AneWeightStaging::SylvesterRegister,reuse!=nullptr);
    check(status.encoded_mask()==nc::producer_mask && !status.validated(),"fixture minted an ANE producer permit");
    t.pending.emplace(t.backend.submit(std::move(graph)));
    const auto done=t.pending->wait(std::chrono::seconds(5));
    check(done.ok && done.physically_complete && !done.timed_out && done.submitted_buffers==1 && done.completed_buffers==1,
        "scale reuse producer did not physically succeed");
    uint32_t flags=0;std::memcpy(&flags,status.validation_view().contents(),sizeof(flags));
    check(flags==0,"scale reuse source/codes/A8 validation flags are nonzero");
    t.pending.reset();out.padding();return flags;
}
template<class F> void rejects_atomic(mg::CommandGraph& graph,mg::AneStageStatus& status,F&& operation) {
    const size_t steps=graph.steps().size();const uint32_t mask=status.encoded_mask();bool rejected=false;
    try {operation();}catch(const std::invalid_argument&) {rejected=true;}catch(const tc::ane::CapabilityError&) {rejected=true;}
    check(rejected,"invalid compact layout/alias was accepted");
    check(graph.steps().size()==steps && status.encoded_mask()==mask && !status.validated(),"failed compact encode mutated graph/status");
    uint32_t flags=0;std::memcpy(&flags,status.validation_view().contents(),sizeof(flags));
    check(flags==0,"prevalidation wrote the unsubmitted status word");
}
void prevalidation(Trial& t,const mg::PreparedAneStages& stage) {
    uint64_t id=100;
    const auto weights=[&](mg::CommandGraph& graph,mg::AneStageStatus& status,const mg::AneCompactWeightScales* reuse,
        const mg::AneCompactWeightScales* fill_cache) {stage.encode_weight_bank(graph,t.a.gate,t.a.up,t.a.down,t.live.bank,status,reuse,fill_cache);};
    for(unsigned test=0;test<6;++test) {
        auto status=mg::AneStageStatus::producer(t.backend,{1,1,id,id,id});++id;mg::CommandGraph graph;
        auto bad=t.cache_a.views;
        if(test==1)bad.gate=t.cache_a.backing.view(gate_offset,gate_bytes,mg::ScalarType::BF16);
        if(test==2)bad.gate=t.cache_a.backing.view(gate_offset,gate_bytes-2,mg::ScalarType::FP16);
        if(test==3)bad.up=t.cache_a.backing.view(gate_offset+gate_bytes-2,up_bytes,mg::ScalarType::FP16);
        if(test==4)bad.gate=t.a.gate.storage.buffer().view(t.a.gate.storage.offset_bytes(),gate_bytes,mg::ScalarType::FP16);
        if(test==5)bad.gate=t.live.bank.sg.buffer.view(0,gate_bytes,mg::ScalarType::FP16);
        rejects_atomic(graph,status,[&] {weights(graph,status,test==0?&bad:nullptr,&bad);});
    }
    auto bad=t.cache_a.views;bad.gate=t.input.storage.buffer().view(t.input.storage.offset_bytes(),gate_bytes,mg::ScalarType::FP16);
    for(bool activation_first:{true,false}) {
        auto status=mg::AneStageStatus::producer(t.backend,{1,1,id,id,id});++id;mg::CommandGraph graph;
        if(activation_first) {
            stage.encode_activation(graph,t.input,t.live.activation,status);
            rejects_atomic(graph,status,[&] {weights(graph,status,nullptr,&bad);});
        } else {
            weights(graph,status,nullptr,&bad);
            rejects_atomic(graph,status,[&] {stage.encode_activation(graph,t.input,t.live.activation,status);});
        }
    }
}
void cpu_only() {
    nc::validate_geometry(shape);
    check(cache_bytes==2560 && gate_offset==64 && up_offset==1152 && down_offset==2240,"tiny compact guard oracle changed");
    check(gate_bytes+up_bytes+down_bytes==2304,"tiny compact payload is not exactly FP16 rows");
    const mg::AneCompactWeightScales empty{};check(!empty.gate && !empty.up && !empty.down,"compact views are not empty by default");
    for(uint32_t block:{128U,512U}) {
        check(mg::native_ane_stage_kernels::register_source(block).find("constant bool tc_w8_specialize = true;")!=std::string::npos,
            "register source is not specialized");
        std::vector<float> zero(block);tc::ane::rotate_block(zero,20260930);
        check(tc::ane::normalized_scale(0.f)==0x5800 && tc::ane::quantize_rotated(0.f,0x5800)==0,"zero-row oracle changed");
        std::vector<float> impulse(block);impulse[0]=1.f;tc::ane::rotate_block(impulse,20260930);
        const float expected=float(tc::ane::rotation_sign(20260930,0))/std::sqrt(float(block));
        for(float x:impulse)check(x==expected,"CPU H128/H512 impulse oracle changed");
        check(tc::ane::normalized_scale(1e-20f)==1,"tiny nonzero scale oracle changed");
    }
    std::cout<<"native_ane_scale_reuse_cpu=PASS checks="<<checks<<" device_created=0 gpu_submissions=0 ane_requests=0 models=0\n";
}
void tiny_device() {
    auto t=std::make_unique<Trial>();
    try {
        check(t->device.native_device()==t->backend.native_device(),"fixture allocator device mismatch");
        const std::array<Bytes,5> source_bytes{snapshot(t->a.fused.storage.buffer()),snapshot(t->a.down.storage.buffer()),
            snapshot(t->b.fused.storage.buffer()),snapshot(t->b.down.storage.buffer()),snapshot(t->input.storage.buffer())};
        std::optional<BankBytes> generic_reference;uint64_t request=1;
        for(const auto* stage:{&t->generic,&t->registers}) {
            prevalidation(*t,*stage);t->live.reset();t->reference.reset();bool cache_ready=false;
            const auto cold_a=run(*t,*stage,t->live,t->a,request++,nullptr,&t->cache_a.views);
            // Fixture-only readiness: exact source A, actual physical ticket and zero flags.
            // This does not mint a controller cache token or an ANE admission permit.
            cache_ready=cold_a==0;gathered(t->cache_a,t->live);oracle(t->a,t->live);
            const auto saved_a=snapshot(t->live);const auto saved_cache=snapshot(t->cache_a.backing);
            const auto cold_b=run(*t,*stage,t->live,t->b,request++,nullptr,&t->scratch.views);gathered(t->scratch,t->live);
            const std::array<const mg::BufferView*,3> a{&t->cache_a.views.gate,&t->cache_a.views.up,&t->cache_a.views.down};
            const std::array<const mg::BufferView*,3> b{&t->scratch.views.gate,&t->scratch.views.up,&t->scratch.views.down};
            for(size_t i=0;i<a.size();++i)check(std::memcmp(a[i]->contents(),b[i]->contents(),a[i]->size_bytes())!=0,"B did not overwrite a scale set");
            unchanged(t->cache_a.backing,saved_cache,"cold B changed cache A");check(cache_ready,"cache A used before physical zero-flags fill");
            const auto hit_a=run(*t,*stage,t->live,t->a,request++,&t->cache_a.views,nullptr);
            equal(t->live,saved_a);gathered(t->cache_a,t->live);unchanged(t->cache_a.backing,saved_cache,"hit A mutated compact cache A");
            const auto reference_a=run(*t,*stage,t->reference,t->a,request++,nullptr,&t->scratch.views);
            equal(t->reference,saved_a);equal(t->live,snapshot(t->reference));gathered(t->scratch,t->reference);
            unchanged(t->cache_a.backing,saved_cache,"fresh reference changed immutable cache A");
            check(cold_a==cold_b && cold_a==hit_a && cold_a==reference_a,"cold/hit status words differ");
            if(generic_reference)equal(t->reference,*generic_reference);else generic_reference=snapshot(t->reference);
            std::cout<<"native_ane_scale_reuse_case=PASS recipe="<<(stage->weight_staging()==mg::AneWeightStaging::Generic?"generic":"register")
                <<" sequence=A-B-hitA-coldA cold_weight_dispatches=9 hit_weight_dispatches=6 a8_dispatches=2"
                <<" flags=0 byte_exact=1 cache_guards=1 bank_padding=1 prevalidation_rejections=8\n";
        }
        unchanged(t->a.fused.storage.buffer(),source_bytes[0],"source A fused offset/padding changed");
        unchanged(t->a.down.storage.buffer(),source_bytes[1],"source A full down pitch changed");
        unchanged(t->b.fused.storage.buffer(),source_bytes[2],"source B fused offset/padding changed");
        unchanged(t->b.down.storage.buffer(),source_bytes[3],"source B full down pitch changed");
        unchanged(t->input.storage.buffer(),source_bytes[4],"activation input/padding changed");
        std::cout<<"native_ane_scale_reuse_device=PASS submissions=8 dispatches=82 ane_requests=0 models=0 timing_loops=0 performance_qualified=0\n";
    } catch(...) {
        // Never free Trial/compact/IOSurface/source owners while physical completion is unknown.
        try {if(t->pending && !t->pending->status().physically_complete)(void)t.release();}
        catch(...) {(void)t.release();}
        throw;
    }
}
} // namespace
int main(int argc,char** argv) {
    if(argc>2 || (argc==2 && std::string_view(argv[1])!="--cpu-only" && std::string_view(argv[1])!="--tiny-device")) {
        std::cerr<<"usage: native-ane-scale-reuse-component [--cpu-only|--tiny-device]\n";return 2;
    }
    try {if(argc==2 && std::string_view(argv[1])=="--tiny-device")tiny_device();else cpu_only();}
    catch(const std::exception& e) {std::cerr<<"native_ane_scale_reuse=FAIL "<<e.what()<<'\n';return 1;}
}
