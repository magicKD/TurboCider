#pragma once

#include "../../native/platform/apple/bridge.hpp"
#include "../../native/backends/mlx.hpp"
#include "../../native/backends/mlx_fd_reader.hpp"
#include "../../native/backends/metal_graph/mlx_bridge.hpp"
#include "../../native/backends/metal_graph/native_linear.hpp"
#include "../../native/backends/metal_graph/native_lora.hpp"
#include "../../native/backends/metal_graph/memory_bridge.hpp"
#include "../../native/models/qwen21/runtime_ffn_graphs.hpp"
#include "../../native/models/qwen21/viggle_adapter.hpp"
#include "../../native/runtime/memory_execution.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/runtime/streaming/canonical_encoding.hpp"
#include "turbocider/turbocider.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <bit>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

namespace tc::qwen_lora_replay {

namespace mg = tc::metal_graph;

inline constexpr uint32_t rows=1024,hidden=4096,width=12288,rank=128;
inline constexpr double rrms_limit=.005;
inline constexpr uint64_t source_bytes=(3ULL*hidden*width+3ULL*rank*(hidden+width)+rows*hidden)*2;
inline constexpr const char* capture_id="75991-1226054446574458";
inline constexpr const char* prefix="transformer_blocks.0.img_mlp.";
inline constexpr const char* prompt="A cobalt blue ceramic teapot on a pale wooden table, soft window light, realistic product photography.";

using Time=std::chrono::steady_clock;
inline double seconds(Time::time_point t) {return std::chrono::duration<double>(Time::now()-t).count();}
inline NSString* ns(std::string_view s) {return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding];}
inline std::string fingerprint(const std::string& value) {
    streaming::CanonicalEncoder encoder("tc-ane-calibration-source-v1");
    encoder.string_field("identity",value);return encoder.sha256();
}
inline std::string held_fingerprint(const streaming::SourceLease& lease) {
    const auto& f=lease.file("checkpoint");auto fd=lease.duplicate_fd("checkpoint");struct stat st{};
    require(::fstat(fd.get(),&st)==0 && S_ISREG(st.st_mode),"held checkpoint stat unavailable");
    const auto path=f.canonical_target_path.string();
    return fingerprint(":"+std::to_string(path.size())+":"+path+":"+std::to_string(st.st_dev)+":"+
        std::to_string(st.st_ino)+":"+std::to_string(st.st_size)+":"+std::to_string(st.st_mtimespec.tv_sec)+":"+
        std::to_string(st.st_mtimespec.tv_nsec)+":"+std::to_string(st.st_ctimespec.tv_sec)+":"+std::to_string(st.st_ctimespec.tv_nsec));
}
inline std::string read_small(int fd,uint64_t bytes) {
    require(bytes && bytes<2ULL<<20,"capture completion exceeds metadata bound");
    std::string out(size_t(bytes),'\0');size_t done=0;
    while(done<out.size()) {auto n=::pread(fd,out.data()+done,out.size()-done,off_t(done));
        if(n<0 && errno==EINTR)continue;
        require(n>0,"held capture metadata short read");done+=size_t(n);}
    require(out.find('\0')==std::string::npos,"metadata contains NUL");return out;
}
inline uint64_t integer(NSDictionary* d,NSString* key) {
    id v=d[key];require([v isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)v)!=CFBooleanGetTypeID(),"invalid capture integer");
    const double n=[v doubleValue];require(std::isfinite(n) && n>=0 && n<1e12 && n==std::floor(n),"capture integer outside bound");return uint64_t(n);
}
inline void write_exclusive(const std::filesystem::path& path,NSDictionary* receipt) {
    const auto text=json(receipt);streaming::OwnedSourceFd fd(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));
    require(bool(fd),"receipt must be new in an existing directory");size_t done=0;
    while(done<text.size()) {const auto n=::write(fd.get(),text.data()+done,text.size()-done);
        if(n<0 && errno==EINTR)continue;
        require(n>0,"receipt write failed");done+=size_t(n);}
    require(::fsync(fd.get())==0,"receipt fsync failed");
}
struct CpuView {const void* data;size_t rows,columns,pitch,extent;};
struct Comparison {double rrms=0,max_abs=0;size_t differences=0,elements=0;};
inline Comparison compare(CpuView actual,CpuView expected) {
    for(const auto& v:{actual,expected})require(v.data && v.rows && v.columns && v.columns<=SIZE_MAX/2 &&
        v.pitch>=v.columns*2 && v.rows<=SIZE_MAX/v.pitch && v.rows*v.pitch<=v.extent,"invalid BF16 CPU view");
    require(actual.rows==expected.rows && actual.columns==expected.columns,"comparison geometry differs");
    Comparison result;double energy=0,error=0;
    for(size_t r=0;r<actual.rows;++r)for(size_t c=0;c<actual.columns;++c) {
        uint16_t av,bv;std::memcpy(&av,static_cast<const char*>(actual.data)+r*actual.pitch+c*2,2);
        std::memcpy(&bv,static_cast<const char*>(expected.data)+r*expected.pitch+c*2,2);
        const double a=std::bit_cast<float>(uint32_t(av)<<16),b=std::bit_cast<float>(uint32_t(bv)<<16);
        require(std::isfinite(a) && std::isfinite(b),"nonfinite FFN output");
        error+=(a-b)*(a-b);energy+=b*b;result.max_abs=std::max(result.max_abs,std::abs(a-b));result.differences+=av!=bv;
    }
    require(energy>0,"degenerate FFN reference");result.rrms=std::sqrt(error/energy);result.elements=actual.rows*actual.columns;return result;
}
inline CpuView cpu(const mg::MatrixView& v) {return {v.storage.contents(),v.rows,v.columns,v.row_stride_bytes,v.storage.size_bytes()};}
inline NSDictionary* dictionary(const Comparison& c) {return @{@"rrms":@(c.rrms),@"max_abs":@(c.max_abs),@"bf16_differences":@(c.differences),@"finite_elements":@(c.elements),@"passes_rrms_limit":@(c.rrms<=rrms_limit)};}
struct Scratch {const char* label;uint32_t columns;mg::ScalarType type;};
inline constexpr std::array<Scratch,6> specs{{{"lora-replay-gate",width,mg::ScalarType::BF16},
    {"lora-replay-up",width,mg::ScalarType::BF16},{"lora-replay-hidden",width,mg::ScalarType::BF16},
    {"lora-replay-output",hidden,mg::ScalarType::BF16},{"lora-replay-low",rank,mg::ScalarType::FP32},
    {"lora-replay-pair-copy",hidden,mg::ScalarType::BF16}}};
inline size_t pitch(const Scratch& s) {return (size_t(s.columns)+8)*mg::scalar_bytes(s.type);}
inline size_t backing(const Scratch& s) {return (rows+2)*pitch(s);}
inline size_t upper(const Scratch& s) {return (backing(s)+16383)&~size_t(16383);}
inline std::vector<mg::AllocationSiteBinding> site_bindings() {
    std::vector<mg::AllocationSiteBinding> out;for(const auto& s:specs)out.push_back({s.label,s.label,MemoryClass::Activation});return out;
}
inline std::unique_ptr<MemoryExecutionContext> make_context(const ProcessMemoryObservation& baseline,const std::string& source,
                                               const char* candidate_id,bool cpu_only=false) {
    require(baseline.available && baseline.process_footprint_bytes && baseline.physical_memory_bytes,"missing process baseline");
    MemoryManifest m;m.candidate_id=candidate_id;m.checkpoint_digest=memory_sha256_hex(source);
    m.backend_revision="shared-six-scratch-v1";m.runtime_revision=memory_runtime_revision();
    uint64_t native_upper=0;for(const auto& s:specs) {
        m.sites.push_back({s.label,"native_lora_ffn","replay",MemoryClass::Activation,AllocationLifetime::Request,UpperProvenance::ExactShapeFormula});
        m.instances.push_back({s.label,0,upper(s),0,2,0,"physical_drain"});native_upper+=upper(s);
    }
    constexpr uint64_t framework=512ULL<<20,system=4ULL<<30;
    require(baseline.physical_memory_bytes>system,"insufficient physical memory guard");
    const auto budget=baseline.physical_memory_bytes-system;
    require(baseline.process_footprint_bytes<budget && native_upper+framework<=budget-baseline.process_footprint_bytes &&
        baseline.system_available_bytes>=system+native_upper+framework,"insufficient available memory for bounded replay");
    auto compiled=compile_memory_plan(m,baseline.process_footprint_bytes,framework,budget,true);
    require(compiled.complete && compiled.fits_budget,"replay phase plan is not executable");
    EffectiveMemoryPolicy p;p.enabled=true;p.effective_budget_bytes=budget;p.framework_upper_bytes=framework;
    p.planned_upper_bytes=compiled.planned_peak_bytes;p.system_reserve_bytes=system;p.denoiser_budget_bytes=native_upper;
    p.capability_level=MemoryCapabilityLevel::AllocationGuarded;p.certification_state=MemoryCertificationState::ExperimentalGuarded;
    p.execution_supported=true;p.route_available=true;p.estimate_fits=true;
    p.enforcement_scope="owned six native scratch backings only; produced MLX sources remain in retained baseline";
    ProcessMemoryObserver observer;if(cpu_only)observer=[baseline]{return baseline;};
    auto c=std::make_unique<MemoryExecutionContext>(p,std::move(compiled),baseline,std::move(observer));c->begin_running();return c;
}
inline mg::MatrixView imported(mg::Backend& b,const Tensor& a) {
    auto v=mg::import_mlx_matrix(b,a);
    const auto owner=v.storage.buffer().allocation_identity();const std::shared_ptr<const void> actual=a.data_shared_ptr();
    require(owner && actual && !owner.owner_before(actual) && !actual.owner_before(owner),"import lost actual MLX Data owner");
    return {v.storage,v.rows,v.columns,v.row_stride_bytes};
}
inline mg::MatrixView allocate(mg::Backend& b,const Scratch& s) {
    auto a=b.allocate(backing(s),s.label,mg::StorageMode::Shared);require(a.contents(),"shared scratch unavailable");
    std::memset(a.contents(),0x5a,a.size_bytes());return {a.view(pitch(s),rows*pitch(s),s.type),rows,s.columns,pitch(s)};
}
inline void padding(const mg::MatrixView& v) {
    const auto* data=static_cast<const unsigned char*>(v.storage.buffer().contents());require(data,"padding storage missing");
    const size_t item=mg::scalar_bytes(v.storage.scalar_type()),start=v.storage.offset_bytes();
    const auto unchanged=[&](size_t begin,size_t end) {for(size_t i=begin;i<end;++i)
        require(data[i]==0x5a,"native scratch guard/padding changed");};
    unchanged(0,start);
    for(size_t r=0;r<v.rows;++r)unchanged(start+r*v.row_stride_bytes+v.columns*item,start+(r+1)*v.row_stride_bytes);
    unchanged(start+v.rows*v.row_stride_bytes,v.storage.buffer().size_bytes());
}
inline std::string matrix_hash(const mg::MatrixView& v) {
    require(v.storage.contents() && v.row_stride_bytes==size_t(v.columns)*mg::scalar_bytes(v.storage.scalar_type()),"source must be compact and shared");
    return memory_sha256_hex({static_cast<const char*>(v.storage.contents()),size_t(v.rows)*v.row_stride_bytes});
}
inline mg::native_lora::Layout layout(const mg::MatrixView& v) {
    return {v.rows,v.columns,v.row_stride_bytes,v.storage.size_bytes(),v.storage.offset_bytes(),v.storage.scalar_type(),
        reinterpret_cast<uintptr_t>(v.storage.buffer().native_handle())};
}
inline bool eligible(const mg::MatrixView& x,const mg::OrderedLoRATerm& term,const mg::MatrixView& out,const mg::MatrixView& low) {
    mg::native_lora::TermLayout t{layout(term.a),layout(term.b),term.effective_scale,term.first_output,term.output_count};
    auto checked=mg::native_lora::validate_bindings({width,width,rank},layout(x),std::span(&t,1),layout(out),layout(out),layout(low));
    return checked.terms.size()==1 && mg::native_lora::sg_mma_eligible(checked.terms.front());
}
inline std::string build_identity(std::string_view expected) {
    char* value=tc_runtime_build_identity();require(value,"runtime identity missing");const std::string s(value);tc_string_free(value);
    require(s==expected,"loaded runtime differs from explicit linked build");return s;
}

} // namespace tc::qwen_lora_replay
