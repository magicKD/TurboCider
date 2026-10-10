#import <Foundation/Foundation.h>
#include "../../models/qwen21/bf16_streaming.hpp"
#include "../../core/json_keys.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <cerrno>

namespace tc::qwen21 {
void bf16_stream_scope(const std::function<void()> &body) {
    @autoreleasepool {body();}
}
namespace {
uint64_t integer(id value) {
    require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),
        "Qwen BF16 stream header requires integer dimensions/offsets");
    const double n=[value doubleValue];
    require(std::isfinite(n) && n>=0 && n==std::floor(n) && n<=double(uint64_t(1)<<53),"Qwen BF16 stream header integer overflow");
    return uint64_t(n);
}
void read_exact(int fd,void *out,uint64_t bytes,uint64_t offset,std::atomic<bool> &cancel) {
    uint64_t done=0;auto *p=static_cast<char *>(out);
    while(done<bytes) {
        checkpoint(cancel);const auto n=::pread(fd,p+done,size_t(std::min(bytes-done,uint64_t(1)<<20)),off_t(offset+done));
        if(n<0 && errno==EINTR)continue;
        require(n>0,"Qwen BF16 stream header read failed");done+=uint64_t(n);
    }
}
}
Bf16StreamPlan bf16_stream_plan(const std::filesystem::path &path,bool encoder,
        uint32_t layers,uint32_t prefix,uint32_t passes,std::atomic<bool> &cancel) {
    require(layers>=2 && layers<=128 && prefix<=layers-2 && passes>0 && passes<=40,"Qwen BF16 stream requires at least two suffix groups and bounded passes");
    Bf16StreamPlan plan;plan.layers=layers;plan.prefix=prefix;
    const auto begin=Clock::now();streaming::SourceFileIdentity source;
    source.logical_id=path.filename().string();source.path=path;
    require(path.extension()==".safetensors","Qwen BF16 streaming requires original safetensors");
    plan.lease=streaming::SourceLease::capture_verified({source},&cancel);
    plan.verification_seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    const auto &identity=plan.lease->file(source.logical_id);auto fd=plan.lease->duplicate_fd(source.logical_id);
    std::array<uint8_t,8> encoded{};read_exact(fd.get(),encoded.data(),8,0,cancel);
    uint64_t header_bytes=0;for(uint32_t i=0;i<8;++i)header_bytes|=uint64_t(encoded[i])<<(i*8);
    require(header_bytes>0 && header_bytes<=(uint64_t(16)<<20) && identity.bytes>=8 && header_bytes<=identity.bytes-8,
        "Qwen BF16 stream invalid header extent");
    std::string header(size_t(header_bytes),'\0');read_exact(fd.get(),header.data(),header_bytes,8,cancel);
    streaming::StageDescriptor stage;stage.id=encoder ? "encoder" : "denoiser";
    stage.adapter_revision="qwen21-original-bf16-dynamic-slots-v1";stage.pass_count=passes;stage.min_slots=stage.max_slots=2;
    const std::string stem=encoder ? "model.layers." : "transformer_blocks.";
    const std::set<std::string> expected=encoder ? std::set<std::string>{
        "input_layernorm.weight","post_attention_layernorm.weight","self_attn.q_norm.weight","self_attn.k_norm.weight",
        "self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight",
        "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"} : std::set<std::string>{
        "attn.norm_k.weight","attn.norm_q.weight","attn.to_k.weight","attn.to_q.weight","attn.to_v.weight","attn.to_out.0.weight",
        "img_mlp.gate_up.weight","img_mlp.out.weight"};
    std::vector<std::map<std::string,streaming::FieldSpec>> block_fields(layers);
    std::vector<std::pair<uint64_t,uint64_t>> intervals;
    std::map<std::string,streaming::FieldSpec> fixed;
    @autoreleasepool {
        NSError *error=nil;id parsed=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:header.data() length:header.size()]
            options:0 error:&error];
        require([parsed isKindOfClass:NSDictionary.class],"Qwen BF16 stream header JSON is not an object");
        reject_duplicate_json_keys(header);NSDictionary *root=parsed;
        for(NSString *key in [[root allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
            checkpoint(cancel);require([key isKindOfClass:NSString.class],"Qwen BF16 stream nonstring tensor key");
            if([key isEqualToString:@"__metadata__"])continue;
            const char *raw=[key UTF8String];require(raw && std::strlen(raw)==[key lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
                "Qwen BF16 stream invalid tensor name");const std::string name=raw;
            NSDictionary *record=root[key];require([record isKindOfClass:NSDictionary.class] && [record[@"dtype"] isKindOfClass:NSString.class] &&
                [record[@"dtype"] isEqualToString:@"BF16"],"Qwen BF16 stream requires original BF16 tensors");
            NSArray *shape=record[@"shape"],*offsets=record[@"data_offsets"];
            require([shape isKindOfClass:NSArray.class] && shape.count>0 && shape.count<=8 &&
                [offsets isKindOfClass:NSArray.class] && offsets.count==2,"Qwen BF16 stream invalid shape/offset metadata");
            uint64_t bytes=2;std::vector<uint64_t> dimensions;
            for(id dimension in shape) {const auto n=integer(dimension);require(n>0 && n<=INT32_MAX && n<=UINT64_MAX/bytes,
                "Qwen BF16 stream tensor shape overflow");bytes*=n;dimensions.push_back(n);}
            const auto lo=integer(offsets[0]),hi=integer(offsets[1]);
            require(lo<=hi && hi-lo==bytes && hi<=identity.bytes-header_bytes-8,"Qwen BF16 stream payload range mismatch");
            intervals.emplace_back(lo,hi);
            // Base language conditioning does not consume the LM head or visual tower.
            if(encoder && (name=="lm_head.weight" || name.starts_with("model.visual.")))continue;
            streaming::Materialization m{"BF16","mlx-metal-shared","copy-bf16-v1",dimensions,
                {{0,8+header_bytes+lo,bytes,name,"BF16",dimensions}},{},0};
            streaming::FieldSpec field{name,name,bytes,16384,std::move(m)};
            if(!name.starts_with(stem)) {require(fixed.emplace(name,std::move(field)).second,"duplicate fixed Qwen BF16 field");continue;}
            const auto dot=name.find('.',stem.size());require(dot!=std::string::npos,"Qwen BF16 stream layer field has no suffix");
            const auto number=name.substr(stem.size(),dot-stem.size());
            require(!number.empty() && number.size()<=3 && number.find_first_not_of("0123456789")==std::string::npos,
                "Qwen BF16 stream malformed layer ordinal");const auto ordinal=std::stoul(number);
            require(ordinal<layers && number==std::to_string(ordinal),"Qwen BF16 stream layer ordinal out of range");
            const auto suffix=name.substr(dot+1);require(expected.contains(suffix),"Qwen BF16 stream unknown layer field");
            field.name=suffix;require(block_fields[ordinal].emplace(suffix,std::move(field)).second,"duplicate Qwen BF16 layer field");
        }
    }
    std::sort(intervals.begin(),intervals.end());uint64_t end=0;
    for(const auto &[lo,hi]:intervals){require(lo==end && hi>lo,"Qwen BF16 stream overlapping/gapped payload");end=hi;}
    require(end==identity.bytes-header_bytes-8,"Qwen BF16 stream unaccounted payload tail");
    require(!fixed.empty() && (!encoder || (fixed.contains("model.embed_tokens.weight") && fixed.contains("model.norm.weight"))),
        "Qwen BF16 stream missing resident embedding/norm/fixed fields");
    for(auto &[name,field]:fixed)stage.resident_fields.push_back(std::move(field));
    for(uint32_t i=0;i<layers;++i) {
        require(block_fields[i].size()==expected.size(),"Qwen BF16 stream layer source closure incomplete");
        streaming::BlockSpec block;block.id=i;block.layout_class=encoder ? "language" : "dit";
        for(auto &[name,field]:block_fields[i]) {
            if(i<prefix) {field.name=stem+std::to_string(i)+"."+name;stage.resident_fields.push_back(std::move(field));}
            else block.fields.push_back(std::move(field));
        }
        if(i>=prefix)stage.blocks.push_back(std::move(block));
    }
    for(uint32_t i=0;i<passes;++i)stage.passes.push_back({i,i==0 ? "prefill" : "decode",{1,layers}});
    plan.descriptor={"qwen-image-2.1",identity.content_digest,"original-bf16-two-slot-private-v1",{std::move(stage)},
        {{source.logical_id,identity.content_digest,identity.bytes,streaming::SourceIdentityKind::content_sha256}},
        {{"component",encoder ? "encoder" : "denoiser"},{"full_layers",std::to_string(layers)},{"resident_prefix",std::to_string(prefix)}}};
    StreamingConfig config;config.enabled=true;config.schema_version=1;config.selection="manual";config.retention="request";
    config.stages[plan.descriptor.stages[0].id]={"streamed",1,2,0,1,1};
    plan.layout=streaming::compile_layout(config,plan.descriptor);
    require(plan.layout.materializations_complete && plan.layout.stages[0].pools.size()==1,"Qwen BF16 stream incomplete/multiple-pool plan");
    plan.lease->revalidate_after_drain();return plan;
}
}
