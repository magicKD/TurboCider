#import <Foundation/Foundation.h>
#include "../../models/qwen21/bf16_streaming.hpp"
#include "../../models/qwen21/source_registration.hpp"
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

using Fields = std::map<std::string, streaming::FieldSpec>;
using ExpectedTypes = std::map<std::string, std::string>;

void add_projection(ExpectedTypes &types, const std::string &stem, bool convrot) {
    types.emplace(stem + ".weight", convrot ? "I8" : "BF16");
    if (convrot) {
        types.emplace(stem + ".weight_scale", "F32");
        types.emplace(stem + ".comfy_quant", "U8");
    }
}

NSDictionary *read_marker(int fd, uint64_t offset, const std::vector<uint64_t> &shape,
                          std::atomic<bool> &cancel) {
    require(shape.size() == 1 && shape.front() <= 256, "invalid Qwen ConvRot marker shape");
    std::string payload(size_t(shape.front()), '\0');
    read_exact(fd, payload.data(), payload.size(), offset, cancel);
    reject_duplicate_json_keys(payload);
    id parsed = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:payload.data() length:payload.size()]
        options:0 error:nil];
    require([parsed isKindOfClass:NSDictionary.class], "Qwen ConvRot marker must be a JSON object");
    return parsed;
}

void validate_projection(const Fields &fields, const std::string &stem, bool tensorops) {
    const auto &codes = fields.at(stem + ".weight").materialization->shape;
    const auto &scales = fields.at(stem + ".weight_scale").materialization->shape;
    require(codes.size() == 2 && codes[1] % 256 == 0 && codes[1] <= 131072 &&
            scales == std::vector<uint64_t>{codes[0], 1},
        "invalid Qwen ConvRot I8 matrix or F32 row-scale geometry: " + stem);
    // No weight padding/copy may escape the admitted source slot budget.
    require(!tensorops || codes[0] % 128 == 0, "Qwen ConvRot TensorOps source requires N aligned to 128: " + stem);
}
}
Bf16StreamPlan bf16_stream_plan(const std::filesystem::path &path,bool encoder,
        uint32_t layers,uint32_t prefix,uint32_t passes,std::atomic<bool> &cancel,StreamWeightMode mode) {
    const bool convrot_request = mode != StreamWeightMode::Bf16;
    const bool convrot = mode == StreamWeightMode::ConvRotFloat || mode == StreamWeightMode::ConvRotW8A8;
    const bool tensorops = mode == StreamWeightMode::ConvRotW8A8;
    require(layers>=2 && layers<=128 && prefix<=layers-2 && passes>0 && passes<=40,
        "Qwen weight stream requires at least two suffix groups and bounded passes");
    require(!convrot_request || prefix == 0, "Qwen ConvRot weight streaming requires zero resident layer prefix");
    require(!tensorops || convrot_w8a8_available(), "Qwen ConvRot W8A8 requires M5 Metal TensorOps and macOS 26.4 or later");
    Bf16StreamPlan plan;plan.layers=layers;plan.prefix=prefix;plan.mode=mode;
    const auto begin=Clock::now();streaming::SourceFileIdentity source;
    source.logical_id=path.filename().string();source.path=path;
    require(path.extension()==".safetensors","Qwen weight streaming requires original safetensors");
    plan.lease=capture_model_source(source,cancel,convrot_request ? streaming::SourceReadPolicy::Uncached :
        streaming::SourceReadPolicy::Default);
    plan.verification_seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    const auto &identity=plan.lease->file(source.logical_id);auto fd=plan.lease->duplicate_fd(source.logical_id);
    std::array<uint8_t,8> encoded{};read_exact(fd.get(),encoded.data(),8,0,cancel);
    uint64_t header_bytes=0;for(uint32_t i=0;i<8;++i)header_bytes|=uint64_t(encoded[i])<<(i*8);
    require(header_bytes>0 && header_bytes<=(uint64_t(16)<<20) && identity.bytes>=8 && header_bytes<=identity.bytes-8,
        "Qwen weight stream invalid header extent");
    std::string header(size_t(header_bytes),'\0');read_exact(fd.get(),header.data(),header_bytes,8,cancel);
    streaming::StageDescriptor stage;stage.id=encoder ? "encoder" : "denoiser";
    stage.adapter_revision=convrot ? "qwen21-original-convrot-dynamic-slots-v1" : "qwen21-original-bf16-dynamic-slots-v1";
    stage.pass_count=passes;stage.min_slots=stage.max_slots=2;
    const std::string stem=encoder ? "model.layers." : "transformer_blocks.";
    const std::vector<std::string> projections = encoder ? std::vector<std::string>{
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} : std::vector<std::string>{
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"};
    ExpectedTypes expected = encoder ? ExpectedTypes{
        {"input_layernorm.weight", "BF16"}, {"post_attention_layernorm.weight", "BF16"},
        {"self_attn.q_norm.weight", "BF16"}, {"self_attn.k_norm.weight", "BF16"}} : ExpectedTypes{
        {"attn.norm_k.weight", "BF16"}, {"attn.norm_q.weight", "BF16"}};
    for (const auto &projection : projections) add_projection(expected, projection, convrot);
    ExpectedTypes fixed_types;
    ExpectedTypes head_types;
    if (convrot && encoder) {
        fixed_types.emplace("model.norm.weight", "BF16");
        add_projection(fixed_types, "model.embed_tokens", true);
        add_projection(head_types, "lm_head", true);
    } else if (convrot) {
        for (const auto *name : {"img_in.weight", "modulation.1.weight", "norm_out.linear.weight",
                "proj_out.weight", "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_2.weight", "txt_in.in_layer.weight",
                "txt_in.out_layer.weight", "txt_in.text_norm.weight"})
            fixed_types.emplace(name, "BF16");
    }
    std::vector<Fields> block_fields(layers);
    std::vector<std::pair<uint64_t,uint64_t>> intervals;
    Fields fixed, unused_head;
    @autoreleasepool {
        NSError *error=nil;id parsed=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:header.data() length:header.size()]
            options:0 error:&error];
        require([parsed isKindOfClass:NSDictionary.class],"Qwen weight stream header JSON is not an object");
        reject_duplicate_json_keys(header);NSDictionary *root=parsed;
        for(NSString *key in [[root allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
            checkpoint(cancel);require([key isKindOfClass:NSString.class],"Qwen weight stream nonstring tensor key");
            if([key isEqualToString:@"__metadata__"])continue;
            const char *raw=[key UTF8String];require(raw && std::strlen(raw)==[key lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
                "Qwen weight stream invalid tensor name");const std::string name=raw;
            NSDictionary *record=root[key];
            require([record isKindOfClass:NSDictionary.class] && [record[@"dtype"] isKindOfClass:NSString.class],
                "Qwen weight stream missing tensor type");
            NSString *type = record[@"dtype"];
            std::string dtype, conversion;
            uint64_t bytes;
            if ([type isEqualToString:@"BF16"]) {
                dtype = "BF16"; conversion = "copy-bf16-v1"; bytes = 2;
            } else if (convrot && [type isEqualToString:@"I8"]) {
                dtype = "I8"; conversion = "copy-i8-v1"; bytes = 1;
            } else if (convrot && [type isEqualToString:@"F32"]) {
                dtype = "F32"; conversion = "copy-f32-v1"; bytes = 4;
            } else if (convrot && [type isEqualToString:@"U8"]) {
                dtype = "U8"; conversion = "copy-u8-v1"; bytes = 1;
            } else {
                throw std::invalid_argument("unsupported Qwen streaming tensor type: " + name);
            }
            NSArray *shape=record[@"shape"],*offsets=record[@"data_offsets"];
            require([shape isKindOfClass:NSArray.class] && shape.count>0 && shape.count<=8 &&
                [offsets isKindOfClass:NSArray.class] && offsets.count==2,"Qwen weight stream invalid shape/offset metadata");
            std::vector<uint64_t> dimensions;
            for(id dimension in shape) {const auto n=integer(dimension);require(n>0 && n<=INT32_MAX && n<=UINT64_MAX/bytes,
                "Qwen weight stream tensor shape overflow");bytes*=n;dimensions.push_back(n);}
            const auto lo=integer(offsets[0]),hi=integer(offsets[1]);
            require(lo<=hi && hi-lo==bytes && hi<=identity.bytes-header_bytes-8,"Qwen weight stream payload range mismatch");
            intervals.emplace_back(lo,hi);
            // Base language conditioning does not consume the LM head or visual tower.
            if (encoder && name.starts_with("model.visual.")) {
                require(dtype == "BF16", "unsupported quantized Qwen3-VL visual tensor: " + name);
                continue;
            }
            if (encoder && !convrot && name == "lm_head.weight") continue;
            const uint64_t offset = 8 + header_bytes + lo;
            if (convrot && name.ends_with(".comfy_quant")) {
                require(dtype == "U8", "Qwen ConvRot marker must be U8: " + name);
                NSDictionary *marker = read_marker(fd.get(), offset, dimensions, cancel);
                id rotation = marker[@"convrot"];
                require(marker.count == 3 && [marker[@"format"] isEqual:@"int8_tensorwise"] &&
                        rotation && CFGetTypeID((__bridge CFTypeRef)rotation) == CFBooleanGetTypeID() &&
                        [rotation boolValue] && integer(marker[@"convrot_groupsize"]) == 256,
                    "Qwen weight streaming requires the original int8_tensorwise / ConvRot H256 marker: " + name);
            }
            streaming::Materialization m{dtype,"mlx-metal-shared",conversion,dimensions,
                {{0,offset,bytes,name,dtype,dimensions}},{},0};
            streaming::FieldSpec field{name,name,bytes,16384,std::move(m)};
            if (!name.starts_with(stem)) {
                const bool is_head = encoder && convrot && name.starts_with("lm_head.");
                const auto &allowed = is_head ? head_types : fixed_types;
                if (convrot) {
                    const auto found = allowed.find(name);
                    require(found != allowed.end() && found->second == dtype,
                        "unexpected fixed Qwen ConvRot tensor or type: " + name);
                }
                auto &destination = is_head ? unused_head : fixed;
                require(destination.emplace(name,std::move(field)).second,"duplicate fixed Qwen weight field");
                continue;
            }
            const auto dot=name.find('.',stem.size());require(dot!=std::string::npos,"Qwen weight stream layer field has no suffix");
            const auto number=name.substr(stem.size(),dot-stem.size());
            require(!number.empty() && number.size()<=3 && number.find_first_not_of("0123456789")==std::string::npos,
                "Qwen weight stream malformed layer ordinal");const auto ordinal=std::stoul(number);
            require(ordinal<layers && number==std::to_string(ordinal),"Qwen weight stream layer ordinal out of range");
            const auto suffix=name.substr(dot+1);
            if (convrot && !encoder && suffix == "attn.comfy_attention.config") {
                require(dtype == "U8", "Qwen attention marker must be U8");
                NSDictionary *marker = read_marker(fd.get(), offset, dimensions, cancel);
                require(marker.count == 1 && [marker[@"attention"] isEqual:@"comfy_kitchen_int8"],
                    "unrecognized Qwen ConvRot attention marker");
                continue;
            }
            const auto found = expected.find(suffix);
            require(found != expected.end() && found->second == dtype,"unexpected Qwen layer field or type: " + name);
            field.name=suffix;require(block_fields[ordinal].emplace(suffix,std::move(field)).second,"duplicate Qwen layer field");
        }
    }
    std::sort(intervals.begin(),intervals.end());uint64_t end=0;
    for(const auto &[lo,hi]:intervals){require(lo==end && hi>lo,"Qwen weight stream overlapping/gapped payload");end=hi;}
    require(end==identity.bytes-header_bytes-8,"Qwen weight stream unaccounted payload tail");
    require(!fixed.empty() && (!encoder || (fixed.contains("model.embed_tokens.weight") && fixed.contains("model.norm.weight"))),
        "Qwen weight stream missing resident embedding/norm/fixed fields");
    if (convrot) {
        require(fixed.size() == fixed_types.size(), "Qwen ConvRot stream fixed source closure incomplete");
        if (encoder) validate_projection(fixed, "model.embed_tokens", false);
        if (!unused_head.empty()) {
            require(unused_head.size() == head_types.size(), "Qwen ConvRot LM head source closure incomplete");
            validate_projection(unused_head, "lm_head", false);
        }
    }
    if (convrot && encoder) {
        plan.embedding = StreamEmbeddingSource{std::move(fixed.at("model.embed_tokens.weight")),
            std::move(fixed.at("model.embed_tokens.weight_scale"))};
        fixed.erase("model.embed_tokens.weight");
        fixed.erase("model.embed_tokens.weight_scale");
        fixed.erase("model.embed_tokens.comfy_quant");
    }
    for(auto &[name,field]:fixed)stage.resident_fields.push_back(std::move(field));
    for(uint32_t i=0;i<layers;++i) {
        require(block_fields[i].size()==expected.size(),"Qwen weight stream layer source closure incomplete");
        if (convrot) for (const auto &projection : projections)
            validate_projection(block_fields[i], projection, tensorops);
        streaming::BlockSpec block;block.id=i;block.layout_class=encoder ? "language" : "dit";
        for(auto &[name,field]:block_fields[i]) {
            if(i<prefix) {field.name=stem+std::to_string(i)+"."+name;stage.resident_fields.push_back(std::move(field));}
            else block.fields.push_back(std::move(field));
        }
        if(i>=prefix)stage.blocks.push_back(std::move(block));
    }
    for(uint32_t i=0;i<passes;++i)stage.passes.push_back({i,i==0 ? "prefill" : "decode",{1,layers}});
    plan.descriptor={"qwen-image-2.1",identity.content_digest,
        convrot ? (tensorops ? "original-convrot-w8a8-sparse-embedding-two-slot-v2" : "original-convrot-float-a-sparse-embedding-two-slot-v2") :
            convrot_request ? "original-bf16-convrot-companion-two-slot-v1" : "original-bf16-two-slot-private-v1",{std::move(stage)},
        {{source.logical_id,identity.content_digest,identity.bytes,streaming::SourceIdentityKind::content_sha256}},
        {{"component",encoder ? "encoder" : "denoiser"},{"full_layers",std::to_string(layers)},{"resident_prefix",std::to_string(prefix)}}};
    StreamingConfig config;config.enabled=true;config.schema_version=1;config.selection="manual";config.retention="request";
    config.stages[plan.descriptor.stages[0].id]={"streamed",1,2,0,1,1};
    plan.layout=streaming::compile_layout(config,plan.descriptor);
    require(plan.layout.materializations_complete && plan.layout.stages[0].pools.size()==1,"Qwen weight stream incomplete/multiple-pool plan");
    plan.lease->revalidate_after_drain();return plan;
}
}
