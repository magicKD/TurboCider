#pragma once
#include "query_worker.hpp"
#include <cmath>
#include <filesystem>

namespace tc_worker::runtime {
inline bool normalized_path(id value) {
    if(!text_value(value) || ![value isAbsolutePath] || [value rangeOfString:@"\0"].location!=NSNotFound)return false;
    // NSString standardization resolves existing /private/var aliases too;
    // wire normalization is lexical. Physical parent checks are separate.
    std::string path([value UTF8String]);return std::filesystem::path(path).lexically_normal().string()==path;
}
inline void keys(NSDictionary *value,NSArray<NSString *> *allowed,NSArray<NSString *> *required=@[]) {
    NSSet *set=[NSSet setWithArray:allowed];
    for(id key in value)require([set containsObject:key],"runtime_worker_fields_invalid");
    for(NSString *key in required)require(value[key]!=nil,"runtime_worker_fields_missing");
}
inline bool number(id value,double minimum,double maximum) {
    return [value isKindOfClass:NSNumber.class] &&
        CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID() &&
        std::isfinite([value doubleValue]) && [value doubleValue]>=minimum && [value doubleValue]<=maximum;
}
inline bool whole(id value,double minimum,double maximum) {
    return number(value,minimum,maximum) && std::floor([value doubleValue])==[value doubleValue];
}
inline NSString *digest(NSDictionary *request,NSDictionary *options) {
    NSMutableData *bytes=[NSMutableData dataWithData:[@"tc-runtime-worker-request-v1\n" dataUsingEncoding:NSUTF8StringEncoding]];
    [bytes appendData:canonical_request(@{@"native_request_v2":request,@"runtime_options":options})];
    require(bytes.length<=UINT32_MAX,"runtime_worker_request_too_large");
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.bytes,CC_LONG(bytes.length),hash);
    NSMutableString *hex=[NSMutableString stringWithCapacity:64];
    for(unsigned char byte:hash)[hex appendFormat:@"%02x",byte];return hex;
}
inline NSDictionary *validate(id raw) {
    NSDictionary *input=object(raw);
    NSArray *fields=@[@"protocol_version",@"job_id",@"request_id",@"request_digest",@"model_installation_ref",@"native_request_v2",@"runtime_options"];
    keys(input,fields,fields);
    require(integer(input[@"protocol_version"],1),"runtime_worker_protocol_unsupported");
    require(uuid(input[@"job_id"]) && uuid(input[@"request_id"]),"runtime_worker_ids_invalid");
    require(normalized_path(input[@"model_installation_ref"]),"runtime_worker_installation_invalid");
    NSDictionary *request=object(input[@"native_request_v2"]),*options=object(input[@"runtime_options"]);
    NSArray *option_keys=@[@"profile_id",@"descriptor_path",@"cache_dir"];
    keys(options,option_keys,option_keys);
    require(equal(options[@"profile_id"],@"private-w8a8-hadamard-512-v1") &&
        normalized_path(options[@"descriptor_path"]) && normalized_path(options[@"cache_dir"]),"runtime_worker_options_unqualified");
    keys(request,@[@"schema_version",@"model",@"operation",@"inputs",@"outputs",@"sampling",@"execution",@"parameters",@"loras",@"lora_strategy"],
        @[@"schema_version",@"model",@"operation",@"inputs",@"outputs",@"sampling",@"execution",@"parameters"]);
    const bool qwen=equal(request[@"model"],@"qwen-image-2.1");
    require(integer(request[@"schema_version"],2) && (qwen || equal(request[@"model"],@"z-image-turbo")),"runtime_worker_model_unqualified");
    NSDictionary *execution=object(request[@"execution"]),*parameters=object(request[@"parameters"]),*sampling=object(request[@"sampling"]);
    keys(execution,@[@"policy",@"ane_manifest",@"allow_approximation",@"hybrid_mlp_mode",@"residency",@"qwen21_w8a8",@"warmup_iterations",@"quantized_cache",@"qwen21_dit_cache"],
        @[@"policy",@"ane_manifest",@"allow_approximation",@"hybrid_mlp_mode",@"residency"]);
    require(equal(execution[@"policy"],@"gpu_ane") && equal(execution[@"hybrid_mlp_mode"],@"runtime") &&
        boolean(execution[@"allow_approximation"],true) && equal(execution[@"ane_manifest"],options[@"descriptor_path"]) &&
        equal(execution[@"residency"],qwen?@"component_staged":@"resident"),"runtime_worker_execution_unqualified");
    require((!execution[@"qwen21_w8a8"] || boolean(execution[@"qwen21_w8a8"],false)) &&
        (!execution[@"warmup_iterations"] || integer(execution[@"warmup_iterations"],0)) &&
        (!execution[@"quantized_cache"] || equal(execution[@"quantized_cache"],@"")) &&
        (!execution[@"qwen21_dit_cache"] || equal(execution[@"qwen21_dit_cache"],@"off")),"runtime_worker_execution_unqualified");
    keys(parameters,qwen?@[@"dynamic_text",@"compile_gpu",@"qwen21_reference_size"]:@[@"dynamic_text",@"compile_gpu"],@[@"dynamic_text"]);
    require(boolean(parameters[@"dynamic_text"],true) && (!parameters[@"compile_gpu"] || boolean(parameters[@"compile_gpu"],false)) &&
        (!qwen || integer(parameters[@"qwen21_reference_size"],1024)),"runtime_worker_parameters_unqualified");
    keys(sampling,@[@"seed",@"steps"],@[@"seed",@"steps"]);
    require(whole(sampling[@"seed"],0,UINT32_MAX),"runtime_worker_sampling_invalid");
    NSDictionary *output=generation_output(request);
    keys(output,@[@"kind",@"path",@"width",@"height",@"frames",@"fps",@"audio"],@[@"kind",@"path",@"width",@"height",@"frames",@"fps",@"audio"]);
    require(integer(output[@"width"],512) && integer(output[@"height"],512) && integer(output[@"frames"],1) &&
        integer(output[@"fps"],24) && boolean(output[@"audio"],false),"runtime_worker_output_unqualified");
    id inputs=request[@"inputs"];
    require([inputs isKindOfClass:NSArray.class] && [inputs count]>=1 && [inputs count]<=(qwen?3:1),"runtime_worker_inputs_unqualified");
    size_t prompts=0,references=0;
    for(id item in inputs) {
        NSDictionary *asset=object(item);
        if(equal(asset[@"kind"],@"text")) {
            keys(asset,@[@"kind",@"role",@"text"],@[@"kind",@"role",@"text"]);
            require(equal(asset[@"role"],@"prompt") && [asset[@"text"] isKindOfClass:NSString.class] &&
                [asset[@"text"] rangeOfString:@"\0"].location==NSNotFound,"runtime_worker_prompt_invalid");++prompts;
        } else {
            keys(asset,@[@"kind",@"role",@"path",@"strength"],@[@"kind",@"role",@"path"]);
            require(qwen && equal(asset[@"kind"],@"image") && equal(asset[@"role"],@"reference") &&
                normalized_path(asset[@"path"]) && (!asset[@"strength"] || number(asset[@"strength"],0,1)),"runtime_worker_reference_invalid");++references;
        }
    }
    require(prompts==1 && equal(request[@"operation"],references?@"image.edit":@"image.generate"),"runtime_worker_operation_unqualified");
    id loras=request[@"loras"]?:@[];
    require([loras isKindOfClass:NSArray.class] && [loras count]<=(qwen?1:8),"runtime_worker_loras_unqualified");
    require((![loras count] && !request[@"lora_strategy"]) || equal(request[@"lora_strategy"],@"inference_time"),"runtime_worker_loras_unqualified");
    for(id item in loras) {
        NSDictionary *lora=object(item);keys(lora,@[@"path",@"strength",@"role"],@[@"path",@"strength",@"role"]);
        require(normalized_path(lora[@"path"]) && equal(lora[@"role"],@"transformer") && number(lora[@"strength"],-8,8),"runtime_worker_loras_unqualified");
        if(qwen)require(equal([lora[@"path"] lastPathComponent],@"Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors") &&
            number(lora[@"strength"],1,1),"runtime_worker_loras_unqualified");
        // The normal native gate/loader still verifies actual checkpoint and
        // adapter provenance; this filename screening never grants authority.
    }
    require(integer(sampling[@"steps"],qwen?([loras count]?6:20):8),"runtime_worker_steps_unqualified");
    require(equal(input[@"request_digest"],digest(request,options)),"runtime_worker_request_digest_mismatch");
    return input;
}
inline NSDictionary *environment(NSDictionary *input) {
    NSMutableDictionary *values=[@{
        @"TURBOCIDER_ANE_BACKEND":@"auto",@"TURBOCIDER_ALLOW_PRIVATE_ANE":@"1",
        @"TURBOCIDER_PRIVATE_ANE_DATA_PATH":@"w8a8",@"TURBOCIDER_PRIVATE_ANE_GPU_IO":@"1",
        @"TURBOCIDER_PRIVATE_ANE_CHANNELS":@"4096",@"TURBOCIDER_RUNTIME_ANE_CHUNKS":@"1",
        @"TURBOCIDER_RUNTIME_ANE_PROFILE":@"0",@"TURBOCIDER_PRIVATE_ANE_SCALE_CACHE":@"1",
        @"TURBOCIDER_PRIVATE_ANE_PREFETCH":@"0",@"TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE":@"0",
        @"TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD":@"0",@"TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE":@"0",
        @"TURBOCIDER_PRIVATE_ANE_CACHE_DIR":input[@"runtime_options"][@"cache_dir"]} mutableCopy];
    if(equal(input[@"native_request_v2"][@"model"],@"qwen-image-2.1")) {
        values[@"TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC"]=@"1";
        values[@"TURBOCIDER_QWEN21_RUNTIME_PREPARE_EARLY"]=@"0";
    }
    return values;
}
inline void verify_environment(NSDictionary *input) {
    NSDictionary *expected=environment(input),*actual=NSProcessInfo.processInfo.environment;
    for(NSString *key in actual) if([key hasPrefix:@"TURBOCIDER_"])
        require(equal(actual[key],expected[key]),"runtime_worker_environment_unqualified");
    for(NSString *key in expected)require(equal(actual[key],expected[key]),"runtime_worker_environment_mismatch");
    // Never use ambient flags as authority. Install only the resolved profile,
    // after rejecting unknown switches (including MPP/calibration capture).
    for(NSString *key in expected)require(setenv(key.UTF8String,[expected[key] UTF8String],1)==0,"runtime_worker_environment_failed");
}
inline void verify_descriptor(NSDictionary *input) {
    NSDictionary *options=input[@"runtime_options"];
    NSString *path=options[@"descriptor_path"],*parent=[path stringByDeletingLastPathComponent],*cache=options[@"cache_dir"];
    char physical[PATH_MAX];struct stat info{};
    require(realpath(parent.fileSystemRepresentation,physical) && equal(parent,@(physical)),"runtime_worker_descriptor_parent_invalid");
    require(lstat(cache.fileSystemRepresentation,&info)==0 && S_ISDIR(info.st_mode) &&
        realpath(cache.fileSystemRepresentation,physical) && equal(cache,@(physical)),"runtime_worker_cache_invalid");
    NSDictionary *descriptor=object(read_input_object(path.fileSystemRepresentation));
    NSArray *fields=@[@"schema_version",@"descriptor_version",@"backend",@"kind",@"rows",@"hidden",@"width",@"tile_k",@"tile_n",@"layout",@"biases",@"lora_inputs"];
    keys(descriptor,fields,fields);
    const bool qwen=equal(input[@"native_request_v2"][@"model"],@"qwen-image-2.1");
    require(integer(descriptor[@"schema_version"],1) && integer(descriptor[@"descriptor_version"],1) &&
        equal(descriptor[@"backend"],@"private_runtime_shape") && equal(descriptor[@"kind"],@"swiglu") &&
        integer(descriptor[@"rows"],1056) && integer(descriptor[@"hidden"],qwen?4096:3840) &&
        integer(descriptor[@"width"],qwen?12288:10240) && integer(descriptor[@"tile_k"],2048) &&
        integer(descriptor[@"tile_n"],1024) && equal(descriptor[@"layout"],@"out_in") &&
        boolean(descriptor[@"biases"],false) && boolean(descriptor[@"lora_inputs"],true),"runtime_worker_descriptor_unqualified");
}
inline void verify_hardware() {
    // Match the machine on which this local profile was qualified. Device
    // discovery is read-only; no model/ANE compilation is used for admission.
    NSDictionary *system=consume(0,tc_system_json(),nullptr);
    require(equal(system[@"gpu"],@"Apple M4 Pro") &&
        number(system[@"physical_memory_bytes"],double(uint64_t(48)<<30),double(uint64_t(48)<<30)),
        "runtime_worker_hardware_unqualified");
}
inline NSDictionary *receipt(NSDictionary *result,NSDictionary *request) {
    NSDictionary *output=generation_output(request),*sampling=object(request[@"sampling"]);
    require(integer(result[@"schema_version"],1) && boolean(result[@"warmup"],false) &&
        equal(result[@"model"],request[@"model"]) && equal(result[@"operation"],request[@"operation"]) &&
        equal(result[@"output"],output[@"path"]) && equal(result[@"width"],output[@"width"]) && equal(result[@"height"],output[@"height"]) &&
        equal(result[@"seed"],sampling[@"seed"]) && equal(result[@"steps"],sampling[@"steps"]) &&
        (!result[@"public_streaming"] || result[@"public_streaming"]==NSNull.null),"runtime_worker_result_mismatch");
    NSDictionary *plan=object(result[@"plan"]),*contract=object(plan[@"runtime_weight_contract"]),*hybrid=object(result[@"hybrid"]),*metrics=object(hybrid[@"runtime_weight"]);
    id selected=contract[@"executor_backend"],backend=metrics[@"executor_backend"];
    require((selected==NSNull.null && backend==NSNull.null) || equal(selected,backend),"runtime_worker_receipt_mismatch");
    for(NSString *key in @[@"data_path",@"partition_axis"])
        require([contract[key] isKindOfClass:NSString.class] && equal(contract[key],metrics[key]),"runtime_worker_receipt_mismatch");
    id ane=metrics[@"ane_channels"],gpu_channels=metrics[@"gpu_channels"],hybrid_blocks=metrics[@"hybrid_blocks_session_total"],
        gpu_blocks=metrics[@"gpu_blocks_session_total"],fallback_blocks=metrics[@"fallback_blocks_session_total"];
    for(id count in @[ane?:NSNull.null,gpu_channels?:NSNull.null,hybrid_blocks?:NSNull.null,gpu_blocks?:NSNull.null,fallback_blocks?:NSNull.null])
        require(whole(count,0,9007199254740991.),"runtime_worker_receipt_mismatch");
    require([metrics[@"backend_fallback_reason"] isKindOfClass:NSString.class] && [metrics[@"failure_reason"] isKindOfClass:NSString.class] &&
        (boolean(hybrid[@"runtime_failed"],false) || boolean(hybrid[@"runtime_failed"],true)),"runtime_worker_receipt_mismatch");
    NSString *reason=metrics[@"backend_fallback_reason"];
    if([metrics[@"failure_reason"] length])reason=metrics[@"failure_reason"];
    const bool private_selected=equal(selected,@"private_ane"),private_executed=private_selected && [hybrid_blocks doubleValue]>0;
    const int width=equal(request[@"model"],@"qwen-image-2.1")?12288:10240;
    if(private_selected) {
        require(equal(contract[@"data_path"],@"w8a8_hadamard") && equal(contract[@"partition_axis"],@"intermediate_channels") &&
            integer(ane,4096) && integer(gpu_channels,width-4096) && equal(plan[@"execution"],@"gpu_ane_experimental"),"runtime_worker_receipt_unqualified");
    } else {
        require(selected==NSNull.null && integer(ane,0) && integer(hybrid_blocks,0) &&
            equal(contract[@"data_path"],@"") && (equal(contract[@"partition_axis"],@"") || equal(contract[@"partition_axis"],@"rows")) &&
            equal(plan[@"execution"],@"gpu"),"runtime_worker_receipt_unqualified");
    }
    if(!private_executed || [fallback_blocks doubleValue]>0 || [hybrid[@"runtime_failed"] boolValue])
        require(reason.length>0,"runtime_worker_fallback_reason_missing");
    return @{@"backend":private_executed?@"private_ane":@"gpu",@"data_path":private_executed?@"w8a8_hadamard":@"",
        @"partition_axis":private_executed?@"intermediate_channels":@"",@"ane_channels":private_executed?@4096:@0,
        @"fallback_reason":reason,@"hybrid_blocks":hybrid_blocks,@"gpu_blocks":gpu_blocks,@"fallback_blocks":fallback_blocks,
        @"selected_backend":selected,@"actual_execution":private_executed?@"gpu_ane":@"gpu",
        @"partial_fallback":@(private_executed && ([fallback_blocks doubleValue]>0 || [hybrid[@"runtime_failed"] boolValue]))};
}
inline int generate(const char *path) {
    NSMutableDictionary *reply=nil;int code=1;
    cancelled=false;
    try {
        NSDictionary *input=validate(read_input_object(path));
        reply=terminal(input,tc::runtime_build_identity());reply[@"runtime_options"]=input[@"runtime_options"];
        reply[@"runtime_receipt"]=NSNull.null;
        NSDictionary *request=input[@"native_request_v2"];
        verify_environment(input);verify_descriptor(input);require_fresh_output(request);verify_hardware();
        struct Signals {
            using Handler=void(*)(int);Handler interrupt,terminate;
            Signals():interrupt(std::signal(SIGINT,signal_stop)),terminate(std::signal(SIGTERM,signal_stop)){}
            ~Signals(){std::signal(SIGINT,interrupt);std::signal(SIGTERM,terminate);}
        } signals;
        NSString *json=[[NSString alloc] initWithData:canonical_request(request) encoding:NSUTF8StringEncoding];
        char *result=nullptr,*error=nullptr;
        // Plan uses the normal parser/module gate before creating a model
        // worker; the generate call revalidates it again without a bypass.
        int status=tc_plan_json(json.UTF8String,&result,&error);(void)consume(status,result,error);
        require(!cancelled,"worker_cancelled");
        tc_engine *raw=nullptr;error=nullptr;
        status=tc_engine_create_model_worker([request[@"model"] UTF8String],[input[@"model_installation_ref"] fileSystemRepresentation],&raw,&error);
        std::unique_ptr<tc_engine,decltype(&tc_engine_free)> engine(raw,tc_engine_free);
        std::unique_ptr<char,decltype(&tc_string_free)> creation_error(error,tc_string_free);
        if(status)throw std::runtime_error(error?error:"worker_engine_create_failed");
        require(engine!=nullptr,"worker_engine_missing");require(!cancelled,"worker_cancelled");
        struct CancellationRelay {
            std::atomic<bool> stopping{false};std::thread worker;
            explicit CancellationRelay(tc_engine *engine):worker([this,engine]{
                while(!stopping.load(std::memory_order_relaxed)) {
                    if(cancelled.load(std::memory_order_relaxed))tc_engine_cancel(engine);
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                }
            }){}
            ~CancellationRelay(){stopping.store(true,std::memory_order_relaxed);worker.join();}
        } cancellation(engine.get());
        Events events(input,tc::runtime_build_identity());events.begin_runtime();result=nullptr;error=nullptr;
        status=tc_engine_generate(engine.get(),json.UTF8String,Events::progress,&events,&result,&error);
        NSDictionary *generated=consume(status,result,error);require(!cancelled,"worker_cancelled");
        NSDictionary *actual=receipt(generated,request),*artifact=generation_artifact(request,cancelled);
        require(!cancelled,"worker_cancelled");
        reply[@"status"]=@"succeeded";reply[@"runtime_receipt"]=actual;reply[@"result"]=generated;reply[@"artifact"]=artifact;code=0;
    } catch(const std::exception &error) {
        if(!reply){std::cerr<<error.what()<<'\n';return 1;}
        const bool stopped=cancelled.load(std::memory_order_relaxed);
        reply[@"status"]=stopped?@"cancelled":@"error";
        reply[@"error"]=@{@"code":stopped?@"worker_cancelled":@"worker_runtime_generate_failed",@"message":@(error.what())};code=stopped?2:1;
    }
    NSData *bytes=canonical_request(reply);
    std::cout.write(static_cast<const char *>(bytes.bytes),std::streamsize(bytes.length));std::cout<<'\n';return code;
}
} // namespace tc_worker::runtime
