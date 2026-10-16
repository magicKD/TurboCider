// Original bounded ordinary-block observer experiment. No borrowed reference
// escapes the callback; only owned CPU BF16 copies are optionally serialized.
#include "models/qwen21/pipeline.hpp"
#include "turbocider/turbocider.h"
#include "qwen_shared_abi.h"
#include "qwen_shared_compute.h"
#include "ffn_cpu_witness.hpp"
#include <fstream>
#include <zlib.h>

using namespace tc;
namespace block_lab {
using tcq::check;
NSString *string(const std::string &value){return [NSString stringWithUTF8String:value.c_str()];}
std::string string(NSString *value){return std::string(value.UTF8String);}
NSString *hash(const Tensor &value) {
    auto bytes=mx::view(mx::contiguous(value),mx::uint8);mx::eval(bytes);mx::synchronize();
    return tcq::sha(bytes.data<uint8_t>(),bytes.size());
}
Tensor rotate(const Tensor &x,const Tensor &cosine,const Tensor &sine) {
    auto shape=x.shape(),paired_shape=shape;paired_shape.back()/=2;paired_shape.push_back(2);
    auto paired=mx::reshape(mx::astype(x,mx::float32),paired_shape);
    auto a=mx::squeeze(slice_axis(paired,-1,0,1),-1),b=mx::squeeze(slice_axis(paired,-1,1,2),-1);
    auto c=mx::reshape(cosine,{1,1,shape[2],shape[3]/2}),s=mx::reshape(sine,{1,1,shape[2],shape[3]/2});
    return mx::astype(mx::reshape(mx::stack({a*c-b*s,a*s+b*c},-1),shape),x.dtype());
}
using Function=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
Function front(const qwen21::DiagnosticBlockObservation &o) {
    const auto &w=o.weights;
    return mx::compile(Function([&w](const std::vector<Tensor>&a){
        const std::string p="transformer_blocks.0.";
        auto input=mx::fast::layer_norm(a[0],{},{},1e-6f)*(Tensor(1.f,a[0].dtype())+a[1]);
        auto q=linear(input,w,p+"attn.to_q"),k=linear(input,w,p+"attn.to_k");
        auto v=heads(linear(input,w,p+"attn.to_v"),32,128);
        q=rotate(mx::fast::rms_norm(heads(q,32,128),w.at(p+"attn.norm_q.weight"),1e-6f),a[5],a[6]);
        k=rotate(mx::fast::rms_norm(heads(k,32,128),w.at(p+"attn.norm_k.weight"),1e-6f),a[5],a[6]);
        auto attention=attend(q,mx::concatenate({a[7],k},2),mx::concatenate({a[8],v},2));
        auto hidden=a[0]+mx::tanh(a[2])*linear(attention,w,p+"attn.to_out.0");
        input=mx::fast::layer_norm(hidden,{},{},1e-6f)*(Tensor(1.f,hidden.dtype())+a[3]);
        return std::vector<Tensor>{hidden,input};
    }));
}
Function finish() {
    return mx::compile(Function([](const auto&a){return std::vector<Tensor>{a[0]+mx::tanh(a[1])*a[2]};}));
}
std::vector<float> floats(const Tensor &value) {
    auto array=mx::contiguous(mx::astype(value,mx::float32));mx::eval(array);mx::synchronize();
    return {array.data<float>(),array.data<float>()+array.size()};
}
std::vector<uint16_t> codes(const Tensor &value) {
    check(value.dtype()==mx::bfloat16,"full BF16 source/output required");
    auto array=mx::view(mx::contiguous(value),mx::uint16);mx::eval(array);mx::synchronize();
    return {array.data<uint16_t>(),array.data<uint16_t>()+array.size()};
}
NSDictionary *oracle(const tc_witness::Result &value) {
    NSMutableArray *rows=[NSMutableArray array],*cols=[NSMutableArray array],*values=[NSMutableArray array];
    for(auto row:value.rows)[rows addObject:@(row)];for(auto col:value.cols)[cols addObject:@(col)];
    for(size_t r=0;r<4;++r){NSMutableArray *part=[NSMutableArray array];for(size_t c=0;c<4;++c)[part addObject:@(value.output[r*4+c])];[values addObject:part];}
    return @{@"rows":rows,@"cols":cols,@"values":values};
}
bool block_pass(NSDictionary *value){return [value[@"finite"] boolValue] && [value[@"relative_l2"] doubleValue]<=.001 && [value[@"cosine"] doubleValue]>=.999;}
struct Saved {std::vector<uint16_t> input,reference;};

void measure(const qwen21::DiagnosticBlockObservation &o,NSDictionary *config,NSMutableDictionary *report,
             const char *path,Saved &saved) {
    check(o.layer==0 && o.steady && o.arguments.size()==9 && o.outputs.size()==1 && o.prefix_tokens>0 && o.prefix_tokens<=512 && !o.weights.has_runtime_loras(),
          "bounded ordinary base steady layer0 observation required");
    check(o.arguments[0].shape()==mx::Shape{1,1024,4096} && o.arguments[0].dtype()==mx::bfloat16,"complete original BF16 ordinary block input required");
    std::vector<NSString*> before;for(const auto &value:o.arguments)before.push_back(hash(value));
    NSString *original_hash=hash(o.outputs[0]);auto block_reference=floats(o.outputs[0]);
    report[@"ordinary_argument_sha256"]=[NSMutableArray array];for(auto value:before)[report[@"ordinary_argument_sha256"] addObject:value];
    report[@"ordinary_block_output_sha256"]=original_hash;report[@"prefix_tokens"]=@(o.prefix_tokens);
    if(config[@"expected_ordinary_argument_sha256"])
        check([report[@"ordinary_argument_sha256"] isEqual:config[@"expected_ordinary_argument_sha256"]] &&
              [original_hash isEqual:config[@"expected_ordinary_block_sha256"]],"actual ordinary context differs from explicit earlier observation");
    report[@"source_step"]=@1;report[@"source_layer"]=@0;report[@"shape_mhf"]=@[@1024,@4096,@12288];report[@"share"]=@2048;
    auto gu=o.weights.at("transformer_blocks.0.img_mlp.gate_up.weight"),down=o.weights.at("transformer_blocks.0.img_mlp.out.weight");
    check(gu.dtype()==mx::bfloat16 && down.dtype()==mx::bfloat16 && gu.shape()==mx::Shape{24576,4096} && down.shape()==mx::Shape{4096,12288},"actual original dense FFN weight ABI differs");
    auto gu_code=codes(gu),down_code=codes(down);
    check([tcq::sha(gu_code.data(),gu_code.size()*2) isEqual:config[@"expected_gate_up_sha256"]] &&
          [tcq::sha(down_code.data(),down_code.size()*2) isEqual:config[@"expected_down_sha256"]],"actual resident weight source differs from frozen private constants");
    auto pg=mx::contiguous(mx::concatenate({mx::slice(gu,{0,0},{2048,4096}),mx::slice(gu,{12288,0},{14336,4096})},0));
    auto pd=mx::contiguous(mx::slice(down,{0,0},{4096,2048}));
    auto rg=mx::contiguous(mx::concatenate({mx::slice(gu,{2048,0},{12288,4096}),mx::slice(gu,{14336,0},{24576,4096})},0));
    auto rd=mx::contiguous(mx::slice(down,{0,2048},{4096,12288}));mx::eval(pg,pd,rg,rd);mx::synchronize();
    auto front_fn=front(o),finish_fn=finish();
    auto full_fn=mx::compile(Function([gu,down](const auto&a){return std::vector<Tensor>{tcq::formula(a[0],gu,down)};}));
    auto suffix_fn=mx::compile(Function([rg,rd](const auto&a){return std::vector<Tensor>{tcq::formula(a[0],rg,rd)};}));
    auto prefix_fn=mx::compile(Function([pg,pd](const auto&a){return std::vector<Tensor>{tcq::formula(a[0],pg,pd)};}));
    auto join_fn=mx::compile(Function([](const auto&a){return std::vector<Tensor>{mx::astype(mx::astype(a[0],mx::float32)+mx::astype(a[1],mx::float32),mx::bfloat16)};}));
    auto producer=front_fn(o.arguments);mx::eval(producer);mx::synchronize();
    auto ffn_reference_array=full_fn({producer[1]})[0];mx::eval(ffn_reference_array);mx::synchronize();
    auto ffn_reference=floats(ffn_reference_array),prefix_reference=floats(prefix_fn({producer[1]})[0]);
    auto staged=finish_fn({producer[0],o.arguments[4],ffn_reference_array})[0];mx::eval(staged);mx::synchronize();
    auto staged_values=floats(staged);report[@"staged_gpu_replay_gate"]=tcq::metric(block_reference.data(),staged_values.data(),block_reference.size());
    check(block_pass(report[@"staged_gpu_replay_gate"]),"fixed ordinary-block replay gate failed; no candidate benchmark");
    saved.input=codes(producer[1]);saved.reference=codes(ffn_reference_array);
    report[@"exposed_ffn_input_sha256"]=tcq::sha(saved.input.data(),saved.input.size()*2);
    report[@"full_ffn_reference_sha256"]=tcq::sha(saved.reference.data(),saved.reference.size()*2);
    if(config[@"expected_input_sha256"])
        check([report[@"exposed_ffn_input_sha256"] isEqual:config[@"expected_input_sha256"]] &&
              [report[@"full_ffn_reference_sha256"] isEqual:config[@"expected_reference_sha256"]],"actual exposed FFN source differs from explicit earlier observation");
    if(config[@"reuse_artifacts"]){
        check([report[@"exposed_ffn_input_sha256"] isEqual:config[@"reuse_input_sha256"]] &&
              [report[@"full_ffn_reference_sha256"] isEqual:config[@"reuse_reference_sha256"]],"recreated actual FFN source differs from explicitly reused artifacts");
    }
    auto independent_full=tc_witness::selected(saved.input,gu_code,down_code,1024,4096,12288,0,12288);
    auto independent_prefix=tc_witness::selected(saved.input,gu_code,down_code,1024,4096,12288,0,2048);
    NSDictionary *full_oracle=oracle(independent_full),*prefix_oracle=oracle(independent_prefix);
    report[@"independent_full_reference_gate"]=tcq::sample_metric(full_oracle,ffn_reference.data());
    check(tcq::passed(report[@"independent_full_reference_gate"]),"independent full FFN reference rejected");
    // A runtime-argument control: zero FFN must yield the actual residual;
    // zero private partial must yield the actual BF16 GPU remainder.
    auto zero_bf=mx::zeros({1,1024,4096},mx::bfloat16),zero_half=mx::zeros({1,1024,4096},mx::float16);mx::eval(zero_bf,zero_half);
    auto no_ffn=finish_fn({producer[0],o.arguments[4],zero_bf})[0];mx::eval(no_ffn);mx::synchronize();
    auto hidden_values=floats(producer[0]),no_ffn_values=floats(no_ffn);
    auto suffix_ready=suffix_fn({producer[1]})[0];auto no_private=join_fn({suffix_ready,zero_half})[0];mx::eval(no_private);mx::synchronize();
    auto tail_values=floats(suffix_ready),no_private_values=floats(no_private);
    auto finish_zero=tcq::metric(hidden_values.data(),no_ffn_values.data(),hidden_values.size());
    auto partial_zero=tcq::metric(tail_values.data(),no_private_values.data(),tail_values.size());
    check([finish_zero[@"max_abs"] doubleValue]==0 && [partial_zero[@"max_abs"] doubleValue]==0,"compiled runtime-argument zero controls failed");
    size_t changed=0;for(size_t i=0;i<staged_values.size();++i)changed+=staged_values[i]!=no_ffn_values[i];
    check(changed>0,"FFN cannot be omitted from complete block candidate");
    report[@"runtime_argument_controls"]=@{@"zero_ffn_yields_original_residual":finish_zero,@"zero_private_yields_original_tail":partial_zero,
        @"nonzero_full_ffn_changes_block_elements":@(changed),@"used_for_speed_ratio":@NO};
    gu_code.clear();gu_code.shrink_to_fit();down_code.clear();down_code.shrink_to_fit();

    tcq::Private private_session;std::vector<IOSurfaceRef> surfaces;NSMutableArray *owners=[NSMutableArray array];
    auto adoption=std::make_shared<tcq::Adoption>();std::atomic<bool> evaluation_unsafe{false};bool gpu_pending=false,mlx_pending=false;
    try {
    private_session.load(config[@"model_directory"],report,false);
    report[@"free_bytes_after_private_load"]=@(std::filesystem::space(string(config[@"output_directory"])).available);
    check([report[@"free_bytes_after_private_load"] unsignedLongLongValue]>=128ull*1024*1024,"setup crossed disk reserve before timed block work");
    auto in=tcq::port(report[@"input_descriptor"]),out=tcq::port(report[@"output_descriptor"]);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();report[@"gpu_name"]=device.name;
    IOSurfaceRef input=tcq::surface(in.total);surfaces.push_back(input);IOSurfaceRef output=tcq::surface(out.total);surfaces.push_back(output);
    {
    id<MTLBuffer> ib=[device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(input) length:IOSurfaceGetAllocSize(input) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> ob=[device newBufferWithBytesNoCopy:IOSurfaceGetBaseAddress(output) length:IOSurfaceGetAllocSize(output) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> si=[device newBufferWithLength:ib.length options:MTLResourceStorageModeShared],so=[device newBufferWithLength:ob.length options:MTLResourceStorageModeShared];
    check(ib&&ob&&si&&so,"actual bounded block feature buffers required");[owners addObjectsFromArray:@[ib,ob,si,so]];
    auto shared_view=tcq::adopt(ob,out,adoption,mlx_pending),copy_view=tcq::adopt(so,out,adoption,mlx_pending);
    id request=[private_session.request_class requestWithInputs:@[[private_session.io_class objectWithIOSurface:input]] inputIndices:@[@0]
        outputs:@[[private_session.io_class objectWithIOSurface:output]] outputIndices:@[@0] perfStats:nil procedureIndex:@0];
    check(request&&[request validate],"complete block request failed validation");report[@"initial_identity"]=tcq::identity(request,input,output,ib,ob);
    NSError *error=nil;id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:tcq::pack_shader] options:nil error:&error];
    check(library&&queue,"actual BF16 block pack unavailable");
    id<MTLComputePipelineState> pack_pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack_bf16"] error:&error];check(pack_pipeline,"pack pipeline unavailable");
    tcq::Worker worker;NSArray *arms=@[@"gpu-original",@"gpu-staged",@"copy-hybrid",@"shared-serial",@"shared-hybrid"];
    uint32_t repeats=uint32_t(tcq::integer(config,@"measured_groups"));check(repeats==10||repeats==30,"bounded balanced group count required");
    report[@"measured_groups"]=@(repeats);report[@"samples"]=[NSMutableArray array];
    for(uint32_t group=0;group<repeats+3;++group)for(uint32_t position=0;position<5;++position) {
        NSString *phase=group==0?@"first":group<3?@"warmup":@"measured";
        uint32_t trial=group==0?0:group<3?group-1:group-3,order=group<3?group:trial;
        NSString *arm=arms[(order%5+(order/5%2?4-position:position))%5];
        bool original=[arm isEqual:@"gpu-original"],staged_gpu=[arm isEqual:@"gpu-staged"],copied=[arm isEqual:@"copy-hybrid"],hybrid=[arm hasSuffix:@"hybrid"];
        mx::synchronize();if(!original&&!staged_gpu){tcq::lock(input);tcq::poison(ib.contents,in.total,ib.length);tcq::unlock(input);
            tcq::lock(output);tcq::poison(ob.contents,out.total,ob.length);tcq::unlock(output);tcq::poison(si.contents,in.total,si.length);tcq::poison(so.contents,out.total,so.length);}
        std::vector<tcq::Event> events;auto note=[&](NSString*n,uint64_t begin){events.push_back({n,begin,tcq::now()-begin});};
        std::optional<Tensor> result,ffn_output,remainder;NSDictionary *before_identity=nil,*after_identity=nil;
        uint64_t started=tcq::now();
        if(original){mlx_pending=true;result.emplace(o.gpu_block(o.arguments)[0]);mx::eval(*result);mx::synchronize();mlx_pending=false;}
        else {
            uint64_t t=tcq::now();mlx_pending=true;auto produced=front_fn(o.arguments);mx::eval(produced);mx::synchronize();mlx_pending=false;note(@"attention-norm-producer-ready",t);
            if(staged_gpu){t=tcq::now();mlx_pending=true;ffn_output.emplace(full_fn({produced[1]})[0]);mx::eval(*ffn_output);mx::synchronize();mlx_pending=false;note(@"full-gpu-ffn-ready",t);}
            else {
                t=tcq::now();tcq::pack(queue,pack_pipeline,produced[1],copied?si:ib,in.row/2,gpu_pending);note(@"gpu-pack-ready",t);
                if(copied){t=tcq::now();tcq::lock(input);std::memcpy(ib.contents,si.contents,in.total);tcq::unlock(input);note(@"cpu-input-feature-copy",t);}
                before_identity=tcq::identity(request,input,output,ib,ob);
                auto evaluate=[&private_session,request,&evaluation_unsafe]()->tcq::Evaluation {
                    @autoreleasepool{NSError*err=nil;bool ok=false;NSString*domain=@"none";NSInteger code=0;uint64_t t=tcq::now();evaluation_unsafe.store(true);
                        @try{ok=[private_session.client evaluateWithModel:private_session.model options:@{@"kANEFPerformanceStatsMask":@0} request:request qos:25 error:&err];
                            if(!ok){domain=err.domain?:@"unknown";code=err.code;}}
                        @catch(NSException*err){domain=err.name;}if(ok)evaluation_unsafe.store(false);return {ok,t,tcq::now()-t,domain,code};}
                };
                std::future<tcq::Evaluation> future;std::optional<tcq::Evaluation> evaluation;
                if(hybrid)future=worker.submit(evaluate);else evaluation.emplace(evaluate());
                t=tcq::now();mlx_pending=true;remainder.emplace(suffix_fn({produced[1]})[0]);mx::eval(*remainder);mx::synchronize();mlx_pending=false;note(@"gpu-remainder-ready",t);
                if(hybrid)evaluation.emplace(future.get());events.push_back({@"private-evaluate-returned",evaluation->start,evaluation->duration});
                check(evaluation->ok,"whole-block private branch failed; no fallback");after_identity=tcq::identity(request,input,output,ib,ob);
                if(copied){t=tcq::now();tcq::lock(output);std::memcpy(so.contents,ob.contents,out.total);tcq::unlock(output);note(@"cpu-output-feature-copy",t);}
                t=tcq::now();mlx_pending=true;ffn_output.emplace(join_fn({*remainder,copied?copy_view:shared_view})[0]);mx::eval(*ffn_output);mx::synchronize();mlx_pending=false;note(@"gpu-ffn-restore-join-ready",t);
            }
            t=tcq::now();mlx_pending=true;result.emplace(finish_fn({produced[0],o.arguments[4],*ffn_output})[0]);mx::eval(*result);mx::synchronize();mlx_pending=false;note(@"modulation-residual-output-ready",t);
        }
        uint64_t wall=tcq::now()-started;
        check(result->dtype()==mx::bfloat16&&result->shape()==mx::Shape{1,1024,4096},"complete original BF16 block output required");
        auto actual=floats(*result);
        NSMutableDictionary *metrics=[@{@"whole_ordinary_block":tcq::metric(block_reference.data(),actual.data(),actual.size())} mutableCopy];
        bool numeric=block_pass(metrics[@"whole_ordinary_block"]),guards=true;NSDictionary *exact_join=nil;
        if(original)numeric&=[metrics[@"whole_ordinary_block"][@"max_abs"] doubleValue]==0;
        else {
            auto ff=floats(*ffn_output);metrics[@"full_ffn"]=tcq::metric(ffn_reference.data(),ff.data(),ff.size());
            metrics[@"independent_full_ffn"]=tcq::sample_metric(full_oracle,ff.data());numeric&=tcq::passed(metrics[@"full_ffn"])&&tcq::passed(metrics[@"independent_full_ffn"]);
            if(!staged_gpu){auto partial=floats(copied?copy_view:shared_view),tail=floats(*remainder);
                metrics[@"prefix"]=tcq::metric(prefix_reference.data(),partial.data(),partial.size());metrics[@"independent_prefix"]=tcq::sample_metric(prefix_oracle,partial.data());
                numeric&=tcq::passed(metrics[@"prefix"])&&tcq::passed(metrics[@"independent_prefix"]);
                auto output_bits=codes(*ffn_output);std::vector<uint16_t> expected(output_bits.size());size_t mismatches=0;
                for(size_t i=0;i<expected.size();++i){float sum=tail[i]+partial[i];uint32_t bits=0;std::memcpy(&bits,&sum,4);expected[i]=uint16_t((bits+0x7fff+((bits>>16)&1))>>16);mismatches+=expected[i]!=output_bits[i];}
                exact_join=@{@"elements":@(expected.size()),@"exact_mismatches":@(mismatches),@"expected_bf16_sha256":tcq::sha(expected.data(),expected.size()*2),@"actual_bf16_sha256":tcq::sha(output_bits.data(),output_bits.size()*2)};
                guards&=mismatches==0;tcq::lock(input);tcq::lock(output);guards&=tcq::tail(ib.contents,in.total,ib.length)&&tcq::tail(ob.contents,out.total,ob.length)&&tcq::tail(si.contents,in.total,si.length)&&tcq::tail(so.contents,out.total,so.length);tcq::unlock(output);tcq::unlock(input);
            }
        }
        std::sort(events.begin(),events.end(),[](const auto&a,const auto&b){return a.start<b.start;});NSMutableArray*host=[NSMutableArray array];
        for(const auto&e:events)[host addObject:@{@"name":e.name,@"start_ns":@(e.start-started),@"duration_ns":@(e.duration),@"clock_domain":@"same-process-uptime-raw-host"}];
        NSMutableDictionary *sample=[@{@"phase":phase,@"trial":@(trial),@"arm":arm,@"wall_ns":@(wall),@"metrics":metrics,@"numeric_pass":@(numeric),@"guards_pass":@(guards),
            @"output_dtype":@"bfloat16",@"host_events":host,@"external_feature_copy_count":copied?@2:@0,@"external_feature_copy_bytes":copied?@16777216:@0,@"physical_overlap_qualified":@NO} mutableCopy];
        if(exact_join)sample[@"exact_runtime_join"]=exact_join;if(before_identity){sample[@"identity_before"]=before_identity;sample[@"identity_after"]=after_identity;}
        [report[@"samples"] addObject:sample];tcq::save(report,path);check(numeric&&guards,"fixed complete-block/FFN/prefix/FP64/join/guard gate rejected arm");
    }
    report[@"final_identity"]=tcq::identity(request,input,output,ib,ob);
    }
    check(private_session.unload(),"private branch unload failed");
    check(adoption->releases.load()==2&&adoption->pointer_ok.load()&&!gpu_pending&&!mlx_pending&&!evaluation_unsafe.load(),"normal owned block branch lifecycle incomplete");
    [owners removeAllObjects];for(auto value:surfaces)CFRelease(value);surfaces.clear();
    report[@"private_unloaded"]=@YES;report[@"restore_adoption_releases"]=@(adoption->releases.load());report[@"owned_mapping_release_pass"]=@YES;
    } catch(...) {
        (void)private_session.unload();(void)CFBridgingRetain(owners);
        report[@"mapping_quarantined_until_child_exit"]=@YES;throw;
    }
    for(size_t i=0;i<o.arguments.size();++i)check([hash(o.arguments[i]) isEqual:before[i]],"ordinary argument/modulation/cache changed");
    check([hash(o.outputs[0]) isEqual:original_hash],"ordinary original output changed");
    report[@"borrowed_inputs_outputs_unchanged"]=@YES;report[@"status"]=@"measured";
}

void write_tensor(const std::filesystem::path &path,const char *key,const std::vector<uint16_t>&data) {
    check(data.size()==4194304&&!std::filesystem::exists(path),"new complete BF16 artifact only");
    std::string header="{\""+std::string(key)+"\":{\"dtype\":\"BF16\",\"shape\":[1,1024,4096],\"data_offsets\":[0,8388608]},\"__metadata__\":{\"recipe\":\"ordinary-block-exposed-ffn-v1\"}}";
    header.append((8-header.size()%8)%8,' ');uint64_t length=header.size();
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);check(fd>=0,"artifact exclusive create failed");
    auto write_all=[&](const void*pointer,size_t bytes){auto*p=static_cast<const char*>(pointer);while(bytes){ssize_t n=::write(fd,p,bytes);if(n<0&&errno==EINTR)continue;check(n>0,"artifact write failed");p+=n;bytes-=size_t(n);}};
    try{write_all(&length,8);write_all(header.data(),header.size());write_all(data.data(),data.size()*2);check(::close(fd)==0,"artifact close failed");}
    catch(...){::close(fd);throw;}
}
NSDictionary *write_archive(const std::filesystem::path &path,const Saved &saved) {
    constexpr size_t count=4194304;check(saved.input.size()==count&&saved.reference.size()==count&&!std::filesystem::exists(path),"complete new archive only");
    std::vector<uint8_t> planes(count*4);
    for(size_t i=0;i<count;++i){planes[i]=uint8_t(saved.input[i]>>8);planes[count+i]=uint8_t(saved.input[i]);
        planes[2*count+i]=uint8_t(saved.reference[i]>>8);planes[3*count+i]=uint8_t(saved.reference[i]);}
    uLongf length=compressBound(uLong(planes.size()));std::vector<uint8_t> compressed(length);
    check(compress2(compressed.data(),&length,planes.data(),uLong(planes.size()),Z_BEST_COMPRESSION)==Z_OK,"lossless BF16 compression failed");compressed.resize(length);
    NSDictionary *metadata=@{@"schema":@"tc-bf16-planes-zlib-v1",@"shape":@[@1,@1024,@4096],@"dtype":@"BF16",
        @"tensor_order":@[@"input",@"reference"],@"plane_order":@"high-then-low-per-tensor",@"uncompressed_bytes":@(planes.size()),@"compressed_bytes":@(compressed.size()),
        @"input_sha256":tcq::sha(saved.input.data(),saved.input.size()*2),@"reference_sha256":tcq::sha(saved.reference.data(),saved.reference.size()*2),
        @"zlib_version":[NSString stringWithUTF8String:zlibVersion()]};
    NSData *json=[NSJSONSerialization dataWithJSONObject:metadata options:0 error:nil];check(json&&json.length<=65536,"bounded archive metadata");
    uint64_t header=json.length;size_t total=8+json.length+compressed.size();
    check(std::filesystem::space(path.parent_path()).available>=128ull*1024*1024+total+(1<<20),"lossless archive actual-size reserve insufficient; no data overwritten");
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);check(fd>=0,"archive exclusive create failed");
    auto write_all=[&](const void*pointer,size_t bytes){auto*p=static_cast<const char*>(pointer);while(bytes){ssize_t n=::write(fd,p,bytes);if(n<0&&errno==EINTR)continue;check(n>0,"archive write failed");p+=n;bytes-=size_t(n);}};
    try{write_all(&header,8);write_all(json.bytes,json.length);write_all(compressed.data(),compressed.size());check(::close(fd)==0,"archive close failed");}
    catch(...){::close(fd);throw;}
    return metadata;
}
}

int main(int argc,const char**argv) {
    @autoreleasepool {
        if(argc!=3){std::fprintf(stderr,"usage: qwen-block-shared config.json new-report.json\n");return 2;}
        NSMutableDictionary *report=[@{@"schema":@"tc-qwen-ordinary-block-shared-v1",@"status":@"starting",@"production_qualified":@NO,
            @"physical_overlap_qualified":@NO,@"runtime_internal_copy_behavior":@"unknown",@"samples":[NSMutableArray array],@"errors":[NSMutableArray array]} mutableCopy];
        @try {try {
            NSData *bytes=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];tcq::check(bytes&&bytes.length<=1024*1024,"bounded explicit block config required");
            NSDictionary *config=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
            uint64_t seed=tcq::integer(config,@"seed",UINT32_MAX);std::string prompt=block_lab::string(config[@"prompt"]);
            report[@"seed"]=@(seed);report[@"prompt_sha256"]=tcq::sha(prompt.data(),prompt.size());report[@"case_id"]=config[@"case_id"];
            report[@"block_gate"]=@{@"relative_l2_max":@.001,@"cosine_min":@.999};report[@"ffn_gate"]=@{@"relative_l2_max":@.03,@"cosine_min":@.999};
            report[@"source_config_sha256"]=tcq::sha(bytes.bytes,bytes.length);report[@"mlx_version"]=[NSString stringWithUTF8String:mx::version()];
            configure_streams();mx::set_default_device(mx::Device::gpu);mx::enable_compile();std::atomic<bool> cancelled{false};
            qwen21::Session session(block_lab::string(config[@"model_root"]));block_lab::Saved saved;size_t observations=0;bool completed=false;
            session.set_diagnostic_block_observer([&](const qwen21::DiagnosticBlockObservation&o){
                tcq::check(o.layer==0&&observations<2&&o.steady==(observations==1),"bounded prefill/step1 observer sequence required");
                ++observations;if(!o.steady)return;
                block_lab::measure(o,config,report,argv[2],saved);throw qwen21::DiagnosticBlockReplayComplete();
            });
            Request request;request.model="qwen-image-2.1";request.width=request.height=512;request.steps=20;request.seed=uint32_t(seed);request.prompt=prompt;
            request.audio=false;request.execution="gpu";request.residency="component_staged";request.prompt_enhance=false;request.qwen21_dit_cache="off";request.qwen21_dit_cache_explicit=true;
            std::filesystem::path output=block_lab::string(config[@"output_directory"]);request.output=(output/"must-not-generate.png").string();
            const Event events=[](const std::string&phase,int step,int total){
                tcq::check((phase.find("vae")==std::string::npos||phase=="load_qwen21_vae")&&phase!="export","bounded block probe entered image decode/export");
                std::cerr<<phase<<' '<<step<<'/'<<total<<'\n';};
            try{(void)session.generate(request,events,cancelled);}catch(const qwen21::DiagnosticBlockReplayComplete&){completed=true;}
            session.set_diagnostic_block_observer({});session.unload();mx::synchronize();
            tcq::check(completed&&observations==2&&!std::filesystem::exists(request.output),"ordinary request did not stop at bounded block observer");
            char*identity=tc_runtime_build_identity();tcq::check(identity,"loaded runtime identity missing");report[@"loaded_runtime_id"]=block_lab::string(identity);tc_string_free(identity);
            report[@"completed_full_dit_steps"]=@1;report[@"stopped_step"]=@1;report[@"stopped_layer"]=@0;report[@"vae_decode"]=@NO;report[@"image_export"]=@NO;
            if(config[@"reuse_artifacts"]){report[@"raw_ffn_artifacts_written"]=@NO;}
            else if([config[@"lossless_archive"] boolValue]){
                report[@"bf16_archive"]=block_lab::write_archive(output/"ffn-bf16.tcbf16z",saved);
                report[@"lossless_ffn_archive_written"]=@YES;report[@"raw_ffn_artifacts_written"]=@NO;
            }
            else {
                tcq::check(std::filesystem::space(output).available>=128ull*1024*1024+2ull*8388608+(1<<20),"post-execution artifact reserve insufficient; no overwrite");
                block_lab::write_tensor(output/"ffn-input.safetensors","input",saved.input);block_lab::write_tensor(output/"ffn-reference.safetensors","reference",saved.reference);
                report[@"raw_ffn_artifacts_written"]=@YES;
            }
        }catch(const std::exception&error){report[@"status"]=@"failed";[report[@"errors"] addObject:[NSString stringWithUTF8String:error.what()]];}
        }@catch(NSException*error){report[@"status"]=@"objc-exception";[report[@"errors"] addObject:error.name];}
        tcq::save(report,argv[2]);std::puts([report[@"status"] UTF8String]);return [report[@"status"] isEqual:@"measured"]?0:3;
    }
}
