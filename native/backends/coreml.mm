#include "coreml.hpp"
namespace tc {
CoreMLBranch::CoreMLBranch(const std::filesystem::path&path,int rows):output_storage_(mx::contiguous(mx::zeros({1,rows,3072},mx::float16))),rows_(rows){
 require(path.extension()==".mlmodelc"&&std::filesystem::is_directory(path),"expected compiled Core ML artifact: "+path.string());
 auto config=[MLModelConfiguration new];config.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
 config.functionName=@"main";
 NSError*error=nil;model_=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(path.c_str())] configuration:config error:&error];
 require(model_!=nil,"Core ML load failed: "+std::string(error?error.localizedDescription.UTF8String:"unknown"));
 auto input=model_.modelDescription.inputDescriptionsByName[@"x"].multiArrayConstraint;
 auto output=model_.modelDescription.outputDescriptionsByName[@"y"].multiArrayConstraint;
 NSArray*shape=@[@1,@3072,@1,@(rows)];
 require(input&&output&&[input.shape isEqual:shape]&&[output.shape isEqual:shape]&&input.dataType==MLMultiArrayDataTypeFloat16&&output.dataType==MLMultiArrayDataTypeFloat16,"Core ML feature ABI mismatch");
 mx::eval(output_storage_);require(output_storage_.data_size()==size_t(rows)*3072&&output_storage_.flags().row_contiguous,"Core ML output backing must be fully materialized and contiguous");
 output_=[[MLMultiArray alloc]initWithDataPointer:output_storage_.data<mx::float16_t>() shape:shape dataType:MLMultiArrayDataTypeFloat16 strides:@[@(rows*3072),@1,@(rows*3072),@3072] deallocator:^(void*){} error:&error];
 require(output_!=nil,"Core ML backing allocation failed");options_=[MLPredictionOptions new];options_.outputBackings=@{@"y":output_};
}
Tensor CoreMLBranch::predict(const Tensor&input,int actual){
 // Input has been materialized before GPU attention submission. No writable alias
 // is exposed to callers; output storage is leased until the block completes.
 auto begin=Clock::now();NSError*error=nil;
 MLMultiArray*in=[[MLMultiArray alloc]initWithDataPointer:(void*)input.data<mx::float16_t>() shape:@[@1,@3072,@1,@(rows_)] dataType:MLMultiArrayDataTypeFloat16 strides:@[@(rows_*3072),@1,@(rows_*3072),@3072] deallocator:^(void*){} error:&error];
 require(in!=nil,"Core ML input binding failed");
 auto provider=[[MLDictionaryFeatureProvider alloc]initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:in]} error:&error];
 auto result=[model_ predictionFromFeatures:provider options:options_ error:&error];
 require(result!=nil,"Core ML prediction failed: "+std::string(error?error.localizedDescription.UTF8String:"unknown"));
 auto actual_output=[result featureValueForName:@"y"].multiArrayValue;
 require(actual_output!=nil,"missing Core ML output");
 if(actual_output.dataPointer!=output_.dataPointer){
  copied_bytes+=uint64_t(rows_)*3072*2;
  // Strides can differ when the framework declines the caller output backing.
  for(int row=0;row<rows_;++row)for(int c=0;c<3072;++c){
   size_t offset=row*[actual_output.strides[3] unsignedLongLongValue]+c*[actual_output.strides[1] unsignedLongLongValue];
   ((uint16_t*)output_storage_.data<mx::float16_t>())[size_t(row)*3072+c]=((uint16_t*)actual_output.dataPointer)[offset];
  }
 }
 // The model coordinator and per-block eval guarantee that the prior
 // consumer has completed before this branch writes its next output. Core ML
 // writes directly into an MLX-owned shared buffer; no tensor escapes the block.
 ++calls;seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
 return slice_axis(output_storage_,1,0,actual);
}
HybridSession::HybridSession(const std::filesystem::path&file,const std::filesystem::path&model,int tokens,const Event&event,std::atomic<bool>&cancelled,int warmups):manifest(file.string()){
 auto begin=Clock::now();auto d=read_json(file);
 require([d[@"schema_version"] isKindOfClass:NSNumber.class]&&[d[@"shape"] isKindOfClass:NSDictionary.class]&&[d[@"source"] isKindOfClass:NSDictionary.class]&&[d[@"artifacts"] isKindOfClass:NSDictionary.class],"invalid hybrid manifest containers");
 require([d[@"shape"][@"K"] isKindOfClass:NSNumber.class]&&[d[@"shape"][@"N"] isKindOfClass:NSNumber.class]&&[d[@"source"][@"checkpoint_bytes"] isKindOfClass:NSNumber.class],"invalid hybrid manifest numbers");
 require([d[@"schema_version"] intValue]==2,"hybrid requires manifest schema 2");
 require([d[@"shape"][@"K"] intValue]==3072&&[d[@"shape"][@"N"] intValue]==3072,"hybrid hidden dimension mismatch");
 NSArray*buckets=d[@"shape"][@"buckets"];require([buckets isKindOfClass:NSArray.class]&&buckets.count==1&&[buckets[0] isKindOfClass:NSNumber.class],"native hybrid requires a single fixed bucket");rows=[buckets[0] intValue];
 require(rows>=tokens&&rows<=8192,"Core ML token bucket cannot serve this request");
 auto checkpoint=model/"transformer/diffusion_pytorch_model.safetensors";
 std::filesystem::path source=string_value(d[@"source"],@"checkpoint");
 require(std::filesystem::equivalent(source,checkpoint)&&[d[@"source"][@"checkpoint_bytes"] unsignedLongLongValue]==std::filesystem::file_size(checkpoint),"artifact checkpoint provenance mismatch");
 // Existing local manifests lack a full source SHA; explicitly research-only.
 for(int i=0;i<20;++i){tc::checkpoint(cancelled);event("coreml_load",i,20);NSString*k=[NSString stringWithFormat:@"%d",i];auto relative=string_value(d[@"artifacts"][k],@"int8_pc");require(!relative.empty(),"Core ML manifest missing block");auto path=file.parent_path()/relative;require(std::filesystem::weakly_canonical(path).string().starts_with(std::filesystem::weakly_canonical(file.parent_path()).string()+"/"),"artifact path escapes manifest directory");branches_.push_back(std::make_unique<CoreMLBranch>(path,rows));}
 if(warmups){auto input=mx::zeros({1,rows,3072},mx::float16);mx::eval(input);for(int iteration=0;iteration<warmups;++iteration)for(int block=0;block<20;++block){tc::checkpoint(cancelled);event("coreml_warmup",iteration*20+block,warmups*20);auto result=branches_[block]->predict(input,rows);mx::eval(result);}}
 load_seconds=std::chrono::duration<double>(Clock::now()-begin).count();
}
Tensor HybridSession::predict(int block,const Tensor&input){return branches_.at(block)->predict(input,input.shape(1));}
NSDictionary *HybridSession::metrics()const{uint64_t calls=0,copied=0;double seconds=0;for(auto&b:branches_){calls+=b->calls;copied+=b->copied_bytes;seconds+=b->seconds;}return @{@"load_seconds":@(load_seconds),@"prediction_seconds_session_total":@(seconds),@"calls_session_total":@(calls),@"bucket":@(rows),@"compute_units":@"cpuAndNeuralEngine",@"observed_ane_residency":@"unknown",@"output_copy_bytes_session_total":@(copied),@"provenance":@"local checkpoint path+size; source SHA absent in legacy artifact; experimental only"};}
}
