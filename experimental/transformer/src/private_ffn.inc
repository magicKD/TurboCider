// Experimental only. In-memory MIL / IOSurface mechanism: maderix/ANE (MIT),
// extended by mac_local_ai; see notes/THIRD_PARTY.md. No production linkage.
class PrivateFFN {
 public:
    PrivateFFN(const std::string &path,int m,int k,int n,void *input,void *output)
      : m_(m),k_(k),n_(n),input_((__fp16*)input),output_((__fp16*)output) {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW);
        Class dc=NSClassFromString(@"_ANEInMemoryModelDescriptor"),mc=NSClassFromString(@"_ANEInMemoryModel"),ic=NSClassFromString(@"_ANEIOSurfaceObject"),rc=NSClassFromString(@"_ANERequest");
        if(!dc||!mc||!ic||!rc)throw std::runtime_error("private ANE classes unavailable");
        NSString *base=[[NSString stringWithUTF8String:path.c_str()] stringByDeletingLastPathComponent];
        NSData *mil=[NSData dataWithContentsOfFile:[base stringByAppendingPathComponent:@"private.mil"]];
        NSMutableDictionary *weights=[NSMutableDictionary dictionary];
        for(NSString *name in @[@"up",@"gate",@"down"]) {
            NSData *blob=[NSData dataWithContentsOfFile:[base stringByAppendingPathComponent:[name stringByAppendingString:@".bin"]]];
            if(!blob)throw std::runtime_error("missing private FFN weight blob");
            weights[[NSString stringWithFormat:@"@model_path/weights/%@.bin",name]]=@{@"offset":@0,@"data":blob};
        }
        id desc=((id(*)(Class,SEL,id,id,id))objc_msgSend)(dc,@selector(modelWithMILText:weights:optionsPlist:),mil,weights,nil);
        model_=((id(*)(Class,SEL,id))objc_msgSend)(mc,@selector(inMemoryModelWithDescriptor:),desc);
        if(!model_)throw std::runtime_error("private model failed");
        NSString *identifier=((id(*)(id,SEL))objc_msgSend)(model_,@selector(hexStringIdentifier));
        dir_=[NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        auto fm=[NSFileManager defaultManager];
        [fm createDirectoryAtPath:[dir_ stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[dir_ stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for(NSString *name in @[@"up",@"gate",@"down"])
            [weights[[NSString stringWithFormat:@"@model_path/weights/%@.bin",name]][@"data"] writeToFile:[dir_ stringByAppendingPathComponent:[NSString stringWithFormat:@"weights/%@.bin",name]] atomically:YES];
        NSError *error=nil;auto t=Clock::now();
        BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model_,@selector(compileWithQoS:options:error:),21,@{},&error);
        double compile=milliseconds(Clock::now()-t);if(!ok)throw std::runtime_error("private FFN compile failed");
        t=Clock::now();ok=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model_,@selector(loadWithQoS:options:error:),21,@{},&error);
        if(!ok)throw std::runtime_error("private FFN load failed");
        std::cerr << "private_setup " << path << " compile_ms=" << compile << " load_ms=" << milliseconds(Clock::now()-t) << "\n";
        in_=surface(size_t(m)*k*2);out_=surface(size_t(m)*n*2);
        if(!in_||!out_)throw std::runtime_error("private IOSurface allocation failed");
        id wi=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ic,@selector(objectWithIOSurface:),in_);
        id wo=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ic,@selector(objectWithIOSurface:),out_);
        request_=((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(rc,@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),@[wi],@[@0],@[wo],@[@0],nil,nil,@0);
        if(!request_)throw std::runtime_error("private request failed");
    }
    ~PrivateFFN(){
        NSError *e=nil;
        if(model_)((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model_,@selector(unloadWithQoS:error:),21,&e);
        if(in_)CFRelease(in_);if(out_)CFRelease(out_);
        if(dir_)[[NSFileManager defaultManager] removeItemAtPath:dir_ error:nil];
    }
    double run(){
        auto t=Clock::now();IOSurfaceLock(in_,0,nullptr);auto p=(__fp16*)IOSurfaceGetBaseAddress(in_);
        for(int i=0;i<m_;++i)for(int j=0;j<k_;++j)p[size_t(j)*m_+i]=input_[size_t(i)*k_+j];
        IOSurfaceUnlock(in_,0,nullptr);NSError *e=nil;
        BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(model_,@selector(evaluateWithQoS:options:request:error:),21,@{},request_,&e);
        if(!ok)throw std::runtime_error("private FFN evaluation failed");
        IOSurfaceLock(out_,kIOSurfaceLockReadOnly,nullptr);p=(__fp16*)IOSurfaceGetBaseAddress(out_);
        for(int i=0;i<m_;++i)for(int j=0;j<n_;++j)output_[size_t(i)*n_+j]=p[size_t(j)*m_+i];
        IOSurfaceUnlock(out_,kIOSurfaceLockReadOnly,nullptr);return milliseconds(Clock::now()-t);
    }
 private:
    static IOSurfaceRef surface(size_t bytes){return IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});}
    int m_,k_,n_;__fp16 *input_,*output_;id model_=nil,request_=nil;NSString *dir_=nil;IOSurfaceRef in_=nullptr,out_=nullptr;
};
