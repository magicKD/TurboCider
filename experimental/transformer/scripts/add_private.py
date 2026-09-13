from pathlib import Path
p=Path(__file__).resolve().parents[1];s=(p/'src/runner.mm').read_text()
s='#import <IOSurface/IOSurface.h>\n#import <objc/message.h>\n#import <dlfcn.h>\n'+s
s=s.replace('class CoreMLGemm {',(p/'src/private_ffn.inc').read_text()+'\nclass CoreMLGemm {',1)
s=s.replace('        MLModelConfiguration *configuration = [MLModelConfiguration new];','''        if(std::getenv("TC_PRIVATE_FFN")) {
            if(!outputBacking)throw std::runtime_error("private FFN requires caller output backing");
            private_=std::make_unique<PrivateFFN>(modelPath,M,K,N,input,outputBacking);
            NSError *err=nil;
            outputBacking_=[[MLMultiArray alloc] initWithDataPointer:outputBacking shape:@[@(M),@(N)] dataType:MLMultiArrayDataTypeFloat16 strides:@[@(N),@1] deallocator:^(void*){} error:&err];
            output_=outputBacking_;usedOutputBacking_=true;
            return;
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];''',1)
a=s.index('class CoreMLGemm {');b=s.index('template <typename Work>',a)
t=s[a:b].replace('    double run() {','    double run() {\n        if(private_)return private_->run();',1).replace('    MLModel *model_ = nil;','    std::unique_ptr<PrivateFFN> private_;\n    MLModel *model_ = nil;')
s=s[:a]+t+s[b:];(p/'src/runner_private.mm').write_text(s)
