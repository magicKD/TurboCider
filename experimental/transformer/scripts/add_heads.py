from pathlib import Path
p=Path(__file__).resolve().parents[1];s=(p/'src/runner.mm').read_text()
s=s.replace('    int M = 256;', '    int headSuffix = 0;\n    std::string headBackend = "ane";\n    std::string headModel;\n    int M = 256;')
s=s.replace('        if (a == "--mode")', '        if (a == "--head-suffix") o.headSuffix = std::stoi(value());\n        else if (a == "--head-backend") o.headBackend = value();\n        else if (a == "--head-model") o.headModel = value();\n        else if (a == "--mode")')
s=s.replace('    return o;','    if (o.headSuffix && (o.headSuffix >= o.heads || o.headSuffix < 0 || o.attention != "materialized" || o.qkvAneN || o.qkvCpuN)) throw std::runtime_error("head suffix requires materialized all-GPU QKV and 0<suffix<heads");\n    return o;',1)
s=s.replace('template <typename Work>',(p/'src/head_branch.inc').read_text()+'\ntemplate <typename Work>',1)
s=s.replace('        if (o.attention == "sdpa") {','''        if (o.headSuffix) {
            headOutput_ = makeBuffer(device, size_t(o.M)*o.headSuffix*headDim_);
            headBranch_ = std::make_unique<HeadBranch>(o,qkv_->output().contents,headOutput_.contents);
            headWorker_ = std::make_unique<PersistentWorker<HeadBranch>>(*headBranch_,o.headBackend=="cpu");
        }
        if (o.attention == "sdpa") {''',1)
s=s.replace('    BlockTimes runGPU() {','    BlockTimes runGPU() {\n        heterogeneousActive_ = false;',1)
s=s.replace('        BlockTimes result;\n        const std::string &layout =','        heterogeneousActive_ = true;\n        BlockTimes result;\n        const std::string &layout =',1)
s=s.replace('        if (o_.ropeTheta > 0.0f && o_.attention != "materialized") {','''        const bool splitHeads = heterogeneousActive_ && o_.headSuffix;
        const int gpuHeads = splitHeads ? o_.heads-o_.headSuffix : o_.heads;
        std::atomic<bool> headGate{false};
        if(splitHeads) {
            [cb commit];[cb waitUntilCompleted];checkCommandBuffer(cb);
            headWorker_->submit(&headGate);
            cb=[queue_ commandBuffer];
        }
        if (o_.ropeTheta > 0.0f && o_.attention != "materialized") {''',1)
s=s.replace('for (int head = 0; head < o_.heads; ++head)', 'for (int head = 0; head < gpuHeads; ++head)')
s=s.replace('kernels_.softmax(cb, scores_, o_.heads * o_.M, o_.M);','kernels_.softmax(cb, scores_, gpuHeads * o_.M, o_.M);')
s=s.replace('        outOp_.encode(cb, attentionContext_,', '''        if(splitHeads) {
            [cb commit];headGate.store(true,std::memory_order_release);
            [cb waitUntilCompleted];checkCommandBuffer(cb);headWorker_->wait();
            const size_t suffixWidth=size_t(o_.headSuffix)*headDim_;
            for(int i=0;i<o_.M;++i) std::memcpy((__fp16*)attentionContext_.contents+size_t(i)*o_.H+gpuHeads*headDim_,(__fp16*)headOutput_.contents+size_t(i)*suffixWidth,suffixWidth*2);
            cb=[queue_ commandBuffer];
        }
        outOp_.encode(cb, attentionContext_,''',1)
s=s.replace('    int headDim_, kvWidth_;','''    int headDim_, kvWidth_;
    bool heterogeneousActive_ = false;
    id<MTLBuffer> headOutput_;
    std::unique_ptr<HeadBranch> headBranch_;
    std::unique_ptr<PersistentWorker<HeadBranch>> headWorker_;''')
(p/'src/runner_heads.mm').write_text(s)
