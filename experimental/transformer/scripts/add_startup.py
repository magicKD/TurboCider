from pathlib import Path
p=Path(__file__).resolve().parents[1];s=(p/'src/runner_private_gpu.mm').read_text()
s=s.replace('        model_ = [MLModel modelWithContentsOfURL:compiledModelURL(modelPath)','        const auto modelLoadStart=Clock::now();\n        model_ = [MLModel modelWithContentsOfURL:compiledModelURL(modelPath)',1)
s=s.replace('        if (!model_) throw std::runtime_error(error.localizedDescription.UTF8String);','        std::cerr << "public_model_load_ms=" << milliseconds(Clock::now()-modelLoadStart) << "\\n";\n        if (!model_) throw std::runtime_error(error.localizedDescription.UTF8String);',1)
s=s.replace('        if(private_)return private_->run();','        if(private_){double elapsed=private_->run();if(firstPrediction_){std::cerr << "private_first_prediction_with_layout_ms=" << elapsed << "\\n";firstPrediction_=false;}return elapsed;}',1)
s=s.replace('            return milliseconds(end - start);','            if(firstPrediction_){std::cerr << "public_first_prediction_ms=" << milliseconds(end-start) << "\\n";firstPrediction_=false;}\n            return milliseconds(end - start);',1)
s=s.replace('    std::unique_ptr<PrivateFFN> private_;','    bool firstPrediction_=true;\n    std::unique_ptr<PrivateFFN> private_;',1)
(p/'src/runner_startup.mm').write_text(s)
