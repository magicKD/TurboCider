"""Apply auditable research-only extensions to the vendored mac_local_ai runner."""
from pathlib import Path
p=Path(__file__).resolve().parents[1]
s=(p/'src/baseline.mm').read_text()
s=s.replace('std::string qkvGpuWeights;', 'std::string qkvGpuWeights;\n    std::string qkvCpuWeights;\n    std::string upCpuWeights;')
s=s.replace('const std::string &gpuWeightsPath = std::string())','const std::string &gpuWeightsPath = std::string(),\n                     const std::string &cpuWeightsPath = std::string())')
s=s.replace('        std::vector<__fp16> aneWeights;', '''        if (cpuN_ > 0 && !cpuWeightsPath.empty()) {
            const auto weights = loadWeights(cpuWeightsPath, static_cast<size_t>(K_) * cpuN_);
            std::memcpy(cpuW_.contents, weights.data(), weights.size() * sizeof(__fp16));
        }
        std::vector<__fp16> aneWeights;''')
s=s.replace('nullptr, 0, 0, o.qkvGpuWeights);','nullptr, 0, 0, o.qkvGpuWeights, o.qkvCpuWeights);')
s=s.replace('            o.upGpuWeights);','            o.upGpuWeights, o.upCpuWeights);')
a=s.index('            current.qkvBias = path(')
b=s.index('            blocks_.push_back(',a)
s=s[:a]+'''            current.qkvBias = path("qkv.bias.fp16");
            current.qkvModel = path("qkv.mlmodelc");
            current.qkvWeights = path("qkv_ane.weights.fp16");
            current.qkvGpuWeights = path("qkv_gpu.weights.fp16");
            current.qkvCpuWeights = path("qkv_cpu.weights.fp16");
            current.upModel = path("up.mlmodelc");
            current.upWeights = path("upgate_ane.weights.fp16");
            current.upGpuWeights = path("upgate_gpu.weights.fp16");
            current.upCpuWeights = path("upgate_cpu.weights.fp16");
            current.rowFFNModel = path("row.mlmodelc");
            current.downWeights = path("down.weights.fp16");
            current.outWeights = path("out.weights.fp16");
            current.norm1Weights = path("norm1.weights.fp16");
            current.norm2Weights = path("norm2.weights.fp16");
'''+s[b:]
# Per iteration paired wall-time evidence, independent of aggregate percentiles.
s=s.replace('    const double gpuP50 = percentile(gpuWall, 0.5);','''    std::cout << "{\\"record_type\\":\\"samples\\",\\"gpu_ms\\":[";
    for (size_t i=0;i<gpuWall.size();++i) std::cout << (i ? "," : "") << gpuWall[i];
    std::cout << "],\\"hetero_ms\\":[";
    for (size_t i=0;i<heteroWall.size();++i) std::cout << (i ? "," : "") << heteroWall[i];
    std::cout << "]}\\n";
    const double gpuP50 = percentile(gpuWall, 0.5);''')
s=s.replace('        MLMultiArray *aneOutput = ane_->output();', '        MLMultiArray *aneOutput = ane_ ? ane_->output() : nil;')
s=s.replace('const bool backing = ane_->usedOutputBacking();', 'const bool backing = !ane_ || ane_->usedOutputBacking();')
s=s.replace('        MPSMatrixDescriptor *ld =', '        if (rows_ == 0 || columns_ == 0 || inner_ == 0) return;\n        MPSMatrixDescriptor *ld =')
(p/'src/runner.mm').write_text(s)
