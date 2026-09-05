#pragma once
#include "runtime.hpp"
#import <CoreML/CoreML.h>
namespace tc {
// Public Core ML only. The framework may execute portions on CPU.
class CoreMLBranch {
    MLModel *model_;
    MLMultiArray *output_;
    MLPredictionOptions *options_;
    Tensor output_storage_;
    int rows_;
public:
    double seconds=0;
    uint64_t calls=0, copied_bytes=0;
    CoreMLBranch(const std::filesystem::path&,int rows);
    Tensor predict(const Tensor& packed_input,int actual_rows);
};
class HybridSession {
    std::vector<std::unique_ptr<CoreMLBranch>> branches_;
public:
    std::string manifest;
    int rows=0;
    double load_seconds=0;
    HybridSession(const std::filesystem::path&,const std::filesystem::path& model,int tokens,const Event&,std::atomic<bool>&,int warmups=0);
    Tensor predict(int block,const Tensor&input);
    NSDictionary *metrics()const;
};
}
