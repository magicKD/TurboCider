#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static double milliseconds(Clock::duration value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

static double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = q * (values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(index));
    const size_t hi = static_cast<size_t>(std::ceil(index));
    return values[lo] + (values[hi] - values[lo]) * (index - lo);
}

struct Options {
    int M = 256;
    int H = 4096;
    int F = 14336;
    int heads = 32;
    int kvHeads = 32;
    int layers = 1;
    int warmup = 5;
    int iterations = 30;
    std::string mode = "both";
    std::string gpuStorage = "private";
    std::string ffnLayout = "materialized";
    std::string coremlOutput = "copy";
    std::string attention = "materialized";
    std::string steelVariant = "bq32_bk32_wm4";
    std::string qkvModel =
        "models/coreml/gemm_matmul_m256_k4096_n4096_fp16_s002.mlpackage";
    std::string qkvWeights =
        "models/coreml/gemm_matmul_m256_k4096_n4096_fp16_s002.weights.fp16";
    std::string upModel =
        "models/coreml/gemm_matmul_m256_k4096_n12288_fp16_s002.mlpackage";
    std::string upWeights =
        "models/coreml/gemm_matmul_m256_k4096_n12288_fp16_s002.weights.fp16";
    std::string downWeights;
    std::string inputData;
    std::string qkvGpuWeights;
    std::string qkvBias;
    std::string outWeights;
    std::string upGpuWeights;
    std::string norm1Weights;
    std::string norm2Weights;
    std::string referenceOutput;
    std::string referenceLogits;
    std::string rowFFNModel;
    std::string stackRoot;
    std::string finalNormWeights;
    std::string lmHeadWeights;
    float ropeTheta = 0.0f;
    int vocab = 0;
    int rowGpuM = 0;
    int qkvGpuN = 7168;
    int qkvCpuN = 1024;
    int qkvAneN = 4096;
    int upGpuN = 14336;
    int upCpuN = 2048;
    int upAneN = 12288;
};

static Options parseOptions(int argc, const char *argv[]) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() {
            if (++i >= argc) throw std::runtime_error("missing argument value");
            return argv[i];
        };
        if (a == "--mode") o.mode = value();
        else if (a == "--m") o.M = std::stoi(value());
        else if (a == "--h") o.H = std::stoi(value());
        else if (a == "--f") o.F = std::stoi(value());
        else if (a == "--heads") o.heads = std::stoi(value());
        else if (a == "--kv-heads") o.kvHeads = std::stoi(value());
        else if (a == "--layers") o.layers = std::stoi(value());
        else if (a == "--gpu-storage") o.gpuStorage = value();
        else if (a == "--ffn-layout") o.ffnLayout = value();
        else if (a == "--coreml-output") o.coremlOutput = value();
        else if (a == "--attention") o.attention = value();
        else if (a == "--steel-variant") o.steelVariant = value();
        else if (a == "--warmup") o.warmup = std::stoi(value());
        else if (a == "--iterations") o.iterations = std::stoi(value());
        else if (a == "--qkv-model") o.qkvModel = value();
        else if (a == "--qkv-weights") o.qkvWeights = value();
        else if (a == "--up-model") o.upModel = value();
        else if (a == "--up-weights") o.upWeights = value();
        else if (a == "--down-weights") o.downWeights = value();
        else if (a == "--input-data") o.inputData = value();
        else if (a == "--qkv-gpu-weights") o.qkvGpuWeights = value();
        else if (a == "--qkv-bias") o.qkvBias = value();
        else if (a == "--out-weights") o.outWeights = value();
        else if (a == "--up-gpu-weights") o.upGpuWeights = value();
        else if (a == "--norm1-weights") o.norm1Weights = value();
        else if (a == "--norm2-weights") o.norm2Weights = value();
        else if (a == "--reference-output") o.referenceOutput = value();
        else if (a == "--reference-logits") o.referenceLogits = value();
        else if (a == "--row-ffn-model") o.rowFFNModel = value();
        else if (a == "--stack-root") o.stackRoot = value();
        else if (a == "--final-norm-weights") o.finalNormWeights = value();
        else if (a == "--lm-head-weights") o.lmHeadWeights = value();
        else if (a == "--rope-theta") o.ropeTheta = std::stof(value());
        else if (a == "--vocab") o.vocab = std::stoi(value());
        else if (a == "--row-gpu-m") o.rowGpuM = std::stoi(value());
        else if (a == "--qkv-gpu-n") o.qkvGpuN = std::stoi(value());
        else if (a == "--qkv-cpu-n") o.qkvCpuN = std::stoi(value());
        else if (a == "--qkv-ane-n") o.qkvAneN = std::stoi(value());
        else if (a == "--up-gpu-n") o.upGpuN = std::stoi(value());
        else if (a == "--up-cpu-n") o.upCpuN = std::stoi(value());
        else if (a == "--up-ane-n") o.upAneN = std::stoi(value());
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (o.mode != "gpu" && o.mode != "hetero" && o.mode != "both" &&
        o.mode != "ffn-ablation") {
        throw std::runtime_error(
            "--mode must be gpu, hetero, both, or ffn-ablation");
    }
    if (o.gpuStorage != "shared" && o.gpuStorage != "private") {
        throw std::runtime_error("--gpu-storage must be shared or private");
    }
    if (o.ffnLayout != "materialized" && o.ffnLayout != "segmented" &&
        o.ffnLayout != "segmented-reduce" &&
        o.ffnLayout != "segmented-pipeline" &&
        o.ffnLayout != "activated-pipeline" &&
        o.ffnLayout != "graph-pipeline" &&
        o.ffnLayout != "down-graph-pipeline" &&
        o.ffnLayout != "supergraph" &&
        o.ffnLayout != "row-supergraph") {
        throw std::runtime_error(
            "--ffn-layout must be materialized, segmented, or "
            "segmented-reduce, segmented-pipeline, activated-pipeline, "
            "graph-pipeline, down-graph-pipeline, supergraph, "
            "or row-supergraph");
    }
    if (o.coremlOutput != "copy" && o.coremlOutput != "backing") {
        throw std::runtime_error(
            "--coreml-output must be copy or backing");
    }
    if (o.attention != "materialized" && o.attention != "sdpa" &&
        o.attention != "steel") {
        throw std::runtime_error(
            "--attention must be materialized, sdpa, or steel");
    }
    if (o.H % o.heads != 0 || o.heads % o.kvHeads != 0) {
        throw std::runtime_error(
            "H must divide into Q heads and Q heads into KV heads");
    }
    if (o.layers <= 0) {
        throw std::runtime_error("--layers must be positive");
    }
    const bool hasTail =
        !o.finalNormWeights.empty() || !o.lmHeadWeights.empty() ||
        !o.referenceLogits.empty() || o.vocab > 0;
    if (hasTail &&
        (o.finalNormWeights.empty() || o.lmHeadWeights.empty() ||
         o.vocab <= 0 || o.stackRoot.empty())) {
        throw std::runtime_error(
            "model tail requires --stack-root, --final-norm-weights, "
            "--lm-head-weights, and positive --vocab");
    }
    const int kvWidth = o.kvHeads * (o.H / o.heads);
    if (o.qkvGpuN + o.qkvCpuN + o.qkvAneN != o.H + 2 * kvWidth) {
        throw std::runtime_error(
            "QKV split must sum to H + 2 * KV width");
    }
    if (o.attention != "materialized" &&
        !((o.qkvGpuN == o.H && o.qkvCpuN == 0 &&
           o.qkvAneN == 2 * kvWidth) ||
          (o.qkvGpuN == o.H + 2 * kvWidth && o.qkvCpuN == 0 &&
           o.qkvAneN == 0))) {
        throw std::runtime_error(
            "fused attention requires either semantic GPU+ANE QKV "
            "(GPU=H,CPU=0,ANE=2*KV) or all-GPU QKV "
            "(GPU=H+2*KV,CPU=0,ANE=0)");
    }
    if (o.upGpuN + o.upCpuN + o.upAneN != 2 * o.F) {
        throw std::runtime_error("up/gate split must sum to 2F");
    }
    if ((o.upGpuN & 1) || (o.upCpuN & 1) || (o.upAneN & 1)) {
        throw std::runtime_error(
            "paired up/gate device shards must each have even width");
    }
    if (o.ffnLayout == "supergraph" && o.upCpuN != 0) {
        throw std::runtime_error(
            "supergraph currently requires CPU=0 so the GPU continuation "
            "can run without a host coherence barrier");
    }
    if ((o.ffnLayout == "segmented-pipeline" ||
         o.ffnLayout == "activated-pipeline" ||
         o.ffnLayout == "graph-pipeline" ||
         o.ffnLayout == "down-graph-pipeline") && o.upCpuN != 0) {
        throw std::runtime_error(
            "pipelined segmented FFN currently requires CPU=0");
    }
    if (o.ffnLayout == "row-supergraph" &&
        (o.rowGpuM <= 0 || o.rowGpuM >= o.M ||
         o.rowFFNModel.empty() || o.upGpuN != 2 * o.F ||
         o.upCpuN != 0 || o.upAneN != 0)) {
        throw std::runtime_error(
            "row-supergraph requires 0<row-gpu-m<M, a row-ffn-model, "
            "and an all-GPU up/gate weight plan");
    }
    return o;
}

static void checkCommandBuffer(id<MTLCommandBuffer> cb) {
    if (cb.status == MTLCommandBufferStatusError) {
        throw std::runtime_error(cb.error.localizedDescription.UTF8String);
    }
}

static id<MTLBuffer> makeBuffer(id<MTLDevice> device, size_t elements) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:std::max<size_t>(elements, 1) * sizeof(__fp16)
                            options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
    return buffer;
}

static id<MTLBuffer> uploadPrivate(id<MTLDevice> device,
                                   id<MTLCommandQueue> queue,
                                   id<MTLBuffer> shared) {
    id<MTLBuffer> result =
        [device newBufferWithLength:shared.length
                            options:MTLResourceStorageModePrivate];
    if (!result) throw std::runtime_error("private Metal allocation failed");
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:shared sourceOffset:0 toBuffer:result
       destinationOffset:0 size:shared.length];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCommandBuffer(cb);
    return result;
}

static void fillFP16(__fp16 *data, size_t count, uint32_t &state,
                     float scale = 1.0f) {
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit =
            (static_cast<int>((state >> 8) & 0xffffu) - 32768) / 32768.0f;
        data[i] = static_cast<__fp16>(unit * scale);
    }
}

static std::vector<__fp16> loadWeights(const std::string &path, size_t count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open weights: " + path);
    std::vector<__fp16> weights(count);
    stream.read(reinterpret_cast<char *>(weights.data()),
                static_cast<std::streamsize>(count * sizeof(__fp16)));
    if (stream.gcount() != static_cast<std::streamsize>(count * sizeof(__fp16))) {
        throw std::runtime_error("weight file has unexpected size: " + path);
    }
    return weights;
}

class MPSGemm {
  public:
    MPSGemm(id<MTLDevice> device, int rows, int columns, int inner,
            bool transposeRight = false, float alpha = 1.0f)
        : rows_(rows), columns_(columns), inner_(inner) {
        op_ = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
             transposeLeft:NO
            transposeRight:transposeRight
                resultRows:rows
             resultColumns:columns
            interiorColumns:inner
                      alpha:alpha
                       beta:0.0];
    }

    void encode(id<MTLCommandBuffer> cb,
                id<MTLBuffer> left, size_t leftOffset, int leftRowBytes,
                id<MTLBuffer> right, size_t rightOffset, int rightRows,
                int rightColumns, int rightRowBytes,
                id<MTLBuffer> result, size_t resultOffset, int resultRowBytes) {
        MPSMatrixDescriptor *ld = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rows_ columns:inner_
                            rowBytes:leftRowBytes
                            dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *rd = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rightRows columns:rightColumns
                            rowBytes:rightRowBytes
                            dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *yd = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rows_ columns:columns_
                            rowBytes:resultRowBytes
                            dataType:MPSDataTypeFloat16];
        MPSMatrix *lm = [[MPSMatrix alloc] initWithBuffer:left
                                                   offset:leftOffset
                                               descriptor:ld];
        MPSMatrix *rm = [[MPSMatrix alloc] initWithBuffer:right
                                                   offset:rightOffset
                                               descriptor:rd];
        MPSMatrix *ym = [[MPSMatrix alloc] initWithBuffer:result
                                                   offset:resultOffset
                                               descriptor:yd];
        [op_ encodeToCommandBuffer:cb leftMatrix:lm rightMatrix:rm resultMatrix:ym];
    }

  private:
    int rows_, columns_, inner_;
    MPSMatrixMultiplication *op_;
};

class MPSGraphAttention {
  public:
    MPSGraphAttention(id<MTLBuffer> qBuffer, id<MTLBuffer> kvBuffer,
                      id<MTLBuffer> outputBuffer, int sequence, int heads,
                      int kvHeads, int headDim)
        : sequence_(sequence), heads_(heads), kvHeads_(kvHeads),
          headDim_(headDim) {
        const int qPerKV = heads_ / kvHeads_;
        graph_ = [MPSGraph new];
        MPSShape *qShape =
            @[@(sequence_), @(kvHeads_), @(qPerKV), @(headDim_)];
        MPSShape *kvShape =
            @[@(sequence_), @2, @(kvHeads_), @(headDim_)];
        q_ = [graph_ placeholderWithShape:qShape
                                 dataType:MPSDataTypeFloat16
                                     name:@"q"];
        kv_ = [graph_ placeholderWithShape:kvShape
                                  dataType:MPSDataTypeFloat16
                                      name:@"kv"];

        MPSGraphTensor *queries = [graph_
            transposeTensor:q_
                 permutation:@[@1, @2, @0, @3]
                        name:@"q_heads_first"];
        MPSGraphTensor *keys = [graph_
            sliceTensor:kv_ dimension:1 start:0 length:1 name:@"keys"];
        keys = [graph_
            transposeTensor:keys
                 permutation:@[@2, @1, @0, @3]
                        name:@"k_heads_first"];
        MPSGraphTensor *values = [graph_
            sliceTensor:kv_ dimension:1 start:1 length:1 name:@"values"];
        values = [graph_
            transposeTensor:values
                 permutation:@[@2, @1, @0, @3]
                        name:@"v_heads_first"];

        std::vector<__fp16> mask(
            static_cast<size_t>(sequence_) * sequence_);
        for (int row = 0; row < sequence_; ++row) {
            for (int column = row + 1; column < sequence_; ++column) {
                mask[static_cast<size_t>(row) * sequence_ + column] =
                    static_cast<__fp16>(-INFINITY);
            }
        }
        NSData *maskData = [NSData
            dataWithBytes:mask.data()
                   length:mask.size() * sizeof(__fp16)];
        MPSGraphTensor *maskTensor = [graph_
            constantWithData:maskData
                       shape:@[@1, @1, @(sequence_), @(sequence_)]
                    dataType:MPSDataTypeFloat16];
        MPSGraphTensor *attention = [graph_
            scaledDotProductAttentionWithQueryTensor:queries
                                           keyTensor:keys
                                         valueTensor:values
                                          maskTensor:maskTensor
                                               scale:1.0f /
                                                     std::sqrt(
                                                         float(headDim_))
                                                name:@"fused_sdpa"];
        attention = [graph_
            transposeTensor:attention
                 permutation:@[@2, @0, @1, @3]
                        name:@"tokens_first"];
        output_ = [graph_
            reshapeTensor:attention
                withShape:@[@(sequence_), @(heads_ * headDim_)]
                     name:@"attention_output"];

        qData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:qBuffer
                        shape:qShape
                     dataType:MPSDataTypeFloat16];
        kvData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:kvBuffer
                        shape:kvShape
                     dataType:MPSDataTypeFloat16];
        outputData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outputBuffer
                        shape:@[@(sequence_), @(heads_ * headDim_)]
                     dataType:MPSDataTypeFloat16];
    }

    id<MTLCommandBuffer> encode(id<MTLCommandBuffer> commandBuffer) {
        MPSCommandBuffer *mpsCommandBuffer =
            [MPSCommandBuffer commandBufferWithCommandBuffer:commandBuffer];
        [graph_ encodeToCommandBuffer:mpsCommandBuffer
                                feeds:@{q_: qData_, kv_: kvData_}
                     targetOperations:nil
                    resultsDictionary:@{output_: outputData_}
                  executionDescriptor:nil];
        return mpsCommandBuffer.rootCommandBuffer;
    }

  private:
    int sequence_, heads_, kvHeads_, headDim_;
    MPSGraph *graph_ = nil;
    MPSGraphTensor *q_ = nil;
    MPSGraphTensor *kv_ = nil;
    MPSGraphTensor *output_ = nil;
    MPSGraphTensorData *qData_ = nil;
    MPSGraphTensorData *kvData_ = nil;
    MPSGraphTensorData *outputData_ = nil;
};

class MPSGraphFFNJoin {
  public:
    MPSGraphFFNJoin(
        id<MTLBuffer> upGateBuffer, id<MTLBuffer> downWeightBuffer,
        id<MTLBuffer> residualBuffer, id<MTLBuffer> prefixBuffer,
        id<MTLBuffer> outputBuffer, int rows, int width, int hidden,
        bool preactivated = false)
        : rows_(rows), width_(width), hidden_(hidden) {
        graph_ = [MPSGraph new];
        MPSShape *upGateShape =
            @[@(rows_), @(preactivated ? width_ : 2 * width_)];
        MPSShape *downShape = @[@(width_), @(hidden_)];
        MPSShape *outputShape = @[@(rows_), @(hidden_)];
        upGate_ = [graph_ placeholderWithShape:upGateShape
                                      dataType:MPSDataTypeFloat16
                                          name:@"up_gate"];
        downWeight_ = [graph_ placeholderWithShape:downShape
                                          dataType:MPSDataTypeFloat16
                                              name:@"down_weight"];
        residual_ = [graph_ placeholderWithShape:outputShape
                                        dataType:MPSDataTypeFloat16
                                            name:@"residual"];
        prefix_ = [graph_ placeholderWithShape:outputShape
                                      dataType:MPSDataTypeFloat16
                                          name:@"prefix"];
        MPSGraphTensor *activated = upGate_;
        if (!preactivated) {
            MPSGraphTensor *up = [graph_
                sliceTensor:upGate_ dimension:1 start:0 length:width_
                       name:@"up"];
            MPSGraphTensor *gate = [graph_
                sliceTensor:upGate_ dimension:1 start:width_ length:width_
                       name:@"gate"];
            MPSGraphTensor *silu = [graph_
                multiplicationWithPrimaryTensor:gate
                                secondaryTensor:[graph_
                                    sigmoidWithTensor:gate name:@"sigmoid"]
                                           name:@"silu"];
            activated = [graph_
                multiplicationWithPrimaryTensor:up
                                secondaryTensor:silu
                                           name:@"activated"];
        }
        MPSGraphTensor *down = [graph_
            matrixMultiplicationWithPrimaryTensor:activated
                                  secondaryTensor:downWeight_
                                             name:@"down"];
        MPSGraphTensor *sum = [graph_
            additionWithPrimaryTensor:residual_
                      secondaryTensor:prefix_
                                 name:@"residual_plus_prefix"];
        output_ = [graph_
            additionWithPrimaryTensor:sum
                      secondaryTensor:down
                                 name:@"output"];

        upGateData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:upGateBuffer
                        shape:upGateShape
                     dataType:MPSDataTypeFloat16];
        downWeightData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:downWeightBuffer
                        shape:downShape
                     dataType:MPSDataTypeFloat16];
        residualData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:residualBuffer
                        shape:outputShape
                     dataType:MPSDataTypeFloat16];
        prefixData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:prefixBuffer
                        shape:outputShape
                     dataType:MPSDataTypeFloat16];
        outputData_ = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outputBuffer
                        shape:outputShape
                     dataType:MPSDataTypeFloat16];
    }

    id<MTLCommandBuffer> encode(id<MTLCommandBuffer> commandBuffer) {
        MPSCommandBuffer *mpsCommandBuffer =
            [MPSCommandBuffer commandBufferWithCommandBuffer:commandBuffer];
        [graph_
            encodeToCommandBuffer:mpsCommandBuffer
                            feeds:@{
                                upGate_: upGateData_,
                                downWeight_: downWeightData_,
                                residual_: residualData_,
                                prefix_: prefixData_
                            }
                 targetOperations:nil
                resultsDictionary:@{output_: outputData_}
              executionDescriptor:nil];
        return mpsCommandBuffer.rootCommandBuffer;
    }

  private:
    int rows_, width_, hidden_;
    MPSGraph *graph_ = nil;
    MPSGraphTensor *upGate_ = nil;
    MPSGraphTensor *downWeight_ = nil;
    MPSGraphTensor *residual_ = nil;
    MPSGraphTensor *prefix_ = nil;
    MPSGraphTensor *output_ = nil;
    MPSGraphTensorData *upGateData_ = nil;
    MPSGraphTensorData *downWeightData_ = nil;
    MPSGraphTensorData *residualData_ = nil;
    MPSGraphTensorData *prefixData_ = nil;
    MPSGraphTensorData *outputData_ = nil;
};

struct SteelAttentionParams {
    int B;
    int H;
    int D;
    int qL;
    int kL;
    int gqaFactor;
    float scale;
    int NQ;
    int NK;
    int NQAligned;
    int NKAligned;
    int qLRemainder;
    int kLRemainder;
    int qLOffset;
    int64_t qStrides[3];
    int64_t kStrides[3];
    int64_t vStrides[3];
    int64_t oStrides[3];
};

class SteelAttention {
  public:
    SteelAttention(id<MTLDevice> device, int sequence, int heads,
                   int kvHeads, int headDim,
                   const std::string &variant)
        : sequence_(sequence), heads_(heads), kvHeads_(kvHeads),
          headDim_(headDim) {
        std::string functionName;
        if (variant == "bq32_bk32_wm4") {
            qBlock_ = 32;
            kBlock_ = 32;
            warpM_ = 4;
            functionName =
                "hetero_steel_attention_f16_bq32_bk32_bd64_wm4_wn1";
        } else if (variant == "bq64_bk32_wm8") {
            qBlock_ = 64;
            kBlock_ = 32;
            warpM_ = 8;
            functionName =
                "hetero_steel_attention_f16_bq64_bk32_bd64_wm8_wn1";
        } else if (variant == "bq32_bk16_wm4") {
            qBlock_ = 32;
            kBlock_ = 16;
            warpM_ = 4;
            functionName =
                "hetero_steel_attention_f16_bq32_bk16_bd64_wm4_wn1";
        } else {
            throw std::runtime_error(
                "unknown --steel-variant: " + variant);
        }
        NSError *error = nil;
        NSURL *url = [NSURL
            fileURLWithPath:@"build/steel_qwen_attention.metallib"];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            throw std::runtime_error(error.localizedDescription.UTF8String);
        }
        bool alignedQ = sequence_ % qBlock_ == 0;
        bool alignedK = sequence_ % kBlock_ == 0;
        bool hasMask = false;
        bool causal = true;
        bool hasSinks = false;
        MTLFunctionConstantValues *constants =
            [MTLFunctionConstantValues new];
        [constants setConstantValue:&alignedQ
                               type:MTLDataTypeBool atIndex:200];
        [constants setConstantValue:&alignedK
                               type:MTLDataTypeBool atIndex:201];
        [constants setConstantValue:&hasMask
                               type:MTLDataTypeBool atIndex:300];
        [constants setConstantValue:&causal
                               type:MTLDataTypeBool atIndex:301];
        [constants setConstantValue:&hasSinks
                               type:MTLDataTypeBool atIndex:302];
        NSString *metalFunctionName =
            [NSString stringWithUTF8String:functionName.c_str()];
        id<MTLFunction> function = [library
            newFunctionWithName:metalFunctionName
             constantValues:constants
                      error:&error];
        if (!function) {
            throw std::runtime_error(error.localizedDescription.UTF8String);
        }
        pipeline_ = [device newComputePipelineStateWithFunction:function
                                                          error:&error];
        if (!pipeline_) {
            throw std::runtime_error(error.localizedDescription.UTF8String);
        }
    }

    void encode(id<MTLCommandBuffer> cb, id<MTLBuffer> queries,
                id<MTLBuffer> keysValues, id<MTLBuffer> output) {
        const int qBlocks = (sequence_ + qBlock_ - 1) / qBlock_;
        const int kBlocks = (sequence_ + kBlock_ - 1) / kBlock_;
        const int kvWidth = kvHeads_ * headDim_;
        SteelAttentionParams params{
            1,
            heads_,
            headDim_,
            sequence_,
            sequence_,
            heads_ / kvHeads_,
            1.0f / std::sqrt(float(headDim_)),
            qBlocks,
            kBlocks,
            sequence_ / qBlock_,
            sequence_ / kBlock_,
            sequence_ - (sequence_ / qBlock_) * qBlock_,
            sequence_ - (sequence_ / kBlock_) * kBlock_,
            0,
            {int64_t(sequence_) * heads_ * headDim_,
             int64_t(headDim_), int64_t(heads_) * headDim_},
            {int64_t(sequence_) * 2 * kvWidth,
             int64_t(headDim_), int64_t(2 * kvWidth)},
            {int64_t(sequence_) * 2 * kvWidth,
             int64_t(headDim_), int64_t(2 * kvWidth)},
            {int64_t(sequence_) * heads_ * headDim_,
             int64_t(headDim_), int64_t(heads_) * headDim_}};
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:queries offset:0 atIndex:0];
        [encoder setBuffer:keysValues offset:0 atIndex:1];
        [encoder setBuffer:keysValues
                    offset:static_cast<size_t>(kvWidth) * sizeof(__fp16)
                   atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(qBlocks, heads_, 1)
                threadsPerThreadgroup:MTLSizeMake(32, warpM_, 1)];
        [encoder endEncoding];
    }

  private:
    int sequence_, heads_, kvHeads_, headDim_;
    int qBlock_ = 32;
    int kBlock_ = 32;
    int warpM_ = 4;
    id<MTLComputePipelineState> pipeline_ = nil;
};

static BNNSNDArrayDescriptor bnnsMatrix(void *data, int rows, int columns) {
    BNNSNDArrayDescriptor descriptor{};
    descriptor.layout = BNNSDataLayoutRowMajorMatrix;
    descriptor.size[0] = static_cast<size_t>(columns);
    descriptor.size[1] = static_cast<size_t>(rows);
    descriptor.stride[0] = 1;
    descriptor.stride[1] = static_cast<size_t>(columns);
    descriptor.data = data;
    descriptor.data_type = BNNSDataTypeFloat16;
    descriptor.data_scale = 1.0f;
    return descriptor;
}

class BNNSGemm {
  public:
    BNNSGemm(int M, int K, int N, void *x, void *w, void *y)
        : a_(bnnsMatrix(x, M, K)), b_(bnnsMatrix(w, K, N)),
          c_(bnnsMatrix(y, M, N)) {
        const ssize_t bytes =
            BNNSMatMulWorkspaceSize(false, false, 1.0f, &a_, &b_, &c_, nullptr);
        if (bytes < 0) throw std::runtime_error("BNNS descriptor rejected");
        workspace_.resize(static_cast<size_t>(bytes));
    }

    double run() {
        const auto start = Clock::now();
        const int status = BNNSMatMul(false, false, 1.0f, &a_, &b_, &c_,
                                     workspace_.empty() ? nullptr : workspace_.data(),
                                     nullptr);
        if (status != 0) throw std::runtime_error("BNNSMatMul failed");
        return milliseconds(Clock::now() - start);
    }

  private:
    BNNSNDArrayDescriptor a_{}, b_{}, c_{};
    std::vector<uint8_t> workspace_;
};

static NSURL *compiledModelURL(const std::string &path) {
    NSURL *source =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if ([source.pathExtension isEqualToString:@"mlmodelc"]) return source;
    NSError *error = nil;
    NSURL *compiled = [MLModel compileModelAtURL:source error:&error];
    if (!compiled) throw std::runtime_error(error.localizedDescription.UTF8String);
    return compiled;
}

class CoreMLGemm {
  public:
    CoreMLGemm(const std::string &modelPath, int M, int K, int N,
               void *input, void *outputBacking = nullptr,
               void *partialBacking = nullptr, int partialN = 0)
        : M_(M), N_(N), partialN_(partialN) {
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError *error = nil;
        model_ = [MLModel modelWithContentsOfURL:compiledModelURL(modelPath)
                                  configuration:configuration error:&error];
        if (!model_) throw std::runtime_error(error.localizedDescription.UTF8String);
        MLMultiArray *array = [[MLMultiArray alloc]
            initWithDataPointer:input
                         shape:@[@(M), @(K)]
                      dataType:MLMultiArrayDataTypeFloat16
                       strides:@[@(K), @1]
                    deallocator:^(void *) {}
                         error:&error];
        if (!array) throw std::runtime_error(error.localizedDescription.UTF8String);
        provider_ = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"x": [MLFeatureValue featureValueWithMultiArray:array]}
                         error:&error];
        if (!provider_) throw std::runtime_error(error.localizedDescription.UTF8String);
        NSMutableDictionary<NSString *, MLMultiArray *> *backings =
            [NSMutableDictionary dictionary];
        if (outputBacking) {
            outputBacking_ = [[MLMultiArray alloc]
                initWithDataPointer:outputBacking
                             shape:@[@(M), @(N)]
                          dataType:MLMultiArrayDataTypeFloat16
                           strides:@[@(N), @1]
                        deallocator:^(void *) {}
                             error:&error];
            if (!outputBacking_) {
                throw std::runtime_error(error.localizedDescription.UTF8String);
            }
            backings[@"y"] = outputBacking_;
        }
        if (partialBacking) {
            partialBacking_ = [[MLMultiArray alloc]
                initWithDataPointer:partialBacking
                             shape:@[@(M), @(partialN)]
                          dataType:MLMultiArrayDataTypeFloat16
                           strides:@[@(partialN), @1]
                        deallocator:^(void *) {}
                             error:&error];
            if (!partialBacking_) {
                throw std::runtime_error(error.localizedDescription.UTF8String);
            }
            backings[@"partial"] = partialBacking_;
        }
        if (backings.count > 0) {
            options_ = [MLPredictionOptions new];
            options_.outputBackings = backings;
        }
    }

    double run() {
        @autoreleasepool {
            NSError *error = nil;
            const auto start = Clock::now();
            id<MLFeatureProvider> result = options_
                ? [model_ predictionFromFeatures:provider_
                                         options:options_ error:&error]
                : [model_ predictionFromFeatures:provider_ error:&error];
            const auto end = Clock::now();
            if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
            output_ = [[result featureValueForName:@"y"] multiArrayValue];
            if (!output_ || output_.dataType != MLMultiArrayDataTypeFloat16 ||
                output_.shape.count != 2 ||
                output_.shape[0].intValue != M_ ||
                output_.shape[1].intValue != N_) {
                throw std::runtime_error(
                    "Core ML output shape or data type mismatch");
            }
            usedOutputBacking_ =
                outputBacking_ && output_ == outputBacking_ &&
                output_.dataPointer == outputBacking_.dataPointer;
            if (partialN_ > 0) {
                partial_ =
                    [[result featureValueForName:@"partial"] multiArrayValue];
                if (!partial_ ||
                    partial_.dataType != MLMultiArrayDataTypeFloat16 ||
                    partial_.shape.count != 2 ||
                    partial_.shape[0].intValue != M_ ||
                    partial_.shape[1].intValue != partialN_) {
                    throw std::runtime_error(
                        "Core ML partial output shape or data type mismatch");
                }
                usedPartialBacking_ =
                    partialBacking_ && partial_ == partialBacking_ &&
                    partial_.dataPointer == partialBacking_.dataPointer;
            }
            return milliseconds(end - start);
        }
    }

    MLMultiArray *output() const { return output_; }
    bool usedOutputBacking() const { return usedOutputBacking_; }
    bool usedPartialBacking() const { return usedPartialBacking_; }

  private:
    MLModel *model_ = nil;
    MLDictionaryFeatureProvider *provider_ = nil;
    MLPredictionOptions *options_ = nil;
    MLMultiArray *outputBacking_ = nil;
    MLMultiArray *partialBacking_ = nil;
    MLMultiArray *output_ = nil;
    MLMultiArray *partial_ = nil;
    bool usedOutputBacking_ = false;
    bool usedPartialBacking_ = false;
    int M_ = 0;
    int N_ = 0;
    int partialN_ = 0;
};

template <typename Work>
class PersistentWorker {
  public:
    PersistentWorker(Work &work, bool utilityQoS)
        : work_(work), utilityQoS_(utilityQoS), thread_([this] { loop(); }) {}
    ~PersistentWorker() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            cv_.notify_one();
        }
        thread_.join();
    }

    void submit(std::atomic<bool> *gate) {
        std::lock_guard<std::mutex> lock(mu_);
        gate_ = gate;
        pending_ = true;
        done_ = false;
        cv_.notify_one();
    }

    double wait() {
        std::unique_lock<std::mutex> lock(mu_);
        doneCv_.wait(lock, [this] { return done_; });
        return elapsed_;
    }

  private:
    void loop() {
        if (utilityQoS_) pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        for (;;) {
            std::atomic<bool> *gate;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return pending_ || stop_; });
                if (stop_) return;
                pending_ = false;
                gate = gate_;
            }
            while (!gate->load(std::memory_order_acquire)) std::this_thread::yield();
            const double elapsed = work_.run();
            {
                std::lock_guard<std::mutex> lock(mu_);
                elapsed_ = elapsed;
                done_ = true;
                doneCv_.notify_one();
            }
        }
    }

    Work &work_;
    bool utilityQoS_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_, doneCv_;
    std::atomic<bool> *gate_ = nullptr;
    bool pending_ = false, done_ = false, stop_ = false;
    double elapsed_ = 0.0;
};

struct ProjectionTimes {
    double wall = 0.0;
    double gpu = 0.0;
    double cpu = 0.0;
    double ane = 0.0;
    double concat = 0.0;
    double bridge = 0.0;
    bool aneOutputBacking = false;
};

class TripleProjection {
  public:
    TripleProjection(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     int M, int K, int gpuN, int cpuN, int aneN,
                     id<MTLBuffer> input, const std::string &modelPath,
                     const std::string &aneWeightsPath, uint32_t seed,
                     float weightScale, bool privateGPUWeights,
                     bool pairedHalves = false,
                     bool useANEOutputBacking = false,
                     void *anePartialBacking = nullptr,
                     int anePartialN = 0,
                     int aneModelN = 0,
                     const std::string &gpuWeightsPath = std::string())
        : device_(device), queue_(queue), M_(M), K_(K), gpuN_(gpuN),
          cpuN_(cpuN), aneN_(aneN), totalN_(gpuN + cpuN + aneN),
          aneOutputN_(aneModelN > 0 ? aneModelN : aneN),
          pairedHalves_(pairedHalves), input_(input),
          gpuW_(makeBuffer(device, static_cast<size_t>(K) * gpuN)),
          cpuW_(makeBuffer(device, static_cast<size_t>(K) * cpuN)),
          gpuY_(makeBuffer(device, static_cast<size_t>(M) * gpuN)),
          cpuY_(makeBuffer(device, static_cast<size_t>(M) * cpuN)),
          aneY_(makeBuffer(
              device, static_cast<size_t>(M) *
                  (aneModelN > 0 ? aneModelN : aneN))),
          joined_(makeBuffer(device, static_cast<size_t>(M) * totalN_)),
          fullW_(makeBuffer(device, static_cast<size_t>(K) * totalN_)),
          gpuOp_(device, M, gpuN, K, false, 1.0f),
          fullOp_(device, M, totalN_, K, false, 1.0f) {
        uint32_t state = seed;
        if (gpuWeightsPath.empty()) {
            fillFP16(static_cast<__fp16 *>(gpuW_.contents),
                     static_cast<size_t>(K_) * gpuN_, state, weightScale);
        } else {
            const auto gpuWeights = loadWeights(
                gpuWeightsPath, static_cast<size_t>(K_) * gpuN_);
            std::memcpy(
                gpuW_.contents, gpuWeights.data(),
                static_cast<size_t>(K_) * gpuN_ * sizeof(__fp16));
        }
        fillFP16(static_cast<__fp16 *>(cpuW_.contents),
                 static_cast<size_t>(K_) * cpuN_, state, weightScale);
        std::vector<__fp16> aneWeights;
        if (aneN_ > 0) {
            aneWeights =
                loadWeights(aneWeightsPath, static_cast<size_t>(K_) * aneN_);
        }

        const __fp16 *gw = static_cast<const __fp16 *>(gpuW_.contents);
        const __fp16 *cw = static_cast<const __fp16 *>(cpuW_.contents);
        __fp16 *full = static_cast<__fp16 *>(fullW_.contents);
        for (int k = 0; k < K_; ++k) {
            __fp16 *row = full + static_cast<size_t>(k) * totalN_;
            const __fp16 *gpuRow = gw + static_cast<size_t>(k) * gpuN_;
            const __fp16 *cpuRow = cw + static_cast<size_t>(k) * cpuN_;
            const __fp16 *aneRow = aneN_ > 0
                ? aneWeights.data() + static_cast<size_t>(k) * aneN_
                : nullptr;
            if (!pairedHalves_) {
                std::memcpy(row, gpuRow,
                            static_cast<size_t>(gpuN_) * sizeof(__fp16));
                std::memcpy(row + gpuN_, cpuRow,
                            static_cast<size_t>(cpuN_) * sizeof(__fp16));
                if (aneN_ > 0) {
                    std::memcpy(row + gpuN_ + cpuN_, aneRow,
                                static_cast<size_t>(aneN_) *
                                    sizeof(__fp16));
                }
            } else {
                const int gpuHalf = gpuN_ / 2;
                const int cpuHalf = cpuN_ / 2;
                const int aneHalf = aneN_ / 2;
                const int width = totalN_ / 2;
                std::memcpy(row, gpuRow,
                            static_cast<size_t>(gpuHalf) * sizeof(__fp16));
                std::memcpy(row + gpuHalf, cpuRow,
                            static_cast<size_t>(cpuHalf) * sizeof(__fp16));
                if (aneHalf > 0) {
                    std::memcpy(row + gpuHalf + cpuHalf, aneRow,
                                static_cast<size_t>(aneHalf) *
                                    sizeof(__fp16));
                }
                std::memcpy(row + width, gpuRow + gpuHalf,
                            static_cast<size_t>(gpuHalf) * sizeof(__fp16));
                std::memcpy(row + width + gpuHalf, cpuRow + cpuHalf,
                            static_cast<size_t>(cpuHalf) * sizeof(__fp16));
                if (aneHalf > 0) {
                    std::memcpy(row + width + gpuHalf + cpuHalf,
                                aneRow + aneHalf,
                                static_cast<size_t>(aneHalf) *
                                    sizeof(__fp16));
                }
            }
        }
        if (privateGPUWeights) {
            gpuWPrivate_ = uploadPrivate(device_, queue_, gpuW_);
            fullWPrivate_ = uploadPrivate(device_, queue_, fullW_);
        }

        if (aneN_ > 0) {
            ane_ = std::make_unique<CoreMLGemm>(
                modelPath, M_, K_, aneOutputN_, input_.contents,
                useANEOutputBacking ? aneY_.contents : nullptr,
                anePartialBacking, anePartialN);
            aneWorker_ =
                std::make_unique<PersistentWorker<CoreMLGemm>>(*ane_, false);
        }
        if (cpuN_ > 0) {
            cpu_ = std::make_unique<BNNSGemm>(
                M_, K_, cpuN_, input_.contents, cpuW_.contents, cpuY_.contents);
            cpuWorker_ =
                std::make_unique<PersistentWorker<BNNSGemm>>(*cpu_, true);
        }
    }

    ProjectionTimes runHeterogeneous(
        const std::function<void()> &overlapHostWork = {},
        bool materialize = true,
        id<MTLCommandBuffer> gpuContinuation = nil,
        const std::function<void(
            id<MTLCommandBuffer>, id<MTLBuffer>)> &gpuPostprocess = {}) {
        const auto start = Clock::now();
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        gpuOp_.encode(cb, input_, 0, K_ * 2,
                      gpuWeightBuffer(), 0, K_, gpuN_, gpuN_ * 2,
                      gpuY_, 0, gpuN_ * 2);
        if (gpuPostprocess) gpuPostprocess(cb, gpuY_);
        std::atomic<bool> gate{false};
        if (cpuWorker_) cpuWorker_->submit(&gate);
        if (aneWorker_) aneWorker_->submit(&gate);
        [cb commit];
        gate.store(true, std::memory_order_release);
        if (overlapHostWork) overlapHostWork();
        if (gpuContinuation) [gpuContinuation commit];
        [cb waitUntilCompleted];
        const double cpuMs = cpuWorker_ ? cpuWorker_->wait() : 0.0;
        const double aneMs = aneWorker_ ? aneWorker_->wait() : 0.0;
        checkCommandBuffer(cb);
        if (gpuContinuation) {
            [gpuContinuation waitUntilCompleted];
            checkCommandBuffer(gpuContinuation);
        }

        double bridgeMs = 0.0;
        if (!materialize) {
            const auto bridgeStart = Clock::now();
            ensureANEOutputBuffer();
            bridgeMs = milliseconds(Clock::now() - bridgeStart);
        }
        double concatMs = 0.0;
        if (materialize) {
            const auto concatStart = Clock::now();
            concat();
            concatMs = milliseconds(Clock::now() - concatStart);
        }
        return {milliseconds(Clock::now() - start),
                (cb.GPUEndTime - cb.GPUStartTime) * 1000.0,
                cpuMs, aneMs, concatMs, bridgeMs,
                ane_ ? ane_->usedOutputBacking() : false};
    }

    void encodeFullGPU(id<MTLCommandBuffer> cb) {
        fullOp_.encode(cb, input_, 0, K_ * 2,
                       fullWeightBuffer(), 0, K_, totalN_, totalN_ * 2,
                       joined_, 0, totalN_ * 2);
    }

    id<MTLBuffer> output() const { return joined_; }
    id<MTLBuffer> gpuOutput() const { return gpuY_; }
    id<MTLBuffer> cpuOutput() const { return cpuY_; }
    id<MTLBuffer> aneOutput() const { return aneY_; }
    int gpuWidth() const { return gpuN_; }
    int cpuWidth() const { return cpuN_; }
    int aneWidth() const { return aneN_; }
    id<MTLBuffer> fullWeights() const { return fullW_; }
    id<MTLBuffer> fullGPUWeights() const { return fullWeightBuffer(); }

  private:
    id<MTLBuffer> gpuWeightBuffer() const {
        return gpuWPrivate_ ? gpuWPrivate_ : gpuW_;
    }

    id<MTLBuffer> fullWeightBuffer() const {
        return fullWPrivate_ ? fullWPrivate_ : fullW_;
    }

    void concat() {
        const __fp16 *gpu = static_cast<const __fp16 *>(gpuY_.contents);
        const __fp16 *cpu = static_cast<const __fp16 *>(cpuY_.contents);
        MLMultiArray *aneOutput = ane_->output();
        const bool backing = ane_->usedOutputBacking();
        const __fp16 *ane = backing
            ? static_cast<const __fp16 *>(aneY_.contents)
            : static_cast<const __fp16 *>(aneOutput.dataPointer);
        const size_t aneRowStride = backing
            ? static_cast<size_t>(aneN_)
            : aneOutput.strides[0].unsignedLongLongValue;
        const size_t aneColumnStride = backing
            ? size_t(1)
            : aneOutput.strides[1].unsignedLongLongValue;
        __fp16 *destination = static_cast<__fp16 *>(joined_.contents);
        for (int m = 0; m < M_; ++m) {
            __fp16 *row = destination + static_cast<size_t>(m) * totalN_;
            const __fp16 *gpuRow = gpu + static_cast<size_t>(m) * gpuN_;
            const __fp16 *cpuRow = cpu + static_cast<size_t>(m) * cpuN_;
            const __fp16 *aneRow =
                ane + static_cast<size_t>(m) * aneRowStride;
            if (!pairedHalves_) {
                std::memcpy(row, gpuRow,
                            static_cast<size_t>(gpuN_) * sizeof(__fp16));
                std::memcpy(row + gpuN_, cpuRow,
                            static_cast<size_t>(cpuN_) * sizeof(__fp16));
                if (aneColumnStride == 1) {
                    std::memcpy(row + gpuN_ + cpuN_, aneRow,
                                static_cast<size_t>(aneN_) * sizeof(__fp16));
                } else {
                    for (int n = 0; n < aneN_; ++n) {
                        row[gpuN_ + cpuN_ + n] =
                            aneRow[static_cast<size_t>(n) * aneColumnStride];
                    }
                }
            } else {
                const int gpuHalf = gpuN_ / 2;
                const int cpuHalf = cpuN_ / 2;
                const int aneHalf = aneN_ / 2;
                const int width = totalN_ / 2;
                std::memcpy(row, gpuRow,
                            static_cast<size_t>(gpuHalf) * sizeof(__fp16));
                std::memcpy(row + gpuHalf, cpuRow,
                            static_cast<size_t>(cpuHalf) * sizeof(__fp16));
                if (aneColumnStride == 1) {
                    std::memcpy(
                        row + gpuHalf + cpuHalf, aneRow,
                        static_cast<size_t>(aneHalf) * sizeof(__fp16));
                } else {
                    for (int n = 0; n < aneHalf; ++n) {
                        row[gpuHalf + cpuHalf + n] =
                            aneRow[static_cast<size_t>(n) * aneColumnStride];
                    }
                }
                std::memcpy(row + width, gpuRow + gpuHalf,
                            static_cast<size_t>(gpuHalf) * sizeof(__fp16));
                std::memcpy(row + width + gpuHalf, cpuRow + cpuHalf,
                            static_cast<size_t>(cpuHalf) * sizeof(__fp16));
                if (aneColumnStride == 1) {
                    std::memcpy(row + width + gpuHalf + cpuHalf,
                                aneRow + aneHalf,
                                static_cast<size_t>(aneHalf) * sizeof(__fp16));
                } else {
                    for (int n = 0; n < aneHalf; ++n) {
                        row[width + gpuHalf + cpuHalf + n] =
                            aneRow[static_cast<size_t>(aneHalf + n) *
                                   aneColumnStride];
                    }
                }
            }
        }
    }

    void ensureANEOutputBuffer() {
        if (!ane_) return;
        MLMultiArray *aneOutput = ane_->output();
        if (!aneOutput || aneOutput.dataType != MLMultiArrayDataTypeFloat16) {
            throw std::runtime_error("Core ML produced no FP16 output");
        }
        if (ane_->usedOutputBacking()) return;
        const __fp16 *source =
            static_cast<const __fp16 *>(aneOutput.dataPointer);
        const size_t sourceRow =
            aneOutput.strides[0].unsignedLongLongValue;
        const size_t sourceColumn =
            aneOutput.strides[1].unsignedLongLongValue;
        __fp16 *destination = static_cast<__fp16 *>(aneY_.contents);
        for (int m = 0; m < M_; ++m) {
            __fp16 *destinationRow =
                destination + static_cast<size_t>(m) * aneOutputN_;
            const __fp16 *sourcePointer =
                source + static_cast<size_t>(m) * sourceRow;
            if (sourceColumn == 1) {
                std::memcpy(destinationRow, sourcePointer,
                            static_cast<size_t>(aneOutputN_) * sizeof(__fp16));
            } else {
                for (int n = 0; n < aneOutputN_; ++n) {
                    destinationRow[n] =
                        sourcePointer[static_cast<size_t>(n) * sourceColumn];
                }
            }
        }
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    int M_, K_, gpuN_, cpuN_, aneN_, totalN_, aneOutputN_;
    bool pairedHalves_;
    id<MTLBuffer> input_;
    id<MTLBuffer> gpuW_, cpuW_, gpuY_, cpuY_, aneY_, joined_, fullW_;
    id<MTLBuffer> gpuWPrivate_, fullWPrivate_;
    MPSGemm gpuOp_, fullOp_;
    std::unique_ptr<BNNSGemm> cpu_;
    std::unique_ptr<CoreMLGemm> ane_;
    std::unique_ptr<PersistentWorker<BNNSGemm>> cpuWorker_;
    std::unique_ptr<PersistentWorker<CoreMLGemm>> aneWorker_;
};

class Kernels {
  public:
    explicit Kernels(id<MTLDevice> device) {
        static NSString *source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(const device half *input [[buffer(0)]],
                    device half *output [[buffer(1)]],
                    const device half *weight [[buffer(2)]],
                    constant uint &width [[buffer(3)]],
                    constant float &epsilon [[buffer(4)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_threadgroup]],
                    uint lanes [[threads_per_threadgroup]],
                    threadgroup float *scratch [[threadgroup(0)]]) {
    float sum = 0.0f;
    const uint base = row * width;
    for (uint i = lane; i < width; i += lanes) {
        float value = float(input[base + i]);
        sum += value * value;
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = rsqrt(scratch[0] / float(width) + epsilon);
    for (uint i = lane; i < width; i += lanes)
        output[base + i] =
            half(float(input[base + i]) * inverse * float(weight[i]));
}

kernel void causal_softmax(device half *scores [[buffer(0)]],
                           constant uint &sequence [[buffer(1)]],
                           uint rowGlobal [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]],
                           uint lanes [[threads_per_threadgroup]],
                           threadgroup float *scratch [[threadgroup(0)]]) {
    const uint row = rowGlobal % sequence;
    const uint base = rowGlobal * sequence;
    float maximum = -INFINITY;
    for (uint column = lane; column < sequence; column += lanes) {
        if (column <= row) maximum = max(maximum, float(scores[base + column]));
    }
    scratch[lane] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] =
            max(scratch[lane], scratch[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maxValue = scratch[0];
    float sum = 0.0f;
    for (uint column = lane; column < sequence; column += lanes) {
        const float value = column <= row
            ? exp(float(scores[base + column]) - maxValue) : 0.0f;
        scores[base + column] = half(value);
        sum += value;
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse = 1.0f / scratch[0];
    for (uint column = lane; column < sequence; column += lanes)
        scores[base + column] = half(float(scores[base + column]) * inverse);
}

kernel void add_arrays(const device half *a [[buffer(0)]],
                       const device half *b [[buffer(1)]],
                       device half *output [[buffer(2)]],
                       constant uint &count [[buffer(3)]],
                       uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = a[index] + b[index];
}

kernel void add_row_bias(device half *values [[buffer(0)]],
                         const device half *bias [[buffer(1)]],
                         constant uint &rows [[buffer(2)]],
                         constant uint &width [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index < count) {
        const uint column = index % width;
        values[index] = half(float(values[index]) + float(bias[column]));
    }
}

kernel void apply_qwen_rope(
    device half *queries [[buffer(0)]],
    device half *keysValues [[buffer(1)]],
    const device half *cosines [[buffer(2)]],
    const device half *sines [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &queryHeads [[buffer(5)]],
    constant uint &kvHeads [[buffer(6)]],
    constant uint &headDim [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint halfDim = headDim / 2;
    const uint heads = queryHeads + kvHeads;
    const uint count = rows * heads * halfDim;
    if (index >= count) return;
    const uint pair = index % halfDim;
    const uint headAndRow = index / halfDim;
    const uint head = headAndRow % heads;
    const uint row = headAndRow / heads;
    const float c = float(cosines[row * halfDim + pair]);
    const float s = float(sines[row * halfDim + pair]);
    if (head < queryHeads) {
        const uint base =
            row * queryHeads * headDim + head * headDim;
        const float x1 = float(queries[base + pair]);
        const float x2 = float(queries[base + pair + halfDim]);
        queries[base + pair] = half(x1 * c - x2 * s);
        queries[base + pair + halfDim] = half(x1 * s + x2 * c);
    } else {
        const uint kvHead = head - queryHeads;
        const uint kvWidth = kvHeads * headDim;
        const uint base =
            row * 2 * kvWidth + kvHead * headDim;
        const float x1 = float(keysValues[base + pair]);
        const float x2 = float(keysValues[base + pair + halfDim]);
        keysValues[base + pair] = half(x1 * c - x2 * s);
        keysValues[base + pair + halfDim] = half(x1 * s + x2 * c);
    }
}

kernel void split_qkv_semantic(const device half *input [[buffer(0)]],
                               device half *queries [[buffer(1)]],
                               device half *keysValues [[buffer(2)]],
                               constant uint &rows [[buffer(3)]],
                               constant uint &queryWidth [[buffer(4)]],
                               constant uint &kvWidth [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
    const uint rowWidth = queryWidth + kvWidth;
    const uint count = rows * rowWidth;
    if (index >= count) return;
    const uint row = index / rowWidth;
    const uint column = index - row * rowWidth;
    if (column < queryWidth) {
        queries[row * queryWidth + column] = input[index];
    } else {
        keysValues[row * kvWidth + column - queryWidth] = input[index];
    }
}

kernel void add_four_arrays(const device half *a [[buffer(0)]],
                            const device half *b [[buffer(1)]],
                            const device half *c [[buffer(2)]],
                            const device half *d [[buffer(3)]],
                            device half *output [[buffer(4)]],
                            constant uint &count [[buffer(5)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = half(
            float(a[index]) + float(b[index]) +
            float(c[index]) + float(d[index]));
    }
}

kernel void add_three_arrays(const device half *a [[buffer(0)]],
                             const device half *b [[buffer(1)]],
                             const device half *c [[buffer(2)]],
                             device half *output [[buffer(3)]],
                             constant uint &count [[buffer(4)]],
                             uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] =
            half(float(a[index]) + float(b[index]) + float(c[index]));
    }
}

kernel void add_ffn_row_shards(
    const device half *residual [[buffer(0)]],
    const device half *gpuRows [[buffer(1)]],
    const device half *aneRows [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &gpuRowCount [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint column = index - row * width;
    const half partial = row < gpuRowCount
        ? gpuRows[row * width + column]
        : aneRows[(row - gpuRowCount) * width + column];
    output[index] = half(float(residual[index]) + float(partial));
}

kernel void swiglu(const device half *input [[buffer(0)]],
                   device half *output [[buffer(1)]],
                   constant uint &rows [[buffer(2)]],
                   constant uint &width [[buffer(3)]],
                   uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint column = index - row * width;
    const uint base = row * width * 2;
    const float up = float(input[base + column]);
    const float gate = float(input[base + width + column]);
    const float silu = gate / (1.0f + exp(-gate));
    output[index] = half(up * silu);
}

kernel void swiglu_segmented(
    const device half *gpuInput [[buffer(0)]],
    const device half *cpuInput [[buffer(1)]],
    const device half *aneInput [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &gpuWidth [[buffer(5)]],
    constant uint &cpuWidth [[buffer(6)]],
    constant uint &aneWidth [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint width = gpuWidth + cpuWidth + aneWidth;
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint column = index - row * width;
    float up;
    float gate;
    if (column < gpuWidth) {
        const uint local = row * gpuWidth * 2 + column;
        up = float(gpuInput[local]);
        gate = float(gpuInput[local + gpuWidth]);
    } else if (column < gpuWidth + cpuWidth) {
        const uint localColumn = column - gpuWidth;
        const uint local = row * cpuWidth * 2 + localColumn;
        up = float(cpuInput[local]);
        gate = float(cpuInput[local + cpuWidth]);
    } else {
        const uint localColumn = column - gpuWidth - cpuWidth;
        const uint local = row * aneWidth * 2 + localColumn;
        up = float(aneInput[local]);
        gate = float(aneInput[local + aneWidth]);
    }
    const float silu = gate / (1.0f + exp(-gate));
    output[index] = half(up * silu);
}

kernel void swiglu_scatter(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &outputWidth [[buffer(4)]],
    constant uint &outputOffset [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint column = index - row * width;
    const uint inputBase = row * width * 2;
    const float up = float(input[inputBase + column]);
    const float gate = float(input[inputBase + width + column]);
    const float silu = gate / (1.0f + exp(-gate));
    output[row * outputWidth + outputOffset + column] = half(up * silu);
}
)METAL";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                     options:nil error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        rms_ = makePipeline(device, library, @"rmsnorm");
        softmax_ = makePipeline(device, library, @"causal_softmax");
        add_ = makePipeline(device, library, @"add_arrays");
        addBias_ = makePipeline(device, library, @"add_row_bias");
        rope_ = makePipeline(device, library, @"apply_qwen_rope");
        splitQKV_ = makePipeline(device, library, @"split_qkv_semantic");
        addFour_ = makePipeline(device, library, @"add_four_arrays");
        addThree_ = makePipeline(device, library, @"add_three_arrays");
        addRowShards_ =
            makePipeline(device, library, @"add_ffn_row_shards");
        swiglu_ = makePipeline(device, library, @"swiglu");
        swigluSegmented_ =
            makePipeline(device, library, @"swiglu_segmented");
        swigluScatter_ = makePipeline(device, library, @"swiglu_scatter");
    }

    void rmsnorm(id<MTLCommandBuffer> cb, id<MTLBuffer> input,
                 id<MTLBuffer> output, id<MTLBuffer> weight,
                 uint32_t rows, uint32_t width, float epsilon) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:rms_];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBuffer:weight offset:0 atIndex:2];
        [encoder setBytes:&width length:sizeof(width) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        constexpr uint32_t threads = 256;
        [encoder setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }

    void rmsnormRow(id<MTLCommandBuffer> cb, id<MTLBuffer> input,
                    uint32_t inputRow, id<MTLBuffer> output,
                    id<MTLBuffer> weight, uint32_t width,
                    float epsilon) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:rms_];
        [encoder setBuffer:input
                    offset:static_cast<NSUInteger>(inputRow) * width *
                           sizeof(__fp16)
                   atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBuffer:weight offset:0 atIndex:2];
        [encoder setBytes:&width length:sizeof(width) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        constexpr uint32_t threads = 256;
        [encoder setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }

    void softmax(id<MTLCommandBuffer> cb, id<MTLBuffer> scores,
                 uint32_t rows, uint32_t sequence) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:softmax_];
        [encoder setBuffer:scores offset:0 atIndex:0];
        [encoder setBytes:&sequence length:sizeof(sequence) atIndex:1];
        constexpr uint32_t threads = 256;
        [encoder setThreadgroupMemoryLength:threads * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }

    void add(id<MTLCommandBuffer> cb, id<MTLBuffer> a, id<MTLBuffer> b,
             id<MTLBuffer> output, uint32_t count) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:add_];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        dispatchLinear(encoder, add_, count);
        [encoder endEncoding];
    }

    void addBias(id<MTLCommandBuffer> cb, id<MTLBuffer> values,
                 id<MTLBuffer> bias, uint32_t rows, uint32_t width) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:addBias_];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:bias offset:0 atIndex:1];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
        [encoder setBytes:&width length:sizeof(width) atIndex:3];
        dispatchLinear(encoder, addBias_, rows * width);
        [encoder endEncoding];
    }

    void rope(id<MTLCommandBuffer> cb, id<MTLBuffer> queries,
              id<MTLBuffer> keysValues, id<MTLBuffer> cosines,
              id<MTLBuffer> sines, uint32_t rows, uint32_t queryHeads,
              uint32_t kvHeads, uint32_t headDim) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:rope_];
        [encoder setBuffer:queries offset:0 atIndex:0];
        [encoder setBuffer:keysValues offset:0 atIndex:1];
        [encoder setBuffer:cosines offset:0 atIndex:2];
        [encoder setBuffer:sines offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&queryHeads length:sizeof(queryHeads) atIndex:5];
        [encoder setBytes:&kvHeads length:sizeof(kvHeads) atIndex:6];
        [encoder setBytes:&headDim length:sizeof(headDim) atIndex:7];
        dispatchLinear(
            encoder, rope_,
            rows * (queryHeads + kvHeads) * (headDim / 2));
        [encoder endEncoding];
    }

    void splitQKV(id<MTLCommandBuffer> cb, id<MTLBuffer> input,
                  id<MTLBuffer> queries, id<MTLBuffer> keysValues,
                  uint32_t rows, uint32_t queryWidth, uint32_t kvWidth) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:splitQKV_];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:queries offset:0 atIndex:1];
        [encoder setBuffer:keysValues offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&queryWidth length:sizeof(queryWidth) atIndex:4];
        [encoder setBytes:&kvWidth length:sizeof(kvWidth) atIndex:5];
        dispatchLinear(encoder, splitQKV_, rows * (queryWidth + kvWidth));
        [encoder endEncoding];
    }

    void swiglu(id<MTLCommandBuffer> cb, id<MTLBuffer> input,
                id<MTLBuffer> output, uint32_t rows, uint32_t width) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:swiglu_];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
        [encoder setBytes:&width length:sizeof(width) atIndex:3];
        dispatchLinear(encoder, swiglu_, rows * width);
        [encoder endEncoding];
    }

    void addFour(id<MTLCommandBuffer> cb, id<MTLBuffer> a,
                 id<MTLBuffer> b, id<MTLBuffer> c, id<MTLBuffer> d,
                 id<MTLBuffer> output, uint32_t count) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:addFour_];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:c offset:0 atIndex:2];
        [encoder setBuffer:d offset:0 atIndex:3];
        [encoder setBuffer:output offset:0 atIndex:4];
        [encoder setBytes:&count length:sizeof(count) atIndex:5];
        dispatchLinear(encoder, addFour_, count);
        [encoder endEncoding];
    }

    void addThree(id<MTLCommandBuffer> cb, id<MTLBuffer> a,
                  id<MTLBuffer> b, id<MTLBuffer> c,
                  id<MTLBuffer> output, uint32_t count) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:addThree_];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:c offset:0 atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        dispatchLinear(encoder, addThree_, count);
        [encoder endEncoding];
    }

    void addRowShards(
        id<MTLCommandBuffer> cb, id<MTLBuffer> residual,
        id<MTLBuffer> gpuRows, id<MTLBuffer> aneRows,
        id<MTLBuffer> output, uint32_t rows, uint32_t width,
        uint32_t gpuRowCount) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:addRowShards_];
        [encoder setBuffer:residual offset:0 atIndex:0];
        [encoder setBuffer:gpuRows offset:0 atIndex:1];
        [encoder setBuffer:aneRows offset:0 atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&width length:sizeof(width) atIndex:5];
        [encoder setBytes:&gpuRowCount
                   length:sizeof(gpuRowCount) atIndex:6];
        dispatchLinear(encoder, addRowShards_, rows * width);
        [encoder endEncoding];
    }

    void swigluSegmented(
        id<MTLCommandBuffer> cb, id<MTLBuffer> gpuInput,
        id<MTLBuffer> cpuInput, id<MTLBuffer> aneInput,
        id<MTLBuffer> output, uint32_t rows, uint32_t gpuWidth,
        uint32_t cpuWidth, uint32_t aneWidth) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:swigluSegmented_];
        [encoder setBuffer:gpuInput offset:0 atIndex:0];
        [encoder setBuffer:cpuInput offset:0 atIndex:1];
        [encoder setBuffer:aneInput offset:0 atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder setBytes:&gpuWidth length:sizeof(gpuWidth) atIndex:5];
        [encoder setBytes:&cpuWidth length:sizeof(cpuWidth) atIndex:6];
        [encoder setBytes:&aneWidth length:sizeof(aneWidth) atIndex:7];
        dispatchLinear(
            encoder, swigluSegmented_,
            rows * (gpuWidth + cpuWidth + aneWidth));
        [encoder endEncoding];
    }

    void swigluScatterThree(
        id<MTLCommandBuffer> cb, id<MTLBuffer> gpuInput,
        id<MTLBuffer> cpuInput, id<MTLBuffer> aneInput,
        id<MTLBuffer> output, uint32_t rows, uint32_t gpuWidth,
        uint32_t cpuWidth, uint32_t aneWidth) {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:swigluScatter_];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
        const uint32_t outputWidth = gpuWidth + cpuWidth + aneWidth;
        [encoder setBytes:&outputWidth length:sizeof(outputWidth) atIndex:4];
        auto encodeShard = [&](id<MTLBuffer> input, uint32_t width,
                               uint32_t outputOffset) {
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBytes:&width length:sizeof(width) atIndex:3];
            [encoder setBytes:&outputOffset
                       length:sizeof(outputOffset) atIndex:5];
            const NSUInteger threads = std::min<NSUInteger>(
                256, swigluScatter_.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(rows * width, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        };
        encodeShard(gpuInput, gpuWidth, 0);
        encodeShard(cpuInput, cpuWidth, gpuWidth);
        encodeShard(aneInput, aneWidth, gpuWidth + cpuWidth);
        [encoder endEncoding];
    }

  private:
    static id<MTLComputePipelineState> makePipeline(
        id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                  error:&error];
        if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        return pipeline;
    }

    static void dispatchLinear(id<MTLComputeCommandEncoder> encoder,
                               id<MTLComputePipelineState> pipeline,
                               uint32_t count) {
        const NSUInteger threads =
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }

    id<MTLComputePipelineState> rms_, softmax_, add_, addBias_, rope_;
    id<MTLComputePipelineState> splitQKV_, addFour_;
    id<MTLComputePipelineState> addThree_, addRowShards_, swiglu_;
    id<MTLComputePipelineState> swigluSegmented_, swigluScatter_;
};

struct BlockTimes {
    double wall = 0.0;
    double norm1 = 0.0;
    double qkv = 0.0;
    double attention = 0.0;
    double upgate = 0.0;
    double final = 0.0;
    double modelTail = 0.0;
    ProjectionTimes qkvProjection;
    ProjectionTimes upProjection;
};

class TransformerBlock {
  public:
    TransformerBlock(id<MTLDevice> device, const Options &o)
        : queue_([device newCommandQueue]), o_(o),
          headDim_(o.H / o.heads), kvWidth_(o.kvHeads * headDim_),
          kernels_(device),
          input_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          norm1_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          attentionContext_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          attentionOut_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          residual1_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          norm2_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          norm1Weights_(makeBuffer(device, o.H)),
          norm2Weights_(makeBuffer(device, o.H)),
          qkvBias_(makeBuffer(
              device, o.H + 2 * o.kvHeads * (o.H / o.heads))),
          ropeCos_(makeBuffer(
              device, o.ropeTheta > 0.0f
                  ? static_cast<size_t>(o.M) * (o.H / o.heads) / 2
                  : size_t(1))),
          ropeSin_(makeBuffer(
              device, o.ropeTheta > 0.0f
                  ? static_cast<size_t>(o.M) * (o.H / o.heads) / 2
                  : size_t(1))),
          activated_(makeBuffer(device, static_cast<size_t>(o.M) * o.F)),
          activatedGpu_(makeBuffer(
              device, static_cast<size_t>(o.M) * (o.upGpuN / 2))),
          activatedCpu_(makeBuffer(
              device, static_cast<size_t>(o.M) * (o.upCpuN / 2))),
          activatedAne_(makeBuffer(
              device, static_cast<size_t>(o.M) * (o.upAneN / 2))),
          downPartialGpu_(makeBuffer(
              device, static_cast<size_t>(o.M) * o.H)),
          downPartialCpu_(makeBuffer(
              device, static_cast<size_t>(o.M) * o.H)),
          downPartialAne_(makeBuffer(
              device, static_cast<size_t>(o.M) * o.H)),
          rowUpgate_(makeBuffer(
              device, o.ffnLayout == "row-supergraph"
                  ? static_cast<size_t>(o.rowGpuM) * 2 * o.F
                  : size_t(1))),
          rowActivated_(makeBuffer(
              device, o.ffnLayout == "row-supergraph"
                  ? static_cast<size_t>(o.rowGpuM) * o.F
                  : size_t(1))),
          rowGpuOutput_(makeBuffer(
              device, o.ffnLayout == "row-supergraph"
                  ? static_cast<size_t>(o.rowGpuM) * o.H
                  : size_t(1))),
          rowAneOutput_(makeBuffer(
              device, o.ffnLayout == "row-supergraph"
                  ? static_cast<size_t>(o.M - o.rowGpuM) * o.H
                  : size_t(1))),
          finalGpu_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          finalHetero_(makeBuffer(device, static_cast<size_t>(o.M) * o.H)),
          scores_(makeBuffer(
              device, o.attention == "materialized"
                  ? static_cast<size_t>(o.heads) * o.M * o.M
                  : size_t(1))),
          outWeights_(makeBuffer(device, static_cast<size_t>(o.H) * o.H)),
          downWeights_(makeBuffer(device, static_cast<size_t>(o.F) * o.H)),
          scoreOp_(device, o.M, o.M, headDim_, true,
                   1.0f / std::sqrt(static_cast<float>(headDim_))),
          contextOp_(device, o.M, headDim_, o.M),
          outOp_(device, o.M, o.H, o.H),
          downOp_(device, o.M, o.H, o.F),
          downGpuOp_(device, o.M, o.H, o.upGpuN / 2),
          downCpuOp_(device, o.M, o.H, o.upCpuN / 2),
          downAneOp_(device, o.M, o.H, o.upAneN / 2) {
        uint32_t state = 7;
        if (o.inputData.empty()) {
            fillFP16(static_cast<__fp16 *>(input_.contents),
                     static_cast<size_t>(o.M) * o.H, state, 0.1f);
        } else {
            const auto input = loadWeights(
                o.inputData, static_cast<size_t>(o.M) * o.H);
            std::memcpy(input_.contents, input.data(), input_.length);
        }
        initialInput_ = input_;
        if (o.outWeights.empty()) {
            fillFP16(static_cast<__fp16 *>(outWeights_.contents),
                     static_cast<size_t>(o.H) * o.H, state, 0.015625f);
        } else {
            const auto weights = loadWeights(
                o.outWeights, static_cast<size_t>(o.H) * o.H);
            std::memcpy(
                outWeights_.contents, weights.data(), outWeights_.length);
        }
        auto initializeNorm = [&](id<MTLBuffer> buffer,
                                  const std::string &path) {
            __fp16 *destination =
                static_cast<__fp16 *>(buffer.contents);
            if (path.empty()) {
                std::fill(destination, destination + o.H, __fp16(1.0f));
            } else {
                const auto weights = loadWeights(path, o.H);
                std::memcpy(
                    destination, weights.data(),
                    static_cast<size_t>(o.H) * sizeof(__fp16));
            }
        };
        initializeNorm(norm1Weights_, o.norm1Weights);
        initializeNorm(norm2Weights_, o.norm2Weights);
        std::memset(qkvBias_.contents, 0, qkvBias_.length);
        hasQKVBias_ = !o.qkvBias.empty();
        if (hasQKVBias_) {
            const auto bias = loadWeights(
                o.qkvBias, static_cast<size_t>(o.H) + 2 * kvWidth_);
            std::memcpy(qkvBias_.contents, bias.data(), qkvBias_.length);
        }
        if (o.ropeTheta > 0.0f) {
            __fp16 *cosines =
                static_cast<__fp16 *>(ropeCos_.contents);
            __fp16 *sines =
                static_cast<__fp16 *>(ropeSin_.contents);
            const int halfDim = headDim_ / 2;
            for (int position = 0; position < o.M; ++position) {
                for (int pair = 0; pair < halfDim; ++pair) {
                    const double frequency = std::pow(
                        static_cast<double>(o.ropeTheta),
                        -static_cast<double>(pair) / halfDim);
                    const double angle = position * frequency;
                    const size_t index =
                        static_cast<size_t>(position) * halfDim + pair;
                    cosines[index] =
                        static_cast<__fp16>(std::cos(angle));
                    sines[index] =
                        static_cast<__fp16>(std::sin(angle));
                }
            }
        }
        if (o.downWeights.empty()) {
            fillFP16(static_cast<__fp16 *>(downWeights_.contents),
                     static_cast<size_t>(o.F) * o.H, state, 0.001f);
        } else {
            const auto weights = loadWeights(
                o.downWeights, static_cast<size_t>(o.F) * o.H);
            std::memcpy(
                downWeights_.contents, weights.data(), downWeights_.length);
        }
        if (o.gpuStorage == "private") {
            outWeightsPrivate_ =
                uploadPrivate(device, queue_, outWeights_);
            downWeightsPrivate_ =
                uploadPrivate(device, queue_, downWeights_);
        }

        qkv_ = std::make_unique<TripleProjection>(
            device, queue_, o.M, o.H, o.qkvGpuN, o.qkvCpuN, o.qkvAneN,
            norm1_, o.qkvModel, o.qkvWeights, 101, 0.02f,
            o.gpuStorage == "private", false,
            o.coremlOutput == "backing",
            nullptr, 0, 0, o.qkvGpuWeights);
        if (o.qkvAneN == 0) {
            semanticQ_ =
                makeBuffer(device, static_cast<size_t>(o.M) * o.H);
            semanticKV_ = makeBuffer(
                device, static_cast<size_t>(o.M) * 2 * kvWidth_);
        } else {
            semanticQ_ = qkv_->gpuOutput();
            semanticKV_ = qkv_->aneOutput();
        }
        up_ = std::make_unique<TripleProjection>(
            device, queue_, o.M, o.H, o.upGpuN, o.upCpuN, o.upAneN,
            norm2_, o.upModel, o.upWeights, 202, 0.02f,
            o.gpuStorage == "private", true,
            o.coremlOutput == "backing",
            nullptr, 0,
            o.ffnLayout == "supergraph"
                ? o.H
                : (o.ffnLayout == "activated-pipeline"
                    ? o.upAneN / 2
                    : 0),
            o.upGpuWeights);
        if (o.ffnLayout == "row-supergraph") {
            const int aneRows = o.M - o.rowGpuM;
            rowUpOp_ = std::make_unique<MPSGemm>(
                device, o.rowGpuM, 2 * o.F, o.H);
            rowDownOp_ = std::make_unique<MPSGemm>(
                device, o.rowGpuM, o.H, o.F);
            void *aneInput = static_cast<char *>(norm2_.contents) +
                static_cast<size_t>(o.rowGpuM) * o.H * sizeof(__fp16);
            rowANE_ = std::make_unique<CoreMLGemm>(
                o.rowFFNModel, aneRows, o.H, o.H, aneInput,
                rowAneOutput_.contents);
            rowANEWorker_ =
                std::make_unique<PersistentWorker<CoreMLGemm>>(
                    *rowANE_, false);
        }
        if (o.ffnLayout == "graph-pipeline" ||
            o.ffnLayout == "down-graph-pipeline") {
            const int gpuF = o.upGpuN / 2;
            const int aneF = o.upAneN / 2;
            id<MTLBuffer> suffixShared = makeBuffer(
                device, static_cast<size_t>(aneF) * o.H);
            const __fp16 *source =
                static_cast<const __fp16 *>(downWeights_.contents) +
                static_cast<size_t>(gpuF) * o.H;
            std::memcpy(
                suffixShared.contents, source, suffixShared.length);
            downSuffixPrivate_ =
                uploadPrivate(device, queue_, suffixShared);
            ffnJoinGraph_ = std::make_unique<MPSGraphFFNJoin>(
                o.ffnLayout == "down-graph-pipeline"
                    ? activatedAne_
                    : up_->aneOutput(),
                downSuffixPrivate_, residual1_, downPartialGpu_,
                finalHetero_, o.M, aneF, o.H,
                o.ffnLayout == "down-graph-pipeline");
        }
        if (o.attention == "sdpa") {
            sdpa_ = std::make_unique<MPSGraphAttention>(
                semanticQ_, semanticKV_, attentionContext_,
                o.M, o.heads, o.kvHeads, headDim_);
        } else if (o.attention == "steel") {
            steel_ = std::make_unique<SteelAttention>(
                device, o.M, o.heads, o.kvHeads, headDim_,
                o.steelVariant);
        }
    }

    BlockTimes runGPU() {
        const auto start = Clock::now();
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        kernels_.rmsnorm(
            cb, input_, norm1_, norm1Weights_, o_.M, o_.H, 1e-6f);
        qkv_->encodeFullGPU(cb);
        if (hasQKVBias_) {
            kernels_.addBias(
                cb, qkv_->output(), qkvBias_, o_.M,
                o_.H + 2 * kvWidth_);
        }
        if (o_.attention != "materialized") {
            kernels_.splitQKV(
                cb, qkv_->output(), semanticQ_, semanticKV_,
                o_.M, o_.H, 2 * kvWidth_);
        }
        cb = encodeAttention(cb);
        kernels_.rmsnorm(
            cb, residual1_, norm2_, norm2Weights_, o_.M, o_.H, 1e-6f);
        up_->encodeFullGPU(cb);
        encodeFinal(cb, finalGpu_);
        [cb commit];
        [cb waitUntilCompleted];
        checkCommandBuffer(cb);
        BlockTimes result;
        result.wall = milliseconds(Clock::now() - start);
        return result;
    }

    BlockTimes runHeterogeneous(
        const std::string &layoutOverride = std::string()) {
        BlockTimes result;
        const std::string &layout =
            layoutOverride.empty() ? o_.ffnLayout : layoutOverride;
        const auto blockStart = Clock::now();

        auto stageStart = Clock::now();
        if (o_.qkvAneN == 0) {
            id<MTLCommandBuffer> attentionCB = [queue_ commandBuffer];
            kernels_.rmsnorm(
                attentionCB, input_, norm1_, norm1Weights_,
                o_.M, o_.H, 1e-6f);
            qkv_->encodeFullGPU(attentionCB);
            if (hasQKVBias_) {
                kernels_.addBias(
                    attentionCB, qkv_->output(), qkvBias_, o_.M,
                    o_.H + 2 * kvWidth_);
            }
            if (o_.attention != "materialized") {
                kernels_.splitQKV(
                    attentionCB, qkv_->output(), semanticQ_, semanticKV_,
                    o_.M, o_.H, 2 * kvWidth_);
            }
            attentionCB = encodeAttention(attentionCB);
            kernels_.rmsnorm(
                attentionCB, residual1_, norm2_, norm2Weights_,
                o_.M, o_.H, 1e-6f);
            [attentionCB commit];
            [attentionCB waitUntilCompleted];
            checkCommandBuffer(attentionCB);
            result.attention =
                milliseconds(Clock::now() - stageStart);
        } else {
            runOne([&](id<MTLCommandBuffer> cb) {
                kernels_.rmsnorm(
                    cb, input_, norm1_, norm1Weights_,
                    o_.M, o_.H, 1e-6f);
            });
            result.norm1 = milliseconds(Clock::now() - stageStart);

            id<MTLCommandBuffer> attentionCB = [queue_ commandBuffer];
            result.qkvProjection = qkv_->runHeterogeneous(
                [&] {
                    attentionCB = encodeAttention(attentionCB);
                    kernels_.rmsnorm(
                        attentionCB, residual1_, norm2_, norm2Weights_,
                        o_.M, o_.H, 1e-6f);
                },
                o_.attention == "materialized", nil,
                [&](id<MTLCommandBuffer> cb, id<MTLBuffer> values) {
                    if (hasQKVBias_) {
                        kernels_.addBias(
                            cb, values, qkvBias_, o_.M, o_.H);
                    }
                });
            result.qkv = result.qkvProjection.wall;

            stageStart = Clock::now();
            [attentionCB commit];
            [attentionCB waitUntilCompleted];
            checkCommandBuffer(attentionCB);
            result.attention =
                milliseconds(Clock::now() - stageStart);
        }

        id<MTLCommandBuffer> finalCB = [queue_ commandBuffer];
        if (layout == "row-supergraph") {
            result.upProjection = runRowFFN();
            kernels_.addRowShards(
                finalCB, residual1_, rowGpuOutput_, rowAneOutput_,
                finalHetero_, o_.M, o_.H, o_.rowGpuM);
        } else if (layout == "supergraph") {
            id<MTLCommandBuffer> gpuContinuation = [queue_ commandBuffer];
            result.upProjection = up_->runHeterogeneous(
                [&] {
                    encodeSupergraphGPUContinuation(gpuContinuation);
                },
                false, gpuContinuation);
            encodeSupergraphJoin(finalCB, finalHetero_);
        } else if (layout == "segmented-pipeline") {
            id<MTLCommandBuffer> gpuContinuation = [queue_ commandBuffer];
            result.upProjection = up_->runHeterogeneous(
                [&] {
                    encodeSupergraphGPUContinuation(gpuContinuation);
                },
                false, gpuContinuation);
            encodeSegmentedPipelineJoin(finalCB, finalHetero_);
        } else if (layout == "activated-pipeline") {
            id<MTLCommandBuffer> gpuContinuation = [queue_ commandBuffer];
            result.upProjection = up_->runHeterogeneous(
                [&] {
                    encodeSupergraphGPUContinuation(gpuContinuation);
                },
                false, gpuContinuation);
            encodeActivatedPipelineJoin(finalCB, finalHetero_);
        } else if (layout == "graph-pipeline") {
            id<MTLCommandBuffer> gpuContinuation = [queue_ commandBuffer];
            result.upProjection = up_->runHeterogeneous(
                [&] {
                    encodeSupergraphGPUContinuation(gpuContinuation);
                },
                false, gpuContinuation);
            finalCB = ffnJoinGraph_->encode(finalCB);
        } else if (layout == "down-graph-pipeline") {
            id<MTLCommandBuffer> gpuContinuation = [queue_ commandBuffer];
            result.upProjection = up_->runHeterogeneous(
                [&] {
                    encodeSupergraphGPUContinuation(gpuContinuation);
                },
                false, gpuContinuation);
            kernels_.swiglu(
                finalCB, up_->aneOutput(), activatedAne_,
                o_.M, up_->aneWidth() / 2);
            finalCB = ffnJoinGraph_->encode(finalCB);
        } else {
            result.upProjection = up_->runHeterogeneous(
                [&] {
                    if (layout == "segmented-reduce") {
                    encodeFinalSegmented(finalCB, finalHetero_);
                    } else if (layout == "segmented") {
                        encodeFinalSegmentedConsumer(finalCB, finalHetero_);
                    } else {
                        encodeFinal(finalCB, finalHetero_);
                    }
                },
                layout == "materialized");
        }
        result.upgate = result.upProjection.wall;

        stageStart = Clock::now();
        [finalCB commit];
        [finalCB waitUntilCompleted];
        checkCommandBuffer(finalCB);
        result.final = milliseconds(Clock::now() - stageStart);
        result.wall = milliseconds(Clock::now() - blockStart);
        return result;
    }

    BlockTimes runGPUStack() {
        if (o_.layers == 1) return runGPU();
        const auto start = Clock::now();
        BlockTimes total;
        input_ = initialInput_;
        for (int layer = 0; layer < o_.layers; ++layer) {
            const BlockTimes value = runGPU();
            accumulate(total, value);
            input_ = finalGpu_;
        }
        input_ = initialInput_;
        total.wall = milliseconds(Clock::now() - start);
        return total;
    }

    BlockTimes runHeterogeneousStack(
        const std::string &layoutOverride = std::string()) {
        if (o_.layers == 1) return runHeterogeneous(layoutOverride);
        const auto start = Clock::now();
        BlockTimes total;
        input_ = initialInput_;
        for (int layer = 0; layer < o_.layers; ++layer) {
            const BlockTimes value = runHeterogeneous(layoutOverride);
            accumulate(total, value);
            input_ = finalHetero_;
        }
        input_ = initialInput_;
        total.wall = milliseconds(Clock::now() - start);
        return total;
    }

    struct Accuracy {
        double maxAbs = 0.0;
        double nrmse = 0.0;
        double cosine = 0.0;
    };

    Accuracy accuracy() const {
        const __fp16 *reference =
            static_cast<const __fp16 *>(finalGpu_.contents);
        const __fp16 *actual =
            static_cast<const __fp16 *>(finalHetero_.contents);
        return compare(reference, actual);
    }

    Accuracy accuracyAgainstReference(
        const std::string &path, bool heterogeneous) const {
        const size_t count = static_cast<size_t>(o_.M) * o_.H;
        const auto reference = loadWeights(path, count);
        const __fp16 *actual = static_cast<const __fp16 *>(
            (heterogeneous ? finalHetero_ : finalGpu_).contents);
        return compare(reference.data(), actual);
    }

    Accuracy tailAccuracyAgainstReference(
        const std::string &, bool) {
        return {};
    }

    void setInputBuffer(id<MTLBuffer> input) { input_ = input; }
    id<MTLBuffer> initialInputBuffer() const { return initialInput_; }
    id<MTLBuffer> gpuOutputBuffer() const { return finalGpu_; }
    id<MTLBuffer> heterogeneousOutputBuffer() const {
        return finalHetero_;
    }

    void encodeGPUQKVHeterogeneousPrefix(id<MTLCommandBuffer> cb) {
        if (o_.qkvAneN != 0) {
            throw std::runtime_error(
                "cross-layer GPU prefix requires qkvAneN == 0");
        }
        kernels_.rmsnorm(
            cb, input_, norm1_, norm1Weights_, o_.M, o_.H, 1e-6f);
        qkv_->encodeFullGPU(cb);
        if (hasQKVBias_) {
            kernels_.addBias(
                cb, qkv_->output(), qkvBias_, o_.M,
                o_.H + 2 * kvWidth_);
        }
        if (o_.attention != "materialized") {
            kernels_.splitQKV(
                cb, qkv_->output(), semanticQ_, semanticKV_,
                o_.M, o_.H, 2 * kvWidth_);
        }
        cb = encodeAttention(cb);
        kernels_.rmsnorm(
            cb, residual1_, norm2_, norm2Weights_, o_.M, o_.H, 1e-6f);
    }

    ProjectionTimes runSegmentedFFNAndEncodeTail(
        id<MTLCommandBuffer> tailCB) {
        if (o_.ffnLayout != "segmented-pipeline") {
            throw std::runtime_error(
                "cross-layer transition requires segmented-pipeline");
        }
        id<MTLCommandBuffer> gpuContinuation = [queue_ commandBuffer];
        ProjectionTimes projection = up_->runHeterogeneous(
            [&] {
                encodeSupergraphGPUContinuation(gpuContinuation);
            },
            false, gpuContinuation);
        encodeSegmentedPipelineJoin(tailCB, finalHetero_);
        return projection;
    }

    void encodeHeterogeneousNorm1(id<MTLCommandBuffer> cb) {
        kernels_.rmsnorm(
            cb, input_, norm1_, norm1Weights_, o_.M, o_.H, 1e-6f);
    }

    BlockTimes runHybridQKVFFNAndEncodeTail(
        id<MTLCommandBuffer> tailCB) {
        if (o_.qkvAneN == 0 || o_.ffnLayout != "segmented-pipeline") {
            throw std::runtime_error(
                "hybrid transition requires ANE QKV and segmented-pipeline");
        }
        BlockTimes result;
        id<MTLCommandBuffer> attentionCB = [queue_ commandBuffer];
        result.qkvProjection = qkv_->runHeterogeneous(
            [&] {
                attentionCB = encodeAttention(attentionCB);
                kernels_.rmsnorm(
                    attentionCB, residual1_, norm2_, norm2Weights_,
                    o_.M, o_.H, 1e-6f);
            },
            o_.attention == "materialized", nil,
            [&](id<MTLCommandBuffer> cb, id<MTLBuffer> values) {
                if (hasQKVBias_) {
                    kernels_.addBias(
                        cb, values, qkvBias_, o_.M, o_.H);
                }
            });
        result.qkv = result.qkvProjection.wall;

        auto stageStart = Clock::now();
        [attentionCB commit];
        [attentionCB waitUntilCompleted];
        checkCommandBuffer(attentionCB);
        result.attention = milliseconds(Clock::now() - stageStart);

        result.upProjection =
            runSegmentedFFNAndEncodeTail(tailCB);
        result.upgate = result.upProjection.wall;
        return result;
    }

  private:
    Accuracy compare(const __fp16 *reference, const __fp16 *actual) const {
        const size_t count = static_cast<size_t>(o_.M) * o_.H;
        double errorSq = 0.0, refSq = 0.0, actualSq = 0.0, dot = 0.0;
        double maxAbs = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double r = static_cast<float>(reference[i]);
            const double a = static_cast<float>(actual[i]);
            const double e = a - r;
            maxAbs = std::max(maxAbs, std::abs(e));
            errorSq += e * e;
            refSq += r * r;
            actualSq += a * a;
            dot += r * a;
        }
        return {maxAbs, std::sqrt(errorSq / refSq),
                dot / std::sqrt(refSq * actualSq)};
    }

    static void accumulateProjection(ProjectionTimes &destination,
                                     const ProjectionTimes &source) {
        destination.wall += source.wall;
        destination.gpu += source.gpu;
        destination.cpu += source.cpu;
        destination.ane += source.ane;
        destination.concat += source.concat;
        destination.bridge += source.bridge;
        destination.aneOutputBacking = source.aneOutputBacking;
    }

    static void accumulate(BlockTimes &destination,
                           const BlockTimes &source) {
        destination.wall += source.wall;
        destination.norm1 += source.norm1;
        destination.qkv += source.qkv;
        destination.attention += source.attention;
        destination.upgate += source.upgate;
        destination.final += source.final;
        destination.modelTail += source.modelTail;
        accumulateProjection(
            destination.qkvProjection, source.qkvProjection);
        accumulateProjection(
            destination.upProjection, source.upProjection);
    }

    template <typename Encode>
    void runOne(Encode encode) {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        encode(cb);
        [cb commit];
        [cb waitUntilCompleted];
        checkCommandBuffer(cb);
    }

    id<MTLCommandBuffer> encodeAttention(id<MTLCommandBuffer> cb) {
        if (o_.ropeTheta > 0.0f && o_.attention != "materialized") {
            kernels_.rope(
                cb, semanticQ_, semanticKV_, ropeCos_, ropeSin_,
                o_.M, o_.heads, o_.kvHeads, headDim_);
        }
        if (o_.attention == "sdpa") {
            cb = sdpa_->encode(cb);
        } else if (o_.attention == "steel") {
            steel_->encode(
                cb, semanticQ_, semanticKV_,
                attentionContext_);
        } else {
            id<MTLBuffer> qkv = qkv_->output();
            const int qkvRowBytes = (o_.H + 2 * kvWidth_) * 2;
            const size_t scoreHeadBytes =
                static_cast<size_t>(o_.M) * o_.M * sizeof(__fp16);
            for (int head = 0; head < o_.heads; ++head) {
                const int kvHead = head / (o_.heads / o_.kvHeads);
                const size_t qOffset =
                    static_cast<size_t>(head) * headDim_ * 2;
                const size_t kOffset =
                    static_cast<size_t>(o_.H + kvHead * headDim_) * 2;
                scoreOp_.encode(
                    cb, qkv, qOffset, qkvRowBytes,
                    qkv, kOffset, o_.M, headDim_, qkvRowBytes,
                    scores_, static_cast<size_t>(head) * scoreHeadBytes,
                    o_.M * 2);
            }
            kernels_.softmax(cb, scores_, o_.heads * o_.M, o_.M);
            for (int head = 0; head < o_.heads; ++head) {
                const int kvHead = head / (o_.heads / o_.kvHeads);
                const size_t vOffset =
                    static_cast<size_t>(
                        o_.H + kvWidth_ + kvHead * headDim_) * 2;
                contextOp_.encode(
                    cb, scores_,
                    static_cast<size_t>(head) * scoreHeadBytes, o_.M * 2,
                    qkv, vOffset, o_.M, headDim_, qkvRowBytes,
                    attentionContext_,
                    static_cast<size_t>(head) * headDim_ * 2, o_.H * 2);
            }
        }
        outOp_.encode(cb, attentionContext_, 0, o_.H * 2,
                      outWeightsPrivate_ ? outWeightsPrivate_ : outWeights_,
                      0, o_.H, o_.H, o_.H * 2,
                      attentionOut_, 0, o_.H * 2);
        kernels_.add(cb, input_, attentionOut_, residual1_, o_.M * o_.H);
        return cb;
    }

    void encodeFinal(id<MTLCommandBuffer> cb, id<MTLBuffer> destination) {
        kernels_.swiglu(cb, up_->output(), activated_, o_.M, o_.F);
        downOp_.encode(cb, activated_, 0, o_.F * 2,
                       downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_,
                       0, o_.F, o_.H, o_.H * 2,
                       attentionOut_, 0, o_.H * 2);
        kernels_.add(cb, residual1_, attentionOut_, destination, o_.M * o_.H);
    }

    void encodeFinalSegmented(id<MTLCommandBuffer> cb,
                              id<MTLBuffer> destination) {
        const int gpuF = up_->gpuWidth() / 2;
        const int cpuF = up_->cpuWidth() / 2;
        const int aneF = up_->aneWidth() / 2;
        kernels_.swiglu(
            cb, up_->gpuOutput(), activatedGpu_, o_.M, gpuF);
        kernels_.swiglu(
            cb, up_->cpuOutput(), activatedCpu_, o_.M, cpuF);
        kernels_.swiglu(
            cb, up_->aneOutput(), activatedAne_, o_.M, aneF);

        id<MTLBuffer> downWeights =
            downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_;
        const int downRowBytes = o_.H * 2;
        downGpuOp_.encode(
            cb, activatedGpu_, 0, gpuF * 2,
            downWeights, 0, gpuF, o_.H, downRowBytes,
            downPartialGpu_, 0, downRowBytes);
        downCpuOp_.encode(
            cb, activatedCpu_, 0, cpuF * 2,
            downWeights,
            static_cast<size_t>(gpuF) * o_.H * sizeof(__fp16),
            cpuF, o_.H, downRowBytes,
            downPartialCpu_, 0, downRowBytes);
        downAneOp_.encode(
            cb, activatedAne_, 0, aneF * 2,
            downWeights,
            static_cast<size_t>(gpuF + cpuF) * o_.H * sizeof(__fp16),
            aneF, o_.H, downRowBytes,
            downPartialAne_, 0, downRowBytes);
        kernels_.addFour(
            cb, residual1_, downPartialGpu_, downPartialCpu_,
            downPartialAne_, destination, o_.M * o_.H);
    }

    void encodeSupergraphGPUContinuation(id<MTLCommandBuffer> cb) {
        const int gpuF = up_->gpuWidth() / 2;
        kernels_.swiglu(
            cb, up_->gpuOutput(), activatedGpu_, o_.M, gpuF);
        id<MTLBuffer> downWeights =
            downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_;
        const int downRowBytes = o_.H * 2;
        downGpuOp_.encode(
            cb, activatedGpu_, 0, gpuF * 2,
            downWeights, 0, gpuF, o_.H, downRowBytes,
            downPartialGpu_, 0, downRowBytes);
    }

    ProjectionTimes runRowFFN() {
        const auto start = Clock::now();
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        rowUpOp_->encode(
            cb, norm2_, 0, o_.H * 2,
            up_->fullGPUWeights(), 0, o_.H, 2 * o_.F, 2 * o_.F * 2,
            rowUpgate_, 0, 2 * o_.F * 2);
        kernels_.swiglu(
            cb, rowUpgate_, rowActivated_, o_.rowGpuM, o_.F);
        rowDownOp_->encode(
            cb, rowActivated_, 0, o_.F * 2,
            downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_,
            0, o_.F, o_.H, o_.H * 2,
            rowGpuOutput_, 0, o_.H * 2);

        std::atomic<bool> gate{false};
        rowANEWorker_->submit(&gate);
        [cb commit];
        gate.store(true, std::memory_order_release);
        [cb waitUntilCompleted];
        const double aneMs = rowANEWorker_->wait();
        checkCommandBuffer(cb);
        return {
            milliseconds(Clock::now() - start),
            (cb.GPUEndTime - cb.GPUStartTime) * 1000.0,
            0.0,
            aneMs,
            0.0,
            0.0,
            rowANE_->usedOutputBacking()};
    }

    void encodeSupergraphJoin(id<MTLCommandBuffer> cb,
                              id<MTLBuffer> destination) {
        kernels_.addThree(
            cb, residual1_, downPartialGpu_, up_->aneOutput(),
            destination, o_.M * o_.H);
    }

    void encodeSegmentedPipelineJoin(
        id<MTLCommandBuffer> cb, id<MTLBuffer> destination) {
        const int gpuF = up_->gpuWidth() / 2;
        const int aneF = up_->aneWidth() / 2;
        kernels_.swiglu(
            cb, up_->aneOutput(), activatedAne_, o_.M, aneF);
        id<MTLBuffer> downWeights =
            downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_;
        const int downRowBytes = o_.H * 2;
        downAneOp_.encode(
            cb, activatedAne_, 0, aneF * 2,
            downWeights,
            static_cast<size_t>(gpuF) * o_.H * sizeof(__fp16),
            aneF, o_.H, downRowBytes,
            downPartialAne_, 0, downRowBytes);
        kernels_.addThree(
            cb, residual1_, downPartialGpu_, downPartialAne_,
            destination, o_.M * o_.H);
    }

    void encodeActivatedPipelineJoin(
        id<MTLCommandBuffer> cb, id<MTLBuffer> destination) {
        const int gpuF = up_->gpuWidth() / 2;
        const int aneF = up_->aneWidth() / 2;
        id<MTLBuffer> downWeights =
            downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_;
        const int downRowBytes = o_.H * 2;
        downAneOp_.encode(
            cb, up_->aneOutput(), 0, aneF * 2,
            downWeights,
            static_cast<size_t>(gpuF) * o_.H * sizeof(__fp16),
            aneF, o_.H, downRowBytes,
            downPartialAne_, 0, downRowBytes);
        kernels_.addThree(
            cb, residual1_, downPartialGpu_, downPartialAne_,
            destination, o_.M * o_.H);
    }

    void encodeFinalSegmentedConsumer(id<MTLCommandBuffer> cb,
                                      id<MTLBuffer> destination) {
        kernels_.swigluScatterThree(
            cb, up_->gpuOutput(), up_->cpuOutput(), up_->aneOutput(),
            activated_, o_.M, up_->gpuWidth() / 2, up_->cpuWidth() / 2,
            up_->aneWidth() / 2);
        downOp_.encode(
            cb, activated_, 0, o_.F * 2,
            downWeightsPrivate_ ? downWeightsPrivate_ : downWeights_,
            0, o_.F, o_.H, o_.H * 2,
            attentionOut_, 0, o_.H * 2);
        kernels_.add(
            cb, residual1_, attentionOut_, destination, o_.M * o_.H);
    }

    id<MTLCommandQueue> queue_;
    Options o_;
    int headDim_, kvWidth_;
    Kernels kernels_;
    id<MTLBuffer> input_, initialInput_, norm1_, attentionContext_;
    id<MTLBuffer> semanticQ_, semanticKV_;
    id<MTLBuffer> attentionOut_;
    id<MTLBuffer> residual1_, norm2_, norm1Weights_, norm2Weights_;
    id<MTLBuffer> qkvBias_, ropeCos_, ropeSin_, activated_;
    id<MTLBuffer> activatedGpu_, activatedCpu_, activatedAne_;
    id<MTLBuffer> downPartialGpu_, downPartialCpu_, downPartialAne_;
    id<MTLBuffer> rowUpgate_, rowActivated_, rowGpuOutput_, rowAneOutput_;
    id<MTLBuffer> finalGpu_, finalHetero_, scores_;
    id<MTLBuffer> outWeights_, downWeights_;
    id<MTLBuffer> outWeightsPrivate_, downWeightsPrivate_;
    id<MTLBuffer> downSuffixPrivate_;
    MPSGemm scoreOp_, contextOp_, outOp_, downOp_;
    MPSGemm downGpuOp_, downCpuOp_, downAneOp_;
    std::unique_ptr<MPSGraphAttention> sdpa_;
    std::unique_ptr<MPSGraphFFNJoin> ffnJoinGraph_;
    std::unique_ptr<SteelAttention> steel_;
    std::unique_ptr<TripleProjection> qkv_, up_;
    std::unique_ptr<MPSGemm> rowUpOp_, rowDownOp_;
    std::unique_ptr<CoreMLGemm> rowANE_;
    std::unique_ptr<PersistentWorker<CoreMLGemm>> rowANEWorker_;
    bool hasQKVBias_ = false;
};

class FinalModelTail {
  public:
    FinalModelTail(id<MTLDevice> device, int H, int vocab,
                   const std::string &normWeights,
                   const std::string &headWeights)
        : queue_([device newCommandQueue]), H_(H), vocab_(vocab),
          kernels_(device), normWeights_(makeBuffer(device, H)),
          normalized_(makeBuffer(device, H)),
          logits_(makeBuffer(device, vocab)),
          headWeights_(
              makeBuffer(device, static_cast<size_t>(H) * vocab)),
          headOp_(device, 1, vocab, H) {
        const auto norm = loadWeights(normWeights, H);
        std::memcpy(
            normWeights_.contents, norm.data(), normWeights_.length);
        const auto head = loadWeights(
            headWeights, static_cast<size_t>(H) * vocab);
        std::memcpy(
            headWeights_.contents, head.data(), headWeights_.length);
        headWeightsPrivate_ =
            uploadPrivate(device, queue_, headWeights_);
    }

    double run(id<MTLBuffer> hidden, int inputRow) {
        const auto start = Clock::now();
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        kernels_.rmsnormRow(
            cb, hidden, inputRow, normalized_, normWeights_,
            H_, 1e-6f);
        headOp_.encode(
            cb, normalized_, 0, H_ * sizeof(__fp16),
            headWeightsPrivate_ ? headWeightsPrivate_ : headWeights_,
            0, H_, vocab_, vocab_ * sizeof(__fp16),
            logits_, 0, vocab_ * sizeof(__fp16));
        [cb commit];
        [cb waitUntilCompleted];
        checkCommandBuffer(cb);
        return milliseconds(Clock::now() - start);
    }

    TransformerBlock::Accuracy accuracyAgainstReference(
        const std::string &path) const {
        const auto reference = loadWeights(path, vocab_);
        const __fp16 *actual =
            static_cast<const __fp16 *>(logits_.contents);
        double errorSq = 0.0, refSq = 0.0, actualSq = 0.0, dot = 0.0;
        double maxAbs = 0.0;
        for (int i = 0; i < vocab_; ++i) {
            const double r = static_cast<float>(reference[i]);
            const double a = static_cast<float>(actual[i]);
            const double e = a - r;
            maxAbs = std::max(maxAbs, std::abs(e));
            errorSq += e * e;
            refSq += r * r;
            actualSq += a * a;
            dot += r * a;
        }
        return {
            maxAbs, std::sqrt(errorSq / refSq),
            dot / std::sqrt(refSq * actualSq)};
    }

  private:
    id<MTLCommandQueue> queue_;
    int H_, vocab_;
    Kernels kernels_;
    id<MTLBuffer> normWeights_, normalized_, logits_;
    id<MTLBuffer> headWeights_, headWeightsPrivate_;
    MPSGemm headOp_;
};

class RealTransformerStack {
  public:
    using Accuracy = TransformerBlock::Accuracy;

    RealTransformerStack(id<MTLDevice> device, const Options &options)
        : queue_([device newCommandQueue]), options_(options) {
        if (options_.stackRoot.empty()) {
            throw std::runtime_error("real stack root is empty");
        }
        for (int layer = 0; layer < options_.layers; ++layer) {
            Options current = options_;
            current.layers = 1;
            current.stackRoot.clear();
            current.referenceOutput.clear();
            if (layer > 0) current.inputData.clear();

            std::ostringstream name;
            name << "layer_" << std::setfill('0') << std::setw(2) << layer;
            const std::string directory =
                options_.stackRoot + "/" + name.str();
            auto path = [&](const std::string &file) {
                return directory + "/" + file;
            };

            current.qkvBias = path("qkv.bias.fp16");
            if (current.qkvAneN > 0) {
                current.qkvModel =
                    path("kv_m" + std::to_string(current.M) + ".mlpackage");
                current.qkvWeights = path("kv_ane.weights.fp16");
                current.qkvGpuWeights = path("q_gpu.weights.fp16");
            } else {
                current.qkvGpuWeights = path("qkv_gpu.weights.fp16");
            }

            const int aneF = current.upAneN / 2;
            std::string modelKind = "upgate";
            if (current.ffnLayout == "supergraph") modelKind = "ffn";
            if (current.ffnLayout == "activated-pipeline") {
                modelKind = "activation";
            }
            current.upModel = path(
                modelKind + "_suffix" + std::to_string(aneF) + "_m" +
                std::to_string(current.M) + ".mlpackage");
            current.upWeights = path("upgate_ane.weights.fp16");
            current.upGpuWeights = path("upgate_gpu.weights.fp16");
            current.downWeights = path("down.weights.fp16");
            current.outWeights = path("out.weights.fp16");
            current.norm1Weights = path("norm1.weights.fp16");
            current.norm2Weights = path("norm2.weights.fp16");
            blocks_.push_back(
                std::make_unique<TransformerBlock>(device, current));
        }
        initialInput_ = blocks_.front()->initialInputBuffer();
        if (!options_.finalNormWeights.empty()) {
            tail_ = std::make_unique<FinalModelTail>(
                device, options_.H, options_.vocab,
                options_.finalNormWeights, options_.lmHeadWeights);
        }
    }

    BlockTimes runGPUStack() {
        const auto start = Clock::now();
        BlockTimes total;
        id<MTLBuffer> input = initialInput_;
        for (auto &block : blocks_) {
            block->setInputBuffer(input);
            const BlockTimes value = block->runGPU();
            accumulate(total, value);
            input = block->gpuOutputBuffer();
        }
        if (tail_) {
            total.modelTail = tail_->run(input, options_.M - 1);
        }
        total.wall = milliseconds(Clock::now() - start);
        return total;
    }

    BlockTimes runHeterogeneousStack() {
        if (options_.qkvAneN == 0 &&
            options_.ffnLayout == "segmented-pipeline" &&
            options_.attention == "steel") {
            return runCrossLayerGPUQKVStack();
        }
        if (options_.qkvAneN > 0 &&
            options_.ffnLayout == "segmented-pipeline" &&
            options_.attention == "steel") {
            return runCrossLayerHybridQKVStack();
        }
        const auto start = Clock::now();
        BlockTimes total;
        id<MTLBuffer> input = initialInput_;
        for (auto &block : blocks_) {
            block->setInputBuffer(input);
            const BlockTimes value = block->runHeterogeneous();
            accumulate(total, value);
            input = block->heterogeneousOutputBuffer();
        }
        if (tail_) {
            total.modelTail = tail_->run(input, options_.M - 1);
        }
        total.wall = milliseconds(Clock::now() - start);
        return total;
    }

    Accuracy accuracy() const { return blocks_.back()->accuracy(); }

    Accuracy accuracyAgainstReference(
        const std::string &path, bool heterogeneous) const {
        return blocks_.back()->accuracyAgainstReference(path, heterogeneous);
    }

    Accuracy tailAccuracyAgainstReference(
        const std::string &path, bool heterogeneous) {
        if (!tail_ || path.empty()) return {};
        id<MTLBuffer> hidden = heterogeneous
            ? blocks_.back()->heterogeneousOutputBuffer()
            : blocks_.back()->gpuOutputBuffer();
        tail_->run(hidden, options_.M - 1);
        return tail_->accuracyAgainstReference(path);
    }

  private:
    BlockTimes runCrossLayerGPUQKVStack() {
        const auto start = Clock::now();
        BlockTimes total;
        blocks_.front()->setInputBuffer(initialInput_);

        auto stageStart = Clock::now();
        id<MTLCommandBuffer> prefixCB = [queue_ commandBuffer];
        blocks_.front()->encodeGPUQKVHeterogeneousPrefix(prefixCB);
        [prefixCB commit];
        [prefixCB waitUntilCompleted];
        checkCommandBuffer(prefixCB);
        total.attention += milliseconds(Clock::now() - stageStart);

        for (size_t layer = 0; layer < blocks_.size(); ++layer) {
            id<MTLCommandBuffer> transitionCB = [queue_ commandBuffer];
            const ProjectionTimes projection =
                blocks_[layer]->runSegmentedFFNAndEncodeTail(transitionCB);
            accumulateProjection(total.upProjection, projection);
            total.upgate += projection.wall;

            if (layer + 1 < blocks_.size()) {
                blocks_[layer + 1]->setInputBuffer(
                    blocks_[layer]->heterogeneousOutputBuffer());
                blocks_[layer + 1]->encodeGPUQKVHeterogeneousPrefix(
                    transitionCB);
            }

            stageStart = Clock::now();
            [transitionCB commit];
            [transitionCB waitUntilCompleted];
            checkCommandBuffer(transitionCB);
            const double transition =
                milliseconds(Clock::now() - stageStart);
            if (layer + 1 < blocks_.size()) {
                total.attention += transition;
            } else {
                total.final += transition;
            }
        }
        if (tail_) {
            total.modelTail = tail_->run(
                blocks_.back()->heterogeneousOutputBuffer(),
                options_.M - 1);
        }
        total.wall = milliseconds(Clock::now() - start);
        return total;
    }

    BlockTimes runCrossLayerHybridQKVStack() {
        const auto start = Clock::now();
        BlockTimes total;
        blocks_.front()->setInputBuffer(initialInput_);

        auto stageStart = Clock::now();
        id<MTLCommandBuffer> normCB = [queue_ commandBuffer];
        blocks_.front()->encodeHeterogeneousNorm1(normCB);
        [normCB commit];
        [normCB waitUntilCompleted];
        checkCommandBuffer(normCB);
        total.norm1 += milliseconds(Clock::now() - stageStart);

        for (size_t layer = 0; layer < blocks_.size(); ++layer) {
            id<MTLCommandBuffer> transitionCB = [queue_ commandBuffer];
            const BlockTimes partial =
                blocks_[layer]->runHybridQKVFFNAndEncodeTail(transitionCB);
            total.qkv += partial.qkv;
            total.attention += partial.attention;
            total.upgate += partial.upgate;
            accumulateProjection(
                total.qkvProjection, partial.qkvProjection);
            accumulateProjection(
                total.upProjection, partial.upProjection);

            if (layer + 1 < blocks_.size()) {
                blocks_[layer + 1]->setInputBuffer(
                    blocks_[layer]->heterogeneousOutputBuffer());
                blocks_[layer + 1]->encodeHeterogeneousNorm1(
                    transitionCB);
            }

            stageStart = Clock::now();
            [transitionCB commit];
            [transitionCB waitUntilCompleted];
            checkCommandBuffer(transitionCB);
            const double transition =
                milliseconds(Clock::now() - stageStart);
            if (layer + 1 < blocks_.size()) {
                total.norm1 += transition;
            } else {
                total.final += transition;
            }
        }
        if (tail_) {
            total.modelTail = tail_->run(
                blocks_.back()->heterogeneousOutputBuffer(),
                options_.M - 1);
        }
        total.wall = milliseconds(Clock::now() - start);
        return total;
    }

    static void accumulateProjection(
        ProjectionTimes &destination, const ProjectionTimes &source) {
        destination.wall += source.wall;
        destination.gpu += source.gpu;
        destination.cpu += source.cpu;
        destination.ane += source.ane;
        destination.concat += source.concat;
        destination.bridge += source.bridge;
        destination.aneOutputBacking = source.aneOutputBacking;
    }

    static void accumulate(BlockTimes &destination,
                           const BlockTimes &source) {
        destination.norm1 += source.norm1;
        destination.qkv += source.qkv;
        destination.attention += source.attention;
        destination.upgate += source.upgate;
        destination.final += source.final;
        destination.modelTail += source.modelTail;
        accumulateProjection(
            destination.qkvProjection, source.qkvProjection);
        accumulateProjection(
            destination.upProjection, source.upProjection);
    }

    id<MTLCommandQueue> queue_;
    Options options_;
    id<MTLBuffer> initialInput_;
    std::unique_ptr<FinalModelTail> tail_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
};

static void printResult(const Options &o, NSString *device,
                        const std::vector<BlockTimes> &gpu,
                        const std::vector<BlockTimes> &hetero,
                        TransformerBlock::Accuracy accuracy,
                        TransformerBlock::Accuracy gpuReferenceAccuracy,
                        TransformerBlock::Accuracy heteroReferenceAccuracy,
                        TransformerBlock::Accuracy gpuLogitsAccuracy,
                        TransformerBlock::Accuracy heteroLogitsAccuracy) {
    std::vector<double> gpuWall, heteroWall, norm1, qkv, attention, upgate;
    std::vector<double> final, modelTail;
    std::vector<double> qkvGpu, qkvCpu, qkvAne, qkvConcat;
    std::vector<double> upGpu, upCpu, upAne, upConcat;
    std::vector<double> qkvBridge, upBridge;
    int qkvBackingCount = 0;
    int upBackingCount = 0;
    for (const auto &value : gpu) gpuWall.push_back(value.wall);
    for (const auto &value : hetero) {
        heteroWall.push_back(value.wall);
        norm1.push_back(value.norm1);
        qkv.push_back(value.qkv);
        attention.push_back(value.attention);
        upgate.push_back(value.upgate);
        final.push_back(value.final);
        modelTail.push_back(value.modelTail);
        qkvGpu.push_back(value.qkvProjection.gpu);
        qkvCpu.push_back(value.qkvProjection.cpu);
        qkvAne.push_back(value.qkvProjection.ane);
        qkvConcat.push_back(value.qkvProjection.concat);
        qkvBridge.push_back(value.qkvProjection.bridge);
        if (value.qkvProjection.aneOutputBacking) ++qkvBackingCount;
        upGpu.push_back(value.upProjection.gpu);
        upCpu.push_back(value.upProjection.cpu);
        upAne.push_back(value.upProjection.ane);
        upConcat.push_back(value.upProjection.concat);
        upBridge.push_back(value.upProjection.bridge);
        if (value.upProjection.aneOutputBacking) ++upBackingCount;
    }
    const double gpuP50 = percentile(gpuWall, 0.5);
    const double heteroP50 = percentile(heteroWall, 0.5);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"benchmark\":\"prefill_transformer_block\","
              << "\"device\":\"" << device.UTF8String << "\","
              << "\"M\":" << o.M << ",\"H\":" << o.H
              << ",\"F\":" << o.F << ",\"heads\":" << o.heads
              << ",\"kv_heads\":" << o.kvHeads
              << ",\"layers\":" << o.layers << ","
              << "\"gpu_storage\":\"" << o.gpuStorage << "\","
              << "\"attention\":\"" << o.attention << "\","
              << "\"steel_variant\":\"" << o.steelVariant << "\","
              << "\"ffn_layout\":\"" << o.ffnLayout << "\","
              << "\"coreml_output\":\"" << o.coremlOutput << "\","
              << "\"gpu_wall_p50_ms\":" << gpuP50 << ","
              << "\"gpu_wall_p95_ms\":" << percentile(gpuWall, 0.95) << ","
              << "\"hetero_wall_p50_ms\":" << heteroP50 << ","
              << "\"hetero_wall_p95_ms\":" << percentile(heteroWall, 0.95) << ","
              << "\"speedup\":" << (gpuP50 / heteroP50) << ","
              << "\"hetero_norm1_p50_ms\":" << percentile(norm1, 0.5) << ","
              << "\"hetero_qkv_p50_ms\":" << percentile(qkv, 0.5) << ","
              << "\"hetero_attention_p50_ms\":" << percentile(attention, 0.5) << ","
              << "\"hetero_upgate_p50_ms\":" << percentile(upgate, 0.5) << ","
              << "\"hetero_final_p50_ms\":" << percentile(final, 0.5) << ","
              << "\"model_tail_enabled\":"
              << (!o.finalNormWeights.empty() ? "true" : "false") << ","
              << "\"hetero_model_tail_p50_ms\":"
              << percentile(modelTail, 0.5) << ","
              << "\"qkv_gpu_p50_ms\":" << percentile(qkvGpu, 0.5) << ","
              << "\"qkv_cpu_p50_ms\":" << percentile(qkvCpu, 0.5) << ","
              << "\"qkv_ane_p50_ms\":" << percentile(qkvAne, 0.5) << ","
              << "\"qkv_concat_p50_ms\":" << percentile(qkvConcat, 0.5) << ","
              << "\"qkv_bridge_p50_ms\":" << percentile(qkvBridge, 0.5) << ","
              << "\"qkv_ane_output_backing_rate\":"
              << (hetero.empty() ? 0.0
                                 : double(qkvBackingCount) / hetero.size())
              << ","
              << "\"up_gpu_p50_ms\":" << percentile(upGpu, 0.5) << ","
              << "\"up_cpu_p50_ms\":" << percentile(upCpu, 0.5) << ","
              << "\"up_ane_p50_ms\":" << percentile(upAne, 0.5) << ","
              << "\"up_concat_p50_ms\":" << percentile(upConcat, 0.5) << ","
              << "\"up_bridge_p50_ms\":" << percentile(upBridge, 0.5) << ","
              << "\"up_ane_output_backing_rate\":"
              << (hetero.empty() ? 0.0
                                 : double(upBackingCount) / hetero.size())
              << ","
              << "\"max_abs_error\":" << accuracy.maxAbs << ","
              << "\"nrmse\":" << accuracy.nrmse << ","
              << "\"cosine\":" << accuracy.cosine << ","
              << "\"reference_enabled\":"
              << (o.referenceOutput.empty() ? "false" : "true") << ","
              << "\"gpu_reference_max_abs_error\":"
              << gpuReferenceAccuracy.maxAbs << ","
              << "\"gpu_reference_nrmse\":"
              << gpuReferenceAccuracy.nrmse << ","
              << "\"gpu_reference_cosine\":"
              << gpuReferenceAccuracy.cosine << ","
              << "\"hetero_reference_max_abs_error\":"
              << heteroReferenceAccuracy.maxAbs << ","
              << "\"hetero_reference_nrmse\":"
              << heteroReferenceAccuracy.nrmse << ","
              << "\"hetero_reference_cosine\":"
              << heteroReferenceAccuracy.cosine << ","
              << "\"logits_reference_enabled\":"
              << (o.referenceLogits.empty() ? "false" : "true") << ","
              << "\"gpu_logits_reference_nrmse\":"
              << gpuLogitsAccuracy.nrmse << ","
              << "\"gpu_logits_reference_cosine\":"
              << gpuLogitsAccuracy.cosine << ","
              << "\"hetero_logits_reference_nrmse\":"
              << heteroLogitsAccuracy.nrmse << ","
              << "\"hetero_logits_reference_cosine\":"
              << heteroLogitsAccuracy.cosine << "}\n";
}

template <typename Runner>
static void runNormalBenchmark(
    Runner &runner, const Options &o, id<MTLDevice> device) {
    runner.runGPUStack();
    runner.runHeterogeneousStack();

    std::vector<BlockTimes> gpu, hetero;
    for (int i = 0; i < o.warmup + o.iterations; ++i) {
        @autoreleasepool {
            const bool heteroFirst = o.mode == "both" && (i & 1);
            auto runGPU = [&] {
                const BlockTimes value = runner.runGPUStack();
                if (i >= o.warmup) gpu.push_back(value);
            };
            auto runHetero = [&] {
                const BlockTimes value = runner.runHeterogeneousStack();
                if (i >= o.warmup) hetero.push_back(value);
            };
            if (heteroFirst) {
                runHetero();
                runGPU();
            } else {
                if (o.mode == "gpu" || o.mode == "both") runGPU();
                if (o.mode == "hetero" || o.mode == "both") runHetero();
            }
        }
    }

    runner.runGPUStack();
    runner.runHeterogeneousStack();
    const auto accuracy = runner.accuracy();
    TransformerBlock::Accuracy gpuReferenceAccuracy;
    TransformerBlock::Accuracy heteroReferenceAccuracy;
    if (!o.referenceOutput.empty()) {
        gpuReferenceAccuracy = runner.accuracyAgainstReference(
            o.referenceOutput, false);
        heteroReferenceAccuracy = runner.accuracyAgainstReference(
            o.referenceOutput, true);
    }
    TransformerBlock::Accuracy gpuLogitsAccuracy;
    TransformerBlock::Accuracy heteroLogitsAccuracy;
    if (!o.referenceLogits.empty()) {
        gpuLogitsAccuracy = runner.tailAccuracyAgainstReference(
            o.referenceLogits, false);
        heteroLogitsAccuracy = runner.tailAccuracyAgainstReference(
            o.referenceLogits, true);
    }
    printResult(
        o, device.name, gpu, hetero, accuracy,
        gpuReferenceAccuracy, heteroReferenceAccuracy,
        gpuLogitsAccuracy, heteroLogitsAccuracy);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        try {
            const Options o = parseOptions(argc, argv);
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("Metal device unavailable");
            if (!o.stackRoot.empty()) {
                if (o.mode == "ffn-ablation") {
                    throw std::runtime_error(
                        "ffn-ablation is unavailable for a real stack");
                }
                RealTransformerStack stack(device, o);
                runNormalBenchmark(stack, o, device);
                return 0;
            }
            TransformerBlock block(device, o);

            if (o.mode == "ffn-ablation") {
                const std::vector<std::string> layouts = {
                    "materialized", "segmented", "segmented-reduce"};
                block.runGPU();
                for (const auto &layout : layouts) {
                    block.runHeterogeneous(layout);
                }
                std::vector<BlockTimes> gpuSamples;
                std::vector<std::vector<BlockTimes>> layoutSamples(
                    layouts.size());
                for (int i = 0; i < o.warmup + o.iterations; ++i) {
                    const BlockTimes gpuValue = block.runGPU();
                    if (i >= o.warmup) gpuSamples.push_back(gpuValue);
                    for (size_t step = 0; step < layouts.size(); ++step) {
                        const size_t index =
                            (static_cast<size_t>(i) + step) % layouts.size();
                        const BlockTimes value =
                            block.runHeterogeneous(layouts[index]);
                        if (i >= o.warmup) {
                            layoutSamples[index].push_back(value);
                        }
                    }
                }
                std::vector<TransformerBlock::Accuracy> accuracies;
                for (const auto &layout : layouts) {
                    block.runGPU();
                    block.runHeterogeneous(layout);
                    accuracies.push_back(block.accuracy());
                }
                std::vector<double> gpuWall;
                for (const auto &sample : gpuSamples) {
                    gpuWall.push_back(sample.wall);
                }
                std::cout << std::fixed << std::setprecision(6)
                          << "{\"benchmark\":\"segmented_ffn_ablation\","
                          << "\"device\":\"" << device.name.UTF8String << "\","
                          << "\"coreml_output\":\"" << o.coremlOutput << "\","
                          << "\"gpu_wall_p50_ms\":"
                          << percentile(gpuWall, 0.5) << ","
                          << "\"gpu_wall_p95_ms\":"
                          << percentile(gpuWall, 0.95);
                for (size_t index = 0; index < layouts.size(); ++index) {
                    std::vector<double> walls, finals, upWalls, concats, bridges;
                    for (const auto &sample : layoutSamples[index]) {
                        walls.push_back(sample.wall);
                        finals.push_back(sample.final);
                        upWalls.push_back(sample.upgate);
                        concats.push_back(sample.upProjection.concat);
                        bridges.push_back(sample.upProjection.bridge);
                    }
                    const std::string key = layouts[index] == "segmented-reduce"
                        ? "segmented_reduce"
                        : layouts[index];
                    std::cout
                        << ",\"" << key << "_wall_p50_ms\":"
                        << percentile(walls, 0.5)
                        << ",\"" << key << "_wall_p95_ms\":"
                        << percentile(walls, 0.95)
                        << ",\"" << key << "_up_p50_ms\":"
                        << percentile(upWalls, 0.5)
                        << ",\"" << key << "_final_p50_ms\":"
                        << percentile(finals, 0.5)
                        << ",\"" << key << "_concat_p50_ms\":"
                        << percentile(concats, 0.5)
                        << ",\"" << key << "_bridge_p50_ms\":"
                        << percentile(bridges, 0.5)
                        << ",\"" << key << "_nrmse\":"
                        << accuracies[index].nrmse;
                }
                std::cout << "}\n";
                return 0;
            }

            runNormalBenchmark(block, o, device);
        } catch (const std::exception &error) {
            std::cerr << "error: " << error.what() << "\n";
            return 2;
        }
    }
    return 0;
}
