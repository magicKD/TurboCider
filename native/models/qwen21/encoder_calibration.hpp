#pragma once
#include "text_encoder.hpp"
#include "../../backends/ane_smoothquant.hpp"
#include "../../platform/apple/platform.hpp"
#include <mlx/fast.h>
#include <fstream>
#include <cmath>

namespace tc::qwen21 {
inline std::filesystem::path encoder_profile_path(const char* key) {
    const char* raw=std::getenv(key);if(!raw || !*raw)return {};
    std::filesystem::path p(raw);
    require(p.is_absolute() && p==p.lexically_normal() && p.extension()==".safetensors" && p.native().size()<=4096,
        "encoder calibration requires a normalized absolute safetensors path");return p;
}
struct EncoderScaleProfile {
    std::string digest;
    std::vector<float> scales;
    static EncoderScaleProfile load(const std::filesystem::path& path,const std::filesystem::path& checkpoint,
                                   int reference_count,int reference_size,int channels) {
        require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path)<=(2ULL<<20),"encoder profile exceeds2MiB bound");
        const auto digest_before=sha256_file(path);
        auto [arrays,metadata]=mx::load_safetensors(path.string());
        const auto exact=[&](const char* key,const std::string& expected) {
            auto i=metadata.find(key);require(i!=metadata.end() && i->second==expected,std::string("encoder profile binding mismatch: ")+key);
        };
        exact("schema","qwen3vl-encoder-s1-v1");exact("trajectory","complete-gpu-encoder");
        exact("checkpoint",ane::smoothquant::checkpoint_fingerprint(checkpoint));
        exact("reference_count",std::to_string(reference_count));exact("reference_size",std::to_string(reference_size));
        exact("ane_channels",std::to_string(channels));exact("hidden","4096");exact("width","12288");exact("layers","36");
        exact("alpha","0.5");exact("basis","sylvester-dh-input128-hidden512-seed20260930");exact("qualified","false");
        require(arrays.size()==1 && arrays.contains("s1"),"encoder profile must contain only s1 scales");
        auto s=arrays.at("s1");require(s.shape()==mx::Shape{36,4096} && s.dtype()==mx::float32,"encoder profile requires36x4096 FP32 scales");
        mx::eval(s);require(s.flags().row_contiguous,"encoder scales must be contiguous");
        EncoderScaleProfile result{digest_before,{s.data<float>(),s.data<float>()+s.size()}};
        for(float v:result.scales)require(std::isfinite(v) && v>=.0625f && v<=16.f,"invalid encoder S1 scale");
        require(digest_before==sha256_file(path),"encoder profile changed during load");return result;
    }
    std::vector<Tensor> tensors() const {
        require(scales.size()==36*4096,"partial encoder scale profile");std::vector<Tensor> result;
        for(int i=0;i<36;++i)result.emplace_back(scales.data()+i*4096,mx::Shape{1,4096},mx::float32);
        return result;
    }
};
// Streaming, all36 language layers; only channel statistics/scales reach disk.
// No activation/checkpoint export and no DiT profile/LoRA identity substitution.
class EncoderCalibration {
    const Weights& weights_;
    std::filesystem::path checkpoint_;
    std::string source_identity_,references_;
    int references_count_,reference_size_,channels_,next_=0;
    std::vector<Tensor> scales_;
public:
    EncoderCalibration(const Weights& weights,std::filesystem::path checkpoint,int references_count,
                       int reference_size,int channels,std::string reference_hashes)
      :weights_(weights),checkpoint_(std::move(checkpoint)),source_identity_(ane::smoothquant::checkpoint_fingerprint(checkpoint_)),
       references_(std::move(reference_hashes)),references_count_(references_count),reference_size_(reference_size),channels_(channels) {
        require(references_count>=1 && references_count<=3 && (reference_size==512 || reference_size==1024) &&
            channels>0 && channels<12288 && channels%512==0 && !references_.empty(),"invalid encoder calibration domain");
    }
    void observe(int layer,const std::string& p,const Tensor& x) {
        require(layer==next_ && layer<36 && x.ndim()==3 && x.shape(0)==1 && x.shape(2)==4096 && x.dtype()==mx::bfloat16,
            "encoder calibration requires ordered complete GPU layers");
        auto g=weights_.at(p+".mlp.gate_proj.weight"),u=weights_.at(p+".mlp.up_proj.weight");
        require(g.shape()==mx::Shape{12288,4096} && u.shape()==g.shape() && g.dtype()==mx::bfloat16 && u.dtype()==g.dtype(),"encoder calibration source mismatch");
        // Coalesced physical reads, one H-vector output. No abs/full FP32 W.
        static auto kernel=mx::fast::metal_kernel("tc_qwen_encoder_weight_channel_max",{"g","u"},{"out"},R"metal(
            uint c=thread_position_in_grid.x;if(c>=4096)return;
            float maximum=0;
            #pragma clang loop unroll(disable)
            for(uint r=FIRST;r<12288;++r) {
                maximum=max(maximum,abs(float(g[r*4096+c])));
                maximum=max(maximum,abs(float(u[r*4096+c])));
            }
            out[c]=maximum;
        )metal");
        auto wmax=kernel({g,u},{{1,4096}},{mx::float32},{4096,1,1},{128,1,1},{{"FIRST",12288-channels_}}, {},false,{})[0];
        auto amax=mx::reshape(mx::astype(mx::max(mx::abs(x),{0,1}),mx::float32),{1,4096});
        auto s=mx::clip(mx::sqrt(mx::maximum(amax,Tensor(1e-6f)))/mx::sqrt(mx::maximum(wmax,Tensor(1e-6f))),Tensor(.0625f),Tensor(16.f));
        auto finite=mx::all(mx::isfinite(s));mx::eval(s,finite);require(finite.item<bool>(),"nonfinite encoder calibration");
        scales_.push_back(std::move(s));++next_;
    }
    void write(const std::filesystem::path& path) const {
        require(next_==36 && scales_.size()==36,"cannot publish incomplete encoder calibration");
        require(!std::filesystem::exists(path) && path.is_absolute(),"encoder calibration refuses to overwrite");
        require(ane::smoothquant::checkpoint_fingerprint(checkpoint_)==source_identity_,"encoder calibration source changed");
        auto s=mx::concatenate(scales_,0);mx::eval(s);
        mx::save_safetensors(path.string(),{{"s1",s}},{{"schema","qwen3vl-encoder-s1-v1"},{"trajectory","complete-gpu-encoder"},
            {"checkpoint",source_identity_},{"reference_count",std::to_string(references_count_)},{"reference_size",std::to_string(reference_size_)},
            {"ane_channels",std::to_string(channels_)},{"hidden","4096"},{"width","12288"},{"layers","36"},{"alpha","0.5"},
            {"basis","sylvester-dh-input128-hidden512-seed20260930"},{"reference_sha256",references_},{"qualified","false"}});
    }
};
} // namespace tc::qwen21
