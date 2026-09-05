#pragma once
#import <Foundation/Foundation.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/memory.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "turbocider/turbocider.h"
#include "contracts.hpp"
namespace tc {
namespace mx = mlx::core;
using Tensor = mx::array;
using Clock = std::chrono::steady_clock;
using Event = std::function<void(const std::string&,int,int)>;
std::string json(id object);
NSDictionary *read_json(const std::filesystem::path&);
NSDictionary *parse_json(const char *);
std::string string_value(NSDictionary *, NSString *, const std::string& fallback = "");
void require(bool, const std::string&);
struct Cancelled : std::runtime_error { Cancelled():std::runtime_error("generation cancelled"){} };
Request request_from_json(NSDictionary *);
void resolve_profile(Request&);
NSDictionary *make_plan(const Request&);
std::vector<float> flux_sigmas(int image_tokens,int steps);
std::vector<float> flux_gpu_sigmas(int image_tokens,int steps);
Recipe model_recipe(const std::string&);
void validate_recipe(const Recipe&);
class ModelSession {
public:
    virtual ~ModelSession() = default;
    virtual NSDictionary *generate(const Request&, const Event&, std::atomic<bool>&) = 0;
};
struct ModelModule {
    std::string id;
    std::function<Recipe()> recipe;
    std::function<void(const Request&)> validate;
    std::function<std::unique_ptr<ModelSession>(const std::filesystem::path&)> create;
    std::function<NSDictionary*()> describe;
};
const ModelModule& module_for(const std::string&);
NSDictionary *describe_modules();
class Weights {
    std::unordered_map<std::string,Tensor> values_;
public:
    void load(const std::filesystem::path&, const Event&, std::atomic<bool>&);
    void load_file(const std::filesystem::path&,const std::string& prefix="");
    const Tensor& at(const std::string&) const;
    bool has(const std::string&) const;
    void clear();
    size_t bytes() const;
};
Tensor linear(const Tensor&, const Weights&, const std::string&);
Tensor silu(const Tensor&);
Tensor rms(const Tensor&, const Tensor&,float eps);
Tensor norm(const Tensor&);
Tensor slice_axis(const Tensor&,int axis,int start,int stop);
Tensor heads(const Tensor&,int count,int dim);
Tensor attend(const Tensor&,const Tensor&,const Tensor&,bool fp32=false,const std::optional<Tensor>& mask={});
Tensor rope_pairs(const Tensor&,const Tensor&,const Tensor&);
Tensor euler_step(const Tensor&,const Tensor&,float dt);
struct Tokens {std::vector<int> ids; int valid=0;};
class Tokenizer {
    NSDictionary *vocab_;
    NSRegularExpression *pattern_;
    std::unordered_map<std::string,int> ranks_;
    std::vector<std::string> byte_encoder_;
    std::map<std::string,int> special_;
    std::vector<int> encode(const std::string&);
public:
    explicit Tokenizer(const std::filesystem::path&);
    Tokens prompt(const std::string&,bool dynamic=true);
};
class HybridSession;
class Flux : public ModelSession {
    std::filesystem::path root_;
    Tokenizer tokenizer_;
    std::unique_ptr<HybridSession> hybrid_;
    Weights transformer_,vae_;
    std::string cached_prompt_;
    bool cached_dynamic_=true;
    std::optional<Tensor> cached_conditioning_;
public:
    explicit Flux(const std::filesystem::path&);
    ~Flux();
    Tensor encode(const Tokens&, const Event&, std::atomic<bool>&);
    Tensor denoise(const Tensor&,const Tensor&,float,int,int,const Event&,std::atomic<bool>&,const std::vector<float>& reference_ids={});
    Tensor encode_image(const Tensor&,const Event&,std::atomic<bool>&);
    Tensor decode(const Tensor&,int,int,const Event&,std::atomic<bool>&,const std::string&);
    NSDictionary *generate(const Request&,const Event&,std::atomic<bool>&) override;
};
Tensor load_image_tensor(const std::filesystem::path&,int,int,bool reference);
void save_png(const Tensor&,const std::filesystem::path&);
NSDictionary *system_info();
NSDictionary *compile_artifact(const std::filesystem::path&,const std::filesystem::path&);
void checkpoint(std::atomic<bool>&);
}
struct tc_engine {
    std::unique_ptr<tc::ModelSession> session;
    std::mutex mutex;
    std::atomic<bool> cancelled{false};
};
