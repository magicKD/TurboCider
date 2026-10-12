#pragma once
#include "../../backends/ane_memory.hpp"
#include "../../backends/mlx.hpp"
#include "../../core/tokenizer.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include <limits>

namespace tc::qwen21 {
struct EncoderPrefillPack {
    uint32_t qkv = 0, qk = 0, gate_up = 0;
};

inline EncoderPrefillPack pack_encoder_prefill(Weights &weights, std::atomic<bool> &cancel) {
    require(!weights.has_runtime_loras(), "encoder prefill fusion requires original base weights");
    EncoderPrefillPack result;
    const std::string stem = weights.has("model.language_model.embed_tokens.weight") ?
        "model.language_model." : "model.";
    auto fuse = [&](const std::string &target, const std::vector<std::string> &names) {
        const bool packed = weights.quantized(names[0]);
        const auto &first = weights.at(names[0] + ".weight");
        uint64_t growth = uint64_t(1) << 20;
        auto charge = [&](uint64_t bytes) {
            require(bytes <= std::numeric_limits<uint64_t>::max() - growth,
                    "encoder prefill fusion growth overflow");
            growth += bytes;
        };
        for (const auto &name : names) {
            // Original Qwen3-VL projections have no additive bias. Do not
            // silently discard one on an incompatible source/fixture.
            if (weights.quantized(name) != packed || weights.convrot(name) ||
                weights.nvfp4(name) || weights.has(name + ".bias")) return false;
            const auto &a = weights.at(name + ".weight");
            if (a.ndim() != 2 || first.ndim() != 2 ||
                a.shape(1) != first.shape(1) || a.dtype() != first.dtype()) return false;
            charge(a.nbytes());
            if (packed) {
                if (!weights.has(name + ".biases")) return false;
                const auto &s = weights.at(name + ".scales"), &b = weights.at(name + ".biases");
                const auto &first_scales = weights.at(names[0] + ".scales");
                if (s.ndim() != 2 || first_scales.ndim() != 2 || s.shape(0) != a.shape(0) ||
                    s.shape(1) != first_scales.shape(1) || s.dtype() != first_scales.dtype() ||
                    b.shape() != s.shape() || b.dtype() != s.dtype()) return false;
                charge(s.nbytes());
                charge(b.nbytes());
            }
        }
        const auto observed = ane::observe_runtime_memory(mx::get_active_memory());
        const auto admission = ane::admit_memory(observed,
            {uint64_t(4) << 30, observed.physical_bytes}, 0, growth);
        require(admission.allowed(), "encoder packed fusion admission declined: " +
            ane::memory_denial_reason(admission.denial, observed));
        // All three affine planes retain their exact integer/typed fields.
        // Materialize each concat before releasing its lazy source readers.
        for (const auto *plane : packed ? std::vector<const char *>{".weight", ".scales", ".biases"} :
                                         std::vector<const char *>{".weight"}) {
            checkpoint(cancel);
            std::vector<std::string> keys;
            for (const auto &name : names) keys.push_back(name + plane);
            weights.fuse_keys(target + plane, keys, 0);
            mx::eval(weights.at(target + plane));
        }
        return true;
    };
    for (int i = 0; i < 36; ++i) {
        checkpoint(cancel);
        const auto p = stem + "layers." + std::to_string(i) + ".";
        if (fuse(p + "self_attn.qkv_proj", {p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"})) ++result.qkv;
        else if (fuse(p + "self_attn.qk_proj", {p + "self_attn.q_proj", p + "self_attn.k_proj"})) ++result.qk;
        if (fuse(p + "mlp.gate_up", {p + "mlp.gate_proj", p + "mlp.up_proj"})) ++result.gate_up;
    }
    return result;
}

class VerifiedEncoderTokenizer final {
    std::shared_ptr<const streaming::SourceLease> lease_;
    std::unique_ptr<Tokenizer> tokenizer_;
  public:
    VerifiedEncoderTokenizer(const std::filesystem::path &root, std::atomic<bool> &cancel) {
        streaming::SourceFileIdentity source;
        source.logical_id = "processor";
        source.path = root / "tokenizer.json";
        lease_ = streaming::SourceLease::capture_verified({source}, &cancel);
        auto fd = lease_->duplicate_fd("processor");
        tokenizer_ = std::make_unique<Tokenizer>(fd.get(), lease_->file("processor").bytes);
        lease_->revalidate_after_drain();
    }
    void check_unchanged() const { lease_->revalidate_after_drain(); }
    const Tokenizer &tokenizer() const { return *tokenizer_; }
    std::string identity() const { return std::string(lease_->digest()); }
    std::string sha256() const { return lease_->file("processor").content_digest; }
};
}
