#include "../../native/models/qwen21/diagnostic_options.hpp"
#include <cassert>

using tc::qwen21::W8A8CallBudget;
using tc::qwen21::expected_w8a8_calls;

int main() {
    using namespace tc::qwen21;
    assert(!option_enabled(nullptr));
    assert(!option_enabled("0"));
    assert(!option_enabled(""));
    assert(!option_enabled("true"));
    assert(option_enabled("1"));
    assert(binary_option_or_unset(nullptr));
    assert(binary_option_or_unset("0"));
    assert(binary_option_or_unset("1"));
    assert(!binary_option_or_unset("true"));
    assert(runtime_ffn_phase(nullptr)==RuntimeFfnPhase::All);
    assert(runtime_ffn_phase("all")==RuntimeFfnPhase::All);
    assert(runtime_ffn_phase("prefill")==RuntimeFfnPhase::Prefill);
    assert(runtime_ffn_phase("decode")==RuntimeFfnPhase::Decode);
    for(const char *bad:{"","0","1","ALL","gpu","prefill,decode"})
        assert(runtime_ffn_phase(bad)==RuntimeFfnPhase::Invalid);
    assert(runtime_ffn_phase_identity(RuntimeFfnPhase::All).empty());
    assert(runtime_ffn_phase_identity(RuntimeFfnPhase::Prefill)!=runtime_ffn_phase_identity(RuntimeFfnPhase::Decode));
    for(auto phase:{RuntimeFfnPhase::All,RuntimeFfnPhase::Prefill,RuntimeFfnPhase::Decode}) {
        assert(runtime_ffn_phase_runs(phase,false)==(phase!=RuntimeFfnPhase::Decode));
        assert(runtime_ffn_phase_runs(phase,true)==(phase!=RuntimeFfnPhase::Prefill));
    }
    assert(tiled_prefill_layer_count("0") == 0);
    assert(tiled_prefill_layer_count("1") == 32);
    assert(tiled_prefill_layer_count("16") == 16);
    assert(tiled_prefill_layer_count("17") == -1);
    assert(db_cache_threshold(nullptr) == 0.08f);
    assert(db_cache_threshold("0.25") == 0.25f);
    assert(db_cache_threshold("nan") < 0);
    assert(db_cache_threshold("0.51") < 0);
    assert(db_cache_max_consecutive(nullptr) == 2);
    assert(db_cache_max_consecutive("1") == 1);
    assert(db_cache_max_consecutive("4") == 4);
    assert(db_cache_max_consecutive("8") == 8);
    assert(db_cache_max_consecutive("0") < 0);
    assert(db_cache_max_consecutive("9") < 0);
    assert(db_cache_max_consecutive("4oops") < 0);

    tc::Request lora;
    lora.model="qwen-image-2.1";
    lora.width=lora.height=1024;lora.steps=6;lora.allow_approximation=true;
    lora.loras.push_back({"adapter.safetensors",1.f,"transformer"});
    unsetenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC");
    assert(!lora_1024_generation(lora));
    setenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC","1",1);
    assert(lora_1024_generation(lora));
    lora.execution="gpu_ane";lora.hybrid_mlp_mode="runtime";
    assert(lora_1024_generation(lora));
    for(int mode=0;mode<11;++mode) {
        auto invalid=lora;
        if(mode==0)invalid.width=512;
        if(mode==1)invalid.steps=5;
        if(mode==2)invalid.allow_approximation=false;
        if(mode==3)invalid.operation="image.edit";
        if(mode==4)invalid.inputs.resize(1);
        if(mode==5)invalid.loras.clear();
        if(mode==6)invalid.residency="component_staged";
        if(mode==7)invalid.qwen21_w8a8=true;
        if(mode==8)invalid.hybrid_mlp_mode="lora_fused";
        if(mode==9)invalid.prompt_enhance=true;
        if(mode==10)invalid.qwen21_gpu_full_ffn_blocks={3,5,7};
        assert(!lora_1024_generation(invalid));
    }
    setenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC","0",1);
    assert(!lora_1024_generation(lora));
    unsetenv("TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC");

    W8A8CallBudget budget{
        .steps = 20, .decode_layers = 32, .db_skipped_layers = 24,
        .total_tokens = 4096, .tile_rows = 1024,
    };
    assert(expected_w8a8_calls(budget) == 608);
    budget.db_cached_steps = 8;
    assert(expected_w8a8_calls(budget) == 416);
    budget.steps = 40;
    budget.db_cached_steps = 21;
    assert(expected_w8a8_calls(budget) == 744);

    budget.steps = 5;
    budget.db_cached_steps = 0;
    budget.final_reuse_layers = 16;
    budget.penultimate_reuse_layers = 16;
    assert(expected_w8a8_calls(budget) == 96);
    budget.final_reuse_layers = budget.penultimate_reuse_layers = 0;
    budget.tiled_prefill_layers = 16;
    assert(expected_w8a8_calls(budget) == 192);
    budget.last_target_only = true;
    assert(expected_w8a8_calls(budget) == 189);

    budget.prefix_hit = budget.tiled_prefix_reuse = true;
    budget.prefix_tokens = 2100; // incomplete prefix tile
    assert(expected_w8a8_calls(budget) == 159);
    budget.prefix_target_only = true;
    assert(expected_w8a8_calls(budget) == 144);
    budget.prefix_target_only = false;
    budget.prefix_tokens = 2048; // aligned prefix, no tile tail
    assert(expected_w8a8_calls(budget) == 144);

    budget.tiled_prefill_layers = 0;
    budget.decode_layers = 29; // 3/5/7 full GPU fallback
    assert(expected_w8a8_calls(budget) == 116);
    budget.decode_layers = 32;
    budget.decode_tiles = 2; // 1536-row rectangle on a 1024-row graph
    assert(expected_w8a8_calls(budget) == 256);
}
