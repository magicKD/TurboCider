#include "backends/lora_metadata.hpp"

#include <cassert>
#include <iostream>

int main() {
    using tc::lora_metadata::inspect;
    auto parse=[](const std::string& json) {
        return inspect({{"format","pt"},{"lora_adapter_metadata",json}});
    };
    assert(inspect({}).rejection_for(16,{}).empty());
    const auto local=parse(R"({"transformer.alpha_pattern":{},"transformer.bias":"none",
      "transformer.init_lora_weights":true,"transformer.lora_alpha":128,
      "transformer.lora_dropout":0.0,"transformer.peft_type":"LORA",
      "transformer.r":128,"transformer.rank_pattern":{},
      "transformer.target_modules":["img_mlp.gate_layer","modulation.1"],
      "transformer.use_rslora":false})");
    assert(local.rejection_for(128,{}).empty());
    assert(local.rejection_for(128,128.f).empty());
    assert(!local.rejection_for(16,{}).empty());
    assert(!local.rejection_for(128,64.f).empty());
    assert(!parse(R"({"r":16,"lora_alpha":8})").rejection_for(16,{}).empty());
    assert(parse(R"({"r":16,"lora_alpha":8})").rejection_for(16,8.f).empty());
    const auto dmd2 = parse(R"({"transformer.r":32,"transformer.lora_alpha":32,
      "transformer.init_lora_weights":"gaussian","transformer.use_qalora":false,
      "transformer.ensure_weight_tying":false,"transformer.lora_bias":false,
      "transformer.qalora_group_size":16,"transformer.peft_version":"0.20.0",
      "transformer.megatron_core":"megatron.core","transformer.loftq_config":{}})");
    assert(dmd2.rejection_for(32,{}).empty());
    for (const auto *name : {"alora_invocation_tokens", "arrow_config", "auto_mapping",
         "base_model_name_or_path", "corda_config", "eva_config", "exclude_modules",
         "layer_replication", "layers_pattern", "layers_to_transform", "lora_ga_config",
         "megatron_config", "monteclora_config", "revision", "target_parameters", "task_type",
         "trainable_token_indices", "use_bdlora", "velora_config"}) {
        assert(parse(std::string("{\"") + name + "\":null}").rejection_for(32,{}).empty());
        assert(!parse(std::string("{\"") + name + "\":true}").rejection_for(32,{}).empty());
    }
    for (const auto *json : {R"({"use_qalora":true})", R"({"ensure_weight_tying":true})",
         R"({"lora_bias":true})", R"({"qalora_group_size":32})", R"({"peft_version":20})",
         R"({"megatron_core":"other"})", R"({"loftq_config":{"bits":4}})",
         R"({"init_lora_weights":"pissa"})"})
        assert(!parse(json).rejection_for(32,{}).empty());
    for (const auto* json:{R"({"use_rslora":true})",R"({"use_dora":true})",
         R"({"fan_in_fan_out":true})",R"({"alpha_pattern":{"foo":8}})",
         R"({"rank_pattern":{"foo":16}})",R"({"peft_type":"IA3"})",
         R"({"bias":"all"})",R"({"unknown":false})",R"({"r":16,"r":16})",
         R"({"r":16,"transformer.r":16})",R"({"r":01})",R"({"r":1e999})",
         R"({"r":16} trailing)",R"({"r":16,})",R"({"r":NaN})",
         R"({"r":"128"})",R"({"target_modules":[12]})"})
        assert(!parse(json).rejection_for(128,{}).empty());
    assert(parse(R"({"\u0072":128,"lora_alpha":128})").rejection_for(128,{}).empty());
    assert(!parse(std::string(65537,' ')).rejection.empty());
    assert(!inspect({{"ss_network_alpha","8"}}).rejection.empty());
    std::cout << "PASS native LoRA metadata: tensor-alpha equivalence, local metadata, unsupported variants, duplicate keys and bounds\n";
}
