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
