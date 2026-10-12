#include "../../native/backends/ane_fp16_value_config.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/backends/private/ane_mil_round.hpp"
#include "../../native/backends/private/ane_mil_round_compact.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <regex>
#include <set>

int main() {
    using namespace tc::ane;
    unsetenv("TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES");
    assert(!configured_fp16_bf16_values());
    for(const char *value:{"0","1"}) {
        setenv("TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES",value,1);
        assert(configured_fp16_bf16_values()==(std::string(value)=="1"));
    }
    for(const char *value:{"", "yes", "01", "-1"}) {
        setenv("TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES",value,1);
        bool rejected=false;try{(void)configured_fp16_bf16_values();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    unsetenv("TURBOCIDER_PRIVATE_ANE_FP16_BF16_VALUES");
    GraphShape shape{Kind::SwiGLU,32,128,512,64,128,true};
    const auto original=private_api::fp16_program(shape);
    assert(original==private_api::fp16_program(shape,false));
    assert(original.find("gate_bf16")==std::string::npos);
    const auto recipe=private_api::fp16_program(shape,true);
    std::set<std::string> symbols;
    const std::regex declaration("\n        tensor<[^\n]+> ([A-Za-z0-9_]+) =");
    for(std::sregex_iterator it(recipe.begin(),recipe.end(),declaration),end;it!=end;++it)
        assert(symbols.insert((*it)[1].str()).second);
    for(const char *symbol:{"gate_bf16", "up_bf16", "sigmoid_bf16", "silu_bf16", "hidden_bf16", "down_bf16"})
        assert(recipe.find(symbol)!=std::string::npos);
    assert(recipe.find("carrier_invalid")!=std::string::npos);
    assert(recipe.find("65504")!=std::string::npos);
    assert(recipe.find("0x1.ffp+15")!=std::string::npos);
    assert(recipe.find("tensor<bf16")==std::string::npos);
    assert(recipe.find(", hidden_bf16_out)")!=std::string::npos);
    std::string legacy,compact;
    private_api::emit_bf16_value_round(legacy,"x","test","[1, 1, 1, 4096]");
    private_api::emit_bf16_value_round_compact(compact,"x","test","[1, 1, 1, 4096]");
    const auto old_count=std::count(legacy.begin(),legacy.end(),';'),new_count=std::count(compact.begin(),compact.end(),';');
    assert(new_count*2<old_count);
    shape.kind=Kind::Matmul;shape.lora_inputs=false;
    bool rejected=false;try{(void)private_api::fp16_program(shape,true);}catch(const CapabilityError&){rejected=true;}
    assert(rejected);
    std::cout<<"PASS Private FP16 compact value policy/default MIL/carrier/packed hidden; operators="<<new_count<<" legacy="<<old_count<<'\n';
}
