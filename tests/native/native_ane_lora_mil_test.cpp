#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/backends/metal_graph/native_ane_lora_contract.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <set>

namespace contract=tc::metal_graph::native_ane_lora_contract;
using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
size_t checks=0;
void check(bool value,const char* message) {
    ++checks;if(!value)throw std::runtime_error(message);
}
std::string read(const std::string& path) {
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("missing baseline fixture");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
size_t occurrences(const std::string& text,std::string_view needle) {
    size_t count=0,pos=0;while((pos=text.find(needle,pos))!=std::string::npos){++count;pos+=needle.size();}return count;
}
void input_abi(const std::string& mil,const contract::Spec& spec) {
    const auto end=mil.find(") {\n");check(end!=std::string::npos,"native main signature missing");
    const auto signature=mil.substr(0,end);
    check(occurrences(signature,"tensor_buffer<")==spec.inputs().size(),"input count disagrees with spec");
    size_t previous=0;
    for(const auto& input:spec.inputs()) {
        const bool int8=input.type==tc::metal_graph::ScalarType::Int8;
        check(int8 || input.type==tc::metal_graph::ScalarType::FP16,"unsupported ABI input dtype");
        const auto item=tc::metal_graph::scalar_bytes(input.type);
        check(input.pitch_bytes%item==0 && input.payload_bytes%item==0,"nonintegral ABI element strides");
        // The production Spec supplies byte extents; MIL encodes element
        // strides. This checks the cross-module ABI, not just input names.
        const auto pitch=std::to_string(input.pitch_bytes/item);
        const auto plane=std::to_string(input.payload_bytes/item);
        const auto declaration=std::string("tensor_buffer<")+(int8?"int8":"fp16")+
            ", shape=[1, 1, "+std::to_string(input.rows)+", "+std::to_string(input.columns)+
            "], strides=["+plane+", "+plane+", "+pitch+", 1], interleave_factors=[1, 1, 1, 1]> "+
            std::string(input.name);
        const auto pos=signature.find(declaration,previous);
        check(pos!=std::string::npos,"emitted input dtype/shape/pitch/name/order differs from Spec");
        const auto after=pos+declaration.size();
        check(after==signature.size() || signature.compare(after,2,", ")==0,
            "input declaration matched only a prefix of another name");
        previous=after;
    }
}
void final_guard_dependencies(const std::string& mil) {
    const std::regex token_pattern(R"(\b(native_[A-Za-z0-9_]*_token) = reduce_max\()");
    const std::regex node_pattern(R"(\b(native_all_bad[0-9]+) = maximum\(x = ([A-Za-z0-9_]+), y = ([A-Za-z0-9_]+)\))");
    const std::regex root_pattern(R"(\bnative_all_ok = less\(x = ([A-Za-z0-9_]+), y = fp16\(0\.5\)\))");
    std::set<std::string> tokens,visited_tokens,active,visited_nodes;
    std::map<std::string,std::pair<std::string,std::string>> nodes;
    for(auto it=std::sregex_iterator(mil.begin(),mil.end(),token_pattern);it!=std::sregex_iterator();++it)
        check(tokens.insert((*it)[1].str()).second,"finite token defined more than once");
    for(auto it=std::sregex_iterator(mil.begin(),mil.end(),node_pattern);it!=std::sregex_iterator();++it)
        check(nodes.emplace((*it)[1].str(),std::pair{(*it)[2].str(),(*it)[3].str()}).second,
            "final guard node defined more than once");
    check(!tokens.empty() && tokens.size()<=64 && nodes.size()<=64,"empty/unbounded finite dependency graph");
    std::smatch root;check(std::regex_search(mil,root,root_pattern),"final finite root is missing");
    const auto walk=[&](const auto& self,const std::string& symbol)->void {
        if(tokens.contains(symbol)) {visited_tokens.insert(symbol);return;}
        const auto node=nodes.find(symbol);
        check(node!=nodes.end(),"final aggregate references a foreign/non-token dependency");
        check(active.insert(symbol).second,"cycle in final finite aggregation");
        self(self,node->second.first);self(self,node->second.second);
        active.erase(symbol);visited_nodes.insert(symbol);
    };
    walk(walk,root[1].str());
    for(const auto& token:tokens)
        check(visited_tokens.contains(token),"declared finite token does not reach native_all_ok");
    check(visited_nodes.size()==nodes.size(),"dead native_all_bad node outside final guard");
    check(mil.find("native_checked_hscale = select(cond = native_all_ok, a = hscale, b = fp16(-1))")!=std::string::npos,
        "final guard does not control the published hscale sentinel");
}
template<class F> void rejects_mutation(F&& f,const char* message) {
    bool rejected=false;
    try {f();} catch(const std::runtime_error&) {rejected=true;}
    check(rejected,message);
}
void legacy(const std::string& directory) {
    for(int i=0;i<4;++i) {
        const bool tiny=i<2,lora=i%2;
        const GraphShape shape{Kind::SwiGLU,tiny?32:1024,tiny?128:4096,tiny?512:6144,0,0,lora};
        const auto p=w8_swiglu_program(shape,20260930,1.f,W8Basis::SylvesterDH,0,0,false);
        const auto prefix=directory+"/legacy-"+std::to_string(i);
        const auto expected=read(prefix+".bin");
        check(p.mil==read(prefix+".mil"),"legacy MIL bytes changed");
        check(p.constants.size()==expected.size() &&
            std::equal(p.constants.begin(),p.constants.end(),reinterpret_cast<const uint8_t*>(expected.data())),
            "legacy rotation constants bytes changed");
        check(p.mil.find("native_")==std::string::npos,"legacy recipe gained native LoRA checks");
    }
}
void native_programs() {
    using contract::Variant;
    for(const bool tiny:{true,false}) {
        const contract::Geometry geometry=tiny?contract::Geometry{32,128,1024,512}:contract::Geometry{};
        const auto base=contract::Spec::make({.geometry=geometry});
        const auto emitted_base=native_lora_program(base);
        const GraphShape shape{Kind::SwiGLU,int(geometry.rows),int(geometry.hidden),int(geometry.ane_channels),0,0,false};
        const auto legacy_base=w8_swiglu_program(shape,20260930,1.f,W8Basis::SylvesterDH,0,0,false);
        check(emitted_base.mil==legacy_base.mil && emitted_base.constants==legacy_base.constants,
            "native Base provider differs from original Base");
        input_abi(emitted_base.mil,base);
        for(const auto variant:{Variant::HiddenReference,Variant::Rank128Candidate}) {
            const auto spec=contract::Spec::make({.variant=variant,.geometry=geometry,.rank=128,
                .adapter_count=1,.source_strength=1.f});
            const auto p=native_lora_program(spec);const auto& m=p.mil;
            check(p.headroom==1.f && uint32_t(p.packed_rows)==spec.packed_rows(),"native output spec mismatch");
            check(p.constants==legacy_base.constants,"native program embeds factors/checkpoint constants");
            input_abi(m,spec);
            final_guard_dependencies(m);
            check(m.find("add(x = gt, y = dg_t)")<m.find("exp(x = neg)"),"gate delta occurs after activation");
            check(m.find("add(x = us, y = du_safe)")<m.find("exp(x = neg)"),"up delta occurs after activation");
            for(const auto guard:{"native_gate_base_token","native_up_base_token","native_gate_delta_token",
                "native_up_delta_token","native_gate_corrected_token","native_up_corrected_token",
                "native_silu_token","native_hidden_token","native_rotated_hidden_token","native_hscale_token","native_down_normalized_token"})
                check(m.find(guard)!=std::string::npos,"critical finite boundary missing");
            check(occurrences(m,"less_equal(x = native_")>=11 &&
                m.find("less(x = native_ad_abs, y = fp16(65504))")==std::string::npos,
                "native carrier bound disagrees with inclusive FP16 CPU validator");
            check(m.find("native_hscale_token")<m.find(" carrier_ok =") &&
                occurrences(m,"native_hscale_token")>=3,"hscale overflow not aggregated into sentinel");
            check(m.find("native_hidden_token")<m.find(" hq = quantize("),"hidden finite checked after quantization");
            check(m.find("native_rotated_hidden_token")<m.find(" hq = quantize("),"rotation finite checked after quantization");
            check(m.find("native_checked_hscale = select(cond = native_all_ok, a = hscale, b = fp16(-1))")!=std::string::npos,
                "invalid carrier sentinel missing");
            const auto packed=m.find(" packed = concat(");
            check(m.find("native_down_normalized_token")<packed &&
                m.find("native_checked_hscale")<packed,"final down failure omitted from published sentinel");
            check(occurrences(m," y = tensor_to_tensor_buffer")==1 && m.ends_with("    } -> (y);\n}\n"),
                "native recipe must have one packed output covered by physical completion");
            if(variant==Variant::HiddenReference) {
                check(m.find(", native_checked_hscale, hsafe)")!=std::string::npos,"corrected hidden output missing");
                check(m.find(" ad_t =")==std::string::npos,"hidden reference gained rank input");
            } else {
                check(m.find("native_lowp0 = matmul(")<m.find(" h4 = reshape("),"rank dot reads rotated/quantized hidden");
                check(m.find("native_low_partial0_token")<packed && m.find("native_ad_token")<packed,
                    "rank/factor validation missing");
                const size_t tiles=(geometry.ane_channels+2047)/2048;
                check(occurrences(m,"native_lowp")>=tiles*3,"rank reduction tiles missing");
                check(occurrences(m," x = native_ad") == 0 || m.find("x = native_ad0, y = native_ah0")!=std::string::npos,
                    "rank operand orientation changed");
                check(m.find(", native_checked_hscale, native_low")!=std::string::npos,"rank output missing");
                check(m.find(", native_checked_hscale, hsafe)")==std::string::npos,"rank candidate still exports full hidden");
                if(!tiny) {
                    check(m.find("native_low_sum2048_token")<packed && m.find("native_low_sum4096_token")<packed,
                        "rank sum finite checks missing");
                    check(m.find("native_lows4096")!=std::string::npos,"rank summation order/tail wrong");
                }
                if(tiny) {
                    const auto mutate=[&](std::string_view before,std::string_view after) {
                        auto changed=m;const auto at=changed.find(before);
                        check(at!=std::string::npos,"negative ABI fixture target missing");
                        changed.replace(at,before.size(),after);return changed;
                    };
                    auto bad=mutate("tensor_buffer<int8,","tensor_buffer<fp16,");
                    rejects_mutation([&]{input_abi(bad,spec);},"input dtype mutation escaped ABI test");
                    bad=mutate("shape=[1, 1, 128, 32]","shape=[1, 1, 129, 32]");
                    rejects_mutation([&]{input_abi(bad,spec);},"input shape mutation escaped ABI test");
                    bad=mutate("strides=[8192, 8192, 64, 1]","strides=[8192, 8192, 65, 1]");
                    rejects_mutation([&]{input_abi(bad,spec);},"input pitch mutation escaped ABI test");
                    // Keep a token's declaration and first aggregate intact,
                    // but remove its actual edge from the final aggregate.
                    bad=m;const auto edge=bad.rfind(", y = native_hscale_token)");
                    check(edge!=std::string::npos,"negative guard fixture edge missing");
                    bad.replace(edge,std::string_view(", y = native_hscale_token)").size(),", y = native_hidden_token)");
                    rejects_mutation([&]{final_guard_dependencies(bad);},"unused finite token escaped dependency test");
                }
            }
        }
    }
}
}
int main(int argc,char**argv) {
    try {
        if(argc!=2)throw std::runtime_error("pass the explicitly owned legacy byte-fixture directory");
        legacy(argv[1]);native_programs();
        std::cout<<"native_ane_lora_mil_cpu=PASS checks="<<checks
                 <<" legacy_byte_cases=4 device_created=0 shader_compiled=0 models=0\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
