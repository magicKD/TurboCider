#include "../../native/backends/private/ane_mil.hpp"
#include "../../tools/native/private_ane_qdq_share.hpp"
#include <iostream>
#include <regex>

using namespace tc::ane;
int main() {
    try {
        for(int tile:{128,256,1024})for(bool lora:{false,true})for(auto basis:{W8Basis::SylvesterDH,W8Basis::ComfyH256}) {
            auto graph=private_api::w8_swiglu_program({Kind::SwiGLU,33,512,1024,tile,512,lora},
                basis==W8Basis::ComfyH256?0:20260930,1.f,basis);
            const auto shared=research::share_gate_up_qdq(graph.mil);
            if(shared.find("ux0")!=std::string::npos || shared.find("y = gx0)")==std::string::npos ||
               shared.find("x = uw0, y = gx0)")==std::string::npos || shared.find("dx0 = dequantize")==std::string::npos)
                throw std::runtime_error("shared QDQ did not keep gate/up operands and down QDQ");
            const std::regex matmul("matmul\\(");
            if(std::distance(std::sregex_iterator(shared.begin(),shared.end(),matmul),std::sregex_iterator{})!=
               std::distance(std::sregex_iterator(graph.mil.begin(),graph.mil.end(),matmul),std::sregex_iterator{}))
                throw std::runtime_error("shared rewrite changed MatMul count");
            auto changed=graph.mil;
            auto at=changed.find("ux0q = slice_by_size(x = x_t");
            changed.replace(at,std::string("ux0q = slice_by_size(x = x_t").size(),"ux0q = slice_by_size(x = different");
            bool rejected=false;try{research::share_gate_up_qdq(changed);}catch(const std::exception&){rejected=true;}
            if(!rejected)throw std::runtime_error("nonidentical up activation accepted by shared rewrite");
        }
        std::cout<<"PASS 12 shared-QDQ graph controls: same slice/dequantize, unchanged MatMul count, base/LoRA/Sylvester/Comfy, mismatch rejection\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
