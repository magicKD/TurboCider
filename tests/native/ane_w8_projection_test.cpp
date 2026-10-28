#include "../../native/backends/ane_backend.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <cassert>
#include <iostream>
#include <regex>
#include <sstream>

size_t count(const std::string&s,const std::string&token){size_t total=0,at=0;while((at=s.find(token,at))!=std::string::npos){++total;at+=token.size();}return total;}
std::string projection_lines(const std::string &mil,const std::string &name){
    std::istringstream stream(mil);std::string line,result;
    const std::regex variable(" "+name+"[wxps][0-9]+q? = ");
    while(std::getline(stream,line))if(std::regex_search(line,variable))result+=line+"\n";
    assert(!result.empty());return result;
}
int main(){
    using namespace tc::ane;using namespace tc::ane::private_api;
    assert(parse_w8_projection(nullptr)==W8Projection::Descriptor);
    for(const auto *raw:{"descriptor","k4096","single","up-single","down-single"})assert(w8_projection_name(parse_w8_projection(raw))==raw);
    for(const auto *raw:{"","4096","0","SINGLE","single "," single","up_single","up-single ","DownSingle"}){
        bool bad=false;try{(void)parse_w8_projection(raw);}catch(const std::runtime_error&){bad=true;}assert(bad);
    }
    GraphShape shape{Kind::SwiGLU,1024,4096,6144,4096,512,false};
    const auto baseline=w8_swiglu_program(shape,20260930,1,W8Basis::SylvesterDH,0,0,false);
    for(auto policy:{W8Projection::Descriptor,W8Projection::K4096,W8Projection::Single,W8Projection::UpSingle,W8Projection::DownSingle}){
        const auto value=w8_swiglu_projection_program(shape,20260930,1,W8Basis::SylvesterDH,0,0,false,policy);
        assert(value.constants==baseline.constants&&value.packed_rows==baseline.packed_rows&&value.headroom==baseline.headroom);
        assert(count(value.mil," = matmul(")==(policy==W8Projection::Descriptor?7u:policy==W8Projection::K4096?4u:policy==W8Projection::Single?3u:5u));
        if(policy==W8Projection::Descriptor)assert(value.mil==baseline.mil);
        if(policy==W8Projection::K4096)assert(value.mil.find("dp4096 = matmul(")!=std::string::npos);
        if(policy==W8Projection::Single)assert(value.mil.find("slice_by_size")==std::string::npos&&value.mil.find("ds4096")==std::string::npos);
        if(policy!=W8Projection::Descriptor){
            validate_w8_projection_backend(policy,true,{BackendPreference::Private,true},"w8a8",false);
            for(int which=0;which<5;++which){bool bad=false;
                try{validate_w8_projection_backend(policy,which!=0,{which==1?BackendPreference::Public:BackendPreference::Private,which!=2},
                    which==3?"fp16":"w8a8",which==4);}catch(const std::runtime_error&){bad=true;}assert(bad);
            }
        }
    }
    for(auto policy:{W8Projection::K4096,W8Projection::Single,W8Projection::UpSingle,W8Projection::DownSingle}){
        auto lora=shape;lora.lora_inputs=true;bool bad=false;
        try{(void)w8_swiglu_projection_program(lora,20260930,1,W8Basis::SylvesterDH,0,0,false,policy);}catch(const CapabilityError&){bad=true;}assert(bad);
        bad=false;try{(void)w8_swiglu_projection_program({Kind::SwiGLU,33,512,512,256,512,false},0,1,W8Basis::ComfyH256,0,0,false,policy);}
            catch(const CapabilityError&){bad=true;}assert(bad);
    }
    assert(w8_projection_k_tile(W8Projection::Descriptor,1024,4096,false)==1024);
    assert(w8_projection_k_tile(W8Projection::Descriptor,4096,4096,false)==2048);
    assert(w8_projection_k_tile(W8Projection::Single,4096,6144,false)==6144);
    // Exact subgraph isolation, including small reductions where the old
    // identity slice must remain on the unmodified operand.
    for(const auto s:{shape,GraphShape{Kind::SwiGLU,33,128,512,4096,512,false}}){
        const auto old=w8_swiglu_program(s,20260930,1,W8Basis::SylvesterDH,0,0,false);
        const auto full=w8_swiglu_projection_program(s,20260930,1,W8Basis::SylvesterDH,0,0,false,W8Projection::Single);
        for(auto policy:{W8Projection::UpSingle,W8Projection::DownSingle}){
            const auto mixed=w8_swiglu_projection_program(s,20260930,1,W8Basis::SylvesterDH,0,0,false,policy);
            assert(mixed.constants==old.constants&&mixed.packed_rows==old.packed_rows);
            for(const auto *name:{"g","u","d"}){
                const bool down=std::string_view(name)=="d";
                const bool single=(policy==W8Projection::DownSingle)==down;
                assert(w8_projection_leaf(policy,down)==(single?W8Projection::Single:W8Projection::Descriptor));
                assert(projection_lines(mixed.mil,name)==projection_lines(single?full.mil:old.mil,name));
            }
        }
    }
    std::cout<<"PASS W8 projection policy: unchanged default bytes,3/4/5/7 MIL dots, exact isolated gate/up/down subgraphs, fixed ABI/constants and rejected unsupported routes\n";
}
