#pragma once

#include <cmath>
#include <cctype>
#include <locale>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tc::lora_metadata {
// Deliberately bounded metadata reader. This does not alter legacy loading or
// interpret PEFT scaling: it only proves equivalence to the tensor-alpha ABI.
namespace detail {
struct Value {
    enum Kind { Null, Boolean, Number, String, Object, Array } kind=Null;
    bool boolean=false;
    double number=0;
    std::string string;
    std::map<std::string,Value> object;
    std::vector<Value> array;
};
class Parser {
    std::string_view source_; size_t cursor_=0,nodes_=0;
    [[noreturn]] void bad() const { throw std::invalid_argument("unsupported or invalid LoRA metadata JSON"); }
    void space() { while(cursor_<source_.size() && (source_[cursor_]==' ' || source_[cursor_]=='\n' || source_[cursor_]=='\r' || source_[cursor_]=='\t')) ++cursor_; }
    bool take(char c) { space(); if(cursor_<source_.size() && source_[cursor_]==c) { ++cursor_; return true; } return false; }
    std::string string() {
        if(!take('"')) bad(); std::string result;
        while(cursor_<source_.size()) {
            auto c=source_[cursor_++]; if(c=='"') return result;
            if(static_cast<unsigned char>(c)<0x20) bad();
            if(c=='\\') {
                if(cursor_==source_.size()) bad(); c=source_[cursor_++];
                if(c=='"' || c=='\\' || c=='/') result+=c;
                else if(c=='b') result+='\b'; else if(c=='f') result+='\f';
                else if(c=='n') result+='\n'; else if(c=='r') result+='\r'; else if(c=='t') result+='\t';
                else if(c=='u') {
                    unsigned value=0;
                    for(int i=0;i<4;++i) {
                        if(cursor_==source_.size()) bad(); const char h=source_[cursor_++];
                        unsigned digit=h>='0'&&h<='9'?unsigned(h-'0'):h>='a'&&h<='f'?unsigned(h-'a'+10):h>='A'&&h<='F'?unsigned(h-'A'+10):16;
                        if(digit==16) bad(); value=value*16+digit;
                    }
                    // Native policy keys/values are ASCII. Other metadata can
                    // still be used by the unmodified legacy route.
                    if(value>0x7f) bad(); result+=char(value);
                } else bad();
            } else result+=c;
        }
        bad();
    }
    Value value(unsigned depth) {
        if(depth>16 || ++nodes_>4096) bad(); space(); if(cursor_==source_.size()) bad();
        Value out;
        if(source_[cursor_]=='"') { out.kind=Value::String; out.string=string(); return out; }
        if(take('{')) {
            out.kind=Value::Object; if(take('}')) return out;
            do { auto key=string(); if(!take(':')) bad(); auto child=value(depth+1);
                if(!out.object.emplace(std::move(key),std::move(child)).second) bad();
                if(take('}')) return out;
            } while(take(',')); bad();
        }
        if(take('[')) {
            out.kind=Value::Array; if(take(']')) return out;
            do { out.array.push_back(value(depth+1)); if(take(']')) return out; } while(take(',')); bad();
        }
        for(auto literal:{std::string_view("true"),std::string_view("false"),std::string_view("null")}) {
            if(source_.substr(cursor_,literal.size())==literal) {
                cursor_+=literal.size(); out.kind=literal=="null"?Value::Null:Value::Boolean;
                out.boolean=literal=="true"; return out;
            }
        }
        const size_t start=cursor_; if(source_[cursor_]=='-') ++cursor_;
        if(cursor_==source_.size()) bad();
        auto digit=[&] { return cursor_<source_.size() && source_[cursor_]>='0' && source_[cursor_]<='9'; };
        if(source_[cursor_]=='0') ++cursor_;
        else { if(!digit()) bad(); while(digit()) ++cursor_; }
        if(cursor_<source_.size() && source_[cursor_]=='.') { ++cursor_; if(!digit()) bad(); while(digit()) ++cursor_; }
        if(cursor_<source_.size() && (source_[cursor_]=='e' || source_[cursor_]=='E')) {
            ++cursor_; if(cursor_<source_.size() && (source_[cursor_]=='+' || source_[cursor_]=='-')) ++cursor_;
            if(!digit()) bad(); while(digit()) ++cursor_;
        }
        std::istringstream stream(std::string(source_.substr(start,cursor_-start))); stream.imbue(std::locale::classic());
        out.kind=Value::Number; stream>>out.number; if(!stream || !stream.eof() || !std::isfinite(out.number)) bad(); return out;
    }
public:
    explicit Parser(std::string_view source):source_(source) { if(source.size()>65536) bad(); }
    Value parse() { auto result=value(0); space(); if(cursor_!=source_.size()) bad(); return result; }
};
} // namespace detail

struct Policy {
    std::optional<double> rank,alpha;
    std::string rejection;
    std::string rejection_for(int actual_rank,std::optional<float> tensor_alpha) const {
        if(!rejection.empty()) return rejection;
        if(rank && *rank!=actual_rank) return "LoRA PEFT rank metadata differs from tensor rank";
        if(alpha && *alpha!=double(tensor_alpha.value_or(float(actual_rank))))
            return "LoRA PEFT alpha metadata is not equivalent to tensor-alpha/default scaling";
        return {};
    }
};
inline Policy inspect(const std::unordered_map<std::string,std::string>& metadata) {
    Policy policy;
    try {
        for(const auto& [key,value]:metadata) {
            if(key=="format") { if(value!="pt") throw std::invalid_argument("unsupported LoRA metadata format"); continue; }
            if(key!="lora_adapter_metadata") throw std::invalid_argument("unrecognized native LoRA metadata: "+key);
            const auto object=detail::Parser(value).parse();
            using V=detail::Value;
            if(object.kind!=V::Object) throw std::invalid_argument("LoRA metadata must be an object");
            std::map<std::string,bool> seen;
            for(const auto& [raw,item]:object.object) {
                const auto name=raw.starts_with("transformer.")?raw.substr(12):raw;
                if(!seen.emplace(name,true).second) throw std::invalid_argument("duplicate normalized LoRA metadata key");
                bool accepted=false;
                if(name=="r" || name=="lora_alpha") {
                    accepted=item.kind==V::Number && (name!="r" || (item.number>0 && item.number<=32768 && std::floor(item.number)==item.number));
                    if(accepted) (name=="r"?policy.rank:policy.alpha)=item.number;
                } else if(name=="rank_pattern" || name=="alpha_pattern") accepted=item.kind==V::Object && item.object.empty();
                else if(name=="use_rslora" || name=="use_dora" || name=="fan_in_fan_out") accepted=item.kind==V::Boolean && !item.boolean;
                else if(name=="bias") accepted=item.kind==V::String && item.string=="none";
                else if(name=="peft_type") accepted=item.kind==V::String && item.string=="LORA";
                else if(name=="lora_dropout") accepted=item.kind==V::Number && item.number>=0 && item.number<1;
                else if(name=="init_lora_weights" || name=="inference_mode") accepted=item.kind==V::Boolean;
                else if(name=="modules_to_save") accepted=item.kind==V::Null;
                else if(name=="target_modules") {
                    accepted=item.kind==V::Array;
                    for(const auto& target:item.array) accepted=accepted && target.kind==V::String;
                }
                if(!accepted) throw std::invalid_argument("unsupported native LoRA metadata setting: "+raw);
            }
        }
    } catch(const std::exception& error) { policy.rejection=error.what(); }
    return policy;
}
} // namespace tc::lora_metadata
