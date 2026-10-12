#pragma once

// Research-only exact graph rewrite. No inference default or runtime flag.
// Gate/up use the same x_t K slices. Keep dequantize adjacent to each W8
// weight and preserve every MatMul, sum, scale, activation and LoRA boundary.
#include <regex>
#include <sstream>
#include <string>
#include <stdexcept>
#include <map>

namespace tc::ane::research {
inline std::string share_gate_up_qdq(const std::string &mil) {
    const std::regex node(R"(^([[:space:]]*tensor<[^=]+> )([gu]x[0-9]+q?) = (.*);$)");
    std::map<std::string,std::string> gate_nodes;
    std::map<std::string,std::string> substitutions;
    std::stringstream input(mil);
    std::string line,result;
    size_t removed=0;
    while(std::getline(input,line)) {
        std::smatch match;
        if(std::regex_match(line,match,node)) {
            const std::string name=match[2];
            if(name[0]=='g')gate_nodes.emplace(name,match[1].str()+match[3].str());
            else {
                auto renamed=name;renamed[0]='g';
                auto expression=match[3].str();
                for(const auto &[old,wanted]:substitutions)
                    expression=std::regex_replace(expression,std::regex("\\b"+old+"\\b"),wanted);
                const auto found=gate_nodes.find(renamed);
                if(found==gate_nodes.end() || found->second!=match[1].str()+expression)
                    throw std::runtime_error("gate/up QDQ declarations differ; no shared rewrite");
                substitutions.emplace(name,renamed);++removed;continue;
            }
        }
        for(const auto &[old,wanted]:substitutions)
            line=std::regex_replace(line,std::regex("\\b"+old+"\\b"),wanted);
        result+=line+'\n';
    }
    if(gate_nodes.empty() || removed!=gate_nodes.size() || substitutions.size()!=removed)
        throw std::runtime_error("shared QDQ requires every paired gate/up slice/dequantize");
    return result;
}
} // namespace tc::ane::research
