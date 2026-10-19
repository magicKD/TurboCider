#pragma once
// CPU-only normalization of driver procedure metadata. Global symbol indices
// are not local input ordinals, and a function's ID need not be zero.
#include "../ane_runtime.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace tc::ane::private_api {
struct ProcedureSymbols {
    uint32_t id = 0;
    std::vector<uint32_t> inputs, outputs;
};
struct Procedure {
    std::string function;
    uint32_t id = 0;
    std::vector<std::string> inputs, outputs;
    std::vector<uint32_t> input_indices, output_indices;
};

inline std::vector<Procedure> checked_procedures(
    const std::vector<std::string> &input_symbols,
    const std::vector<std::string> &output_symbols,
    const std::map<std::string,uint32_t> &functions,
    const std::vector<ProcedureSymbols> &entries) {
    auto require = [](bool ok) {
        if (!ok) throw CapabilityError("private ANE invalid function/procedure symbol metadata");
    };
    require(!functions.empty() && functions.size() <= 16 && entries.size() == functions.size());
    require(!input_symbols.empty() && input_symbols.size() <= 8192 &&
            !output_symbols.empty() && output_symbols.size() <= 8192);
    for (const auto *symbols : {&input_symbols,&output_symbols})
        for (const auto &name : *symbols) require(!name.empty() && name.size() <= 1024);
    std::vector<Procedure> result(functions.size());
    std::set<uint32_t> ids, described;
    for (const auto &[name,id] : functions) {
        require(!name.empty() && name.size() <= 128 && id < result.size() && ids.insert(id).second);
        result[id].function = name; result[id].id = id;
    }
    for (const auto &entry : entries) {
        require(entry.id < result.size() && described.insert(entry.id).second);
        auto &p = result[entry.id];
        auto bind = [&](const auto &indices,const auto &symbols,auto &names,auto &target) {
            require(!indices.empty() && indices.size() <= 128);
            std::set<uint32_t> unique_indices; std::set<std::string> unique_names;
            for (auto index : indices) {
                require(index < symbols.size() && unique_indices.insert(index).second &&
                        unique_names.insert(symbols[index]).second);
                names.push_back(symbols[index]); target.push_back(index);
            }
        };
        bind(entry.inputs,input_symbols,p.inputs,p.input_indices);
        bind(entry.outputs,output_symbols,p.outputs,p.output_indices);
        // Existing shared completion contract attaches to one output. A
        // multi-output procedure needs a separate all-output completion proof.
        require(p.outputs.size() == 1);
    }
    return result;
}

inline const Procedure &named_procedure(const std::vector<Procedure> &procedures,std::string_view name) {
    auto p = std::find_if(procedures.begin(),procedures.end(),[&](const auto &x){return x.function == name;});
    if (p == procedures.end()) throw CapabilityError("private ANE function is absent: " + std::string(name));
    return *p;
}
} // namespace tc::ane::private_api
