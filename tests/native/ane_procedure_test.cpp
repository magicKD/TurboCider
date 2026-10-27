#include "../../native/backends/private/ane_procedure.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace tc::ane::private_api;
    const std::vector<std::string> input{"w","x","x","w"}, output{"y","y"};
    const std::map<std::string,uint32_t> functions{{"rows32",1},{"rows64",0}};
    const std::vector<ProcedureSymbols> entries{{1,{2,3},{0}},{0,{1,0},{1}}};
    const auto table = checked_procedures(input,output,functions,entries);
    const auto &a = named_procedure(table,"rows32"), &b = named_procedure(table,"rows64");
    assert(a.id == 1 && b.id == 0 && a.input_indices == std::vector<uint32_t>({2,3}));
    assert(b.output_indices == std::vector<uint32_t>({1}) && a.inputs[0] == "x");
    auto denied = [](auto &&call) {try {call();} catch (const tc::ane::CapabilityError &) {return true;}return false;};
    assert(denied([&]{named_procedure(table,"missing");}));
    assert(denied([&]{checked_procedures(input,output,{},{});}));
    assert(denied([&]{checked_procedures(input,output,{{"a",0},{"b",0}},entries);}));
    assert(denied([&]{checked_procedures(input,output,{{"a",0},{"b",2}},entries);}));
    for (const auto &bad : {std::vector<ProcedureSymbols>{{1,{2,3},{0}}},
                          {{1,{2,3},{0}},{1,{1,0},{1}}},
                          {{1,{2,99},{0}},{0,{1,0},{1}}},
                          {{1,{2,2},{0}},{0,{1,0},{1}}},
                          {{1,{2,1},{0}},{0,{1,0},{1}}},
                          {{1,{2,3},{}},{0,{1,0},{1}}},
                          {{1,{2,3},{0,1}},{0,{1,0},{1}}}})
        assert(denied([&]{checked_procedures(input,output,functions,bad);}));
    std::cout << "PASS named procedures, permuted global symbol IDs, duplicate cross-function names and fail-closed metadata\n";
}
