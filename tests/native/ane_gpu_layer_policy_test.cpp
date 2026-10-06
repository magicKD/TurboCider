#include "../../native/backends/ane_gpu_layer_policy.hpp"
#include <cassert>

int main() {
    using namespace tc::ane;
    assert(parse_gpu_layers(nullptr,32).empty());
    assert(parse_gpu_layers("5,2,0",32)==(std::vector<int>{0,2,5}));
    assert(parse_gpu_layers("31",32)==(std::vector<int>{31}));
    for(const char *raw:{"","2,","-1","32","2,2","2,,3"," 2","2.0","9999","none"}) {
        bool rejected=false;try{(void)parse_gpu_layers(raw,32);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    for(const auto &values:{std::vector<int>{-1},std::vector<int>{128},std::vector<int>{2,2}}) {
        bool rejected=false;try{(void)normalized_gpu_layers(values);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
}
