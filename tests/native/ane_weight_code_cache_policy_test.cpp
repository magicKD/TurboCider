#include "../../native/backends/ane_weight_code_cache.hpp"
#include <iostream>
#include <thread>
#include <vector>

using namespace tc::ane::gpu;
void check(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
int main() {
    try {
        check(parse_weight_code_cache_bytes(nullptr)==0 && parse_weight_code_cache_bytes("0")==0 &&
              parse_weight_code_cache_bytes("1342177280")==uint64_t(1280)<<20 &&
              parse_weight_code_cache_bytes("2147483648")==weight_code_cache_max_bytes,"cache parser boundary");
        for(const char *bad:{"","-1","+1"," 1","1x","2147483649","999999999999999999999"}) {
            bool rejected=false;try {parse_weight_code_cache_bytes(bad);}catch(const std::invalid_argument &){rejected=true;}
            check(rejected,"invalid cache budget admitted");
        }
        auto plan=weight_code_cache_plan(5120,4096,16384);
        check(plan && plan->codes_bytes==20971520 && plan->scales_bytes==10240 &&
              plan->allocation_upper==20987904,"normalized code/scale allocation upper");
        auto native=weight_code_cache_plan(5120,4096,16384,true);
        check(native && native->scales_bytes==327680 && native->allocation_upper==21299200,
              "native surface allocation/pitch upper");
        check(parse_weight_code_cache_storage(nullptr)==WeightCodeCacheStorage::CompactCopy &&
              parse_weight_code_cache_storage("surface")==WeightCodeCacheStorage::NativeSurface,"cache storage policy");
        bool bad_storage=false;try{parse_weight_code_cache_storage("auto");}catch(const std::invalid_argument&){bad_storage=true;}
        check(bad_storage,"invalid cache storage policy admitted");
        check(!weight_code_cache_plan(0,4096,16384) && !weight_code_cache_plan(32769,4096,16384) &&
              !weight_code_cache_plan(5120,4096,1000),"invalid allocation plan admitted");
        WeightCodeCacheLedger ledger(1024);
        check(ledger.reserve(1024) && !ledger.reserve(1) && !ledger.reserve(0),"live lease cap violated");
        ledger.release(1024);
        std::vector<std::thread> workers;
        for(int i=0;i<8;++i)workers.emplace_back([&] {
            for(int j=0;j<10000;++j)if(ledger.reserve(128))ledger.release(128);
        });
        for(auto &worker:workers)worker.join();
        check(ledger.live()==0 && ledger.peak()==1024,"concurrent capacity accounting");
        std::cout<<"PASS weight code cache policy: parser, page upper, escaped capacity and concurrent leases\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
