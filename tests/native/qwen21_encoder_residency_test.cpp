#include "../../native/models/qwen21/encoder_residency.hpp"
#include "../../native/backends/ane_backend.hpp"
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
    using namespace tc;
    require(argc==2,"temporary source fixture required");
    const auto file=std::filesystem::path(argv[1]);
    const auto original=qwen21::encoder_source_generation(file);
    require(original.bytes==7,"fixture size mismatch");
    const auto timestamp=std::filesystem::last_write_time(file);
    {std::ofstream stream(file,std::ios::binary|std::ios::trunc);stream<<"changed";}
    std::filesystem::last_write_time(file,timestamp);
    const auto replaced=qwen21::encoder_source_generation(file);
    require(replaced.bytes==original.bytes && replaced.identity!=original.identity,
        "ctime failed to invalidate same-size/mtime source replacement");
    auto alias=file.parent_path()/"alias";std::filesystem::create_symlink(file.filename(),alias);
    require(qwen21::encoder_source_generation(alias).identity==replaced.identity,"canonical source alias mismatch");
    constexpr uint64_t gib=1ull<<30;
    ane::MemoryObservation observation{true,64*gib,32*gib,24*gib,32*gib,8*gib};
    require(qwen21::admit_encoder_weights(observation,17*gib,0).allowed(),"healthy retention rejected");
    require(!qwen21::admit_encoder_weights(observation,21*gib,0).allowed(),"logical source bound bypassed");
    require(!qwen21::admit_encoder_weights(observation,17*gib,22*gib).allowed(),"future model growth ignored");
    require(!qwen21::admit_encoder_weights(observation,17*gib,UINT64_MAX).allowed(),"growth overflow ignored");
    observation.available=false;require(!qwen21::admit_encoder_weights(observation,17*gib,0).allowed(),"unknown memory admitted");
    observation.available=true;observation.physical_bytes=47*gib;
    require(!qwen21::admit_encoder_weights(observation,17*gib,0).allowed(),"small-memory retention admitted");
    setenv("TURBOCIDER_PRIVATE_ANE_CHANNELS","5120",1);
    require(ane::resolved_private_channel_count(12288,std::nullopt,3072)==3072 &&
        ane::private_channel_count(12288)==5120,"component override mutated/inherited global share");
    require(ane::resolved_private_channel_count(12288,std::nullopt,0)==0,"explicit row override lost");
    for(int channels:{-1,256,12288}) {
        bool rejected=false;
        try{ane::resolved_private_channel_count(12288,std::nullopt,channels);}catch(const std::exception&){rejected=true;}
        require(rejected,"invalid component channel override admitted");
    }
    bool rejected=false;
    try{ane::resolved_private_channel_count(12288,5120,3072);}catch(const std::exception&){rejected=true;}
    require(rejected,"explicit override silently replaced calibration");
    std::cout<<"PASS encoder source generation/admission and immutable per-component channel configuration\n";
}
