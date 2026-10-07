#include "../../native/backends/ane_public_w8.hpp"
#include "../../native/backends/ane_gpu.hpp"
#import <Metal/Metal.h>
#include <cstring>
#include <iostream>
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    try {
        tc::ane::PublicW8Graph graph(argv[1],512ull<<20);
        std::string error;if(!graph.self_test(error))throw std::runtime_error(error);
        std::cout<<"PASS Public W8 GPU-stage/CoreML/FP32-restore shared executor self-test, slots="<<graph.slot_bytes()
                 <<" recipe="<<graph.weight_recipe()<<"; actual placement/arithmetic/model qualification unknown\n";
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
