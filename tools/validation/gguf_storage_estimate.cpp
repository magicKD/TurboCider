// Header-only storage prediction through the production checked directory
// reader. No payload mapping/decode, MLX allocation or GPU access.
#include "gguf_directory.hpp"
#include <fcntl.h>
#include <iostream>
#include <map>
#include <sys/stat.h>

namespace {
uint64_t capacity(uint64_t bytes) {
    return tc::gguf::checked_add(bytes,16383)&~uint64_t(16383);
}
struct Summary {uint64_t tensors=0,elements=0,source=0,bf16=0,affine=0,aligned=0;bool native=true;};
}
int main(int argc,char **argv) {
    int fd=-1;
    try {
        tc::gguf::check(argc==2,"usage: gguf-storage-estimate checkpoint.gguf");
        fd=::open(argv[1],O_RDONLY|O_CLOEXEC);struct stat st{};
        tc::gguf::check(fd>=0 && ::fstat(fd,&st)==0 && st.st_size>=0,"cannot open GGUF checkpoint");
        const auto directory=tc::gguf::read_directory(fd,uint64_t(st.st_size));
        std::map<uint32_t,Summary> types;
        for (const auto &tensor:directory.tensors) {
            auto &s=types[tensor.type];++s.tensors;
            s.elements=tc::gguf::checked_add(s.elements,tensor.elements);
            s.source=tc::gguf::checked_add(s.source,tensor.bytes);
            s.bf16=tc::gguf::checked_add(s.bf16,tc::gguf::checked_mul(tensor.elements,2));
            if (tc::gguf::type_info(tensor.type).elements==1) {
                s.affine=tc::gguf::checked_add(s.affine,tensor.bytes);
                s.aligned=tc::gguf::checked_add(s.aligned,capacity(tensor.bytes));
            } else if (tensor.type==2 || tensor.type==3 || tensor.type==8) {
                const auto groups=tensor.elements/32;
                const auto codes=tc::gguf::checked_mul(groups,tensor.type==8 ? 32 : 16);
                const auto meta=tc::gguf::checked_mul(groups,2);
                s.affine=tc::gguf::checked_add(s.affine,tc::gguf::checked_add(codes,tc::gguf::checked_mul(meta,2)));
                s.aligned=tc::gguf::checked_add(s.aligned,tc::gguf::checked_add(capacity(codes),tc::gguf::checked_mul(capacity(meta),2)));
            } else s.native=false;
        }
        std::cout<<"{\"schema\":\"tc-gguf-storage-estimate-v1\",\"tensor_count\":"<<directory.tensors.size()
            <<",\"file_bytes\":"<<st.st_size<<",\"types\":[";
        bool first=true;
        for (const auto &[type,s]:types) {
            if (!first) std::cout<<',';first=false;
            std::cout<<"{\"type\":"<<type<<",\"name\":\""<<tc::gguf::type_info(type).name
                <<"\",\"tensor_count\":"<<s.tensors<<",\"elements\":"<<s.elements
                <<",\"source_payload_bytes\":"<<s.source<<",\"logical_bf16_bytes\":"<<s.bf16
                <<",\"native_affine_payload_bytes\":";
            if (s.native) std::cout<<s.affine;else std::cout<<"null";
            std::cout<<",\"native_affine_capacity_upper_bytes\":";
            if (s.native) std::cout<<s.aligned;else std::cout<<"null";
            std::cout<<'}';
        }
        std::cout<<"]}\n";::close(fd);return 0;
    } catch (const std::exception &error) {
        if (fd>=0) ::close(fd);std::cerr<<error.what()<<'\n';return 1;
    }
}
