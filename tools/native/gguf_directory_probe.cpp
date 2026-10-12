#include "gguf_directory.hpp"
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>

int main(int argc,char **argv) {try {
    if(argc!=2 && argc!=3)throw std::invalid_argument("usage: gguf-directory-probe path [diagnostic-expected-final-bytes]");
    struct File {int fd;~File(){if(fd>=0)::close(fd);}} file{::open(argv[1],O_RDONLY|O_CLOEXEC)};
    struct stat st{};
    if(file.fd<0 || ::fstat(file.fd,&st)!=0 || !S_ISREG(st.st_mode) || st.st_size<0)
        throw std::invalid_argument("directory source must be a regular file");
    const uint64_t bytes=argc==3 ? std::stoull(argv[2]) : uint64_t(st.st_size);
    auto directory=tc::gguf::read_directory(file.fd,bytes,{},false);
    std::cout<<"{\"schema\":\"tc-gguf-directory-probe-v1\",\"scope\":\"bounded directory only; no payload hash or runtime support proof\",\"partial_file_diagnostic\":"
        <<(argc==3?"true":"false")<<",\"observed_file_bytes\":"<<st.st_size<<",\"expected_file_bytes\":"<<bytes
        <<",\"directory_end\":"<<directory.directory_end<<",\"metadata\":[";
    bool comma=false;
    for(const auto &entry:directory.metadata) {
        // Names in the model/architecture directory are diagnostic only;
        // omit large vocabulary arrays and arbitrary free-form string values.
        if(entry.type==9 || entry.key.find('"')!=std::string::npos)continue;
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"key\":\""<<entry.key<<"\",\"type\":"<<entry.type<<",\"unsigned\":"<<entry.unsigned_value<<'}';
    }
    std::cout<<"],\"tensors\":[";comma=false;
    for(const auto &tensor:directory.tensors) {
        if(tensor.name.find('"')!=std::string::npos || tensor.name.find('\\')!=std::string::npos)
            throw std::invalid_argument("unsupported diagnostic tensor name");
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"name\":\""<<tensor.name<<"\",\"type\":"<<tensor.type<<",\"bytes\":"<<tensor.bytes
            <<",\"file_offset\":"<<tensor.file_offset<<",\"shape\":[";
        bool dimension_comma=false;
        for(auto dim:tensor.logical_shape()) {if(dimension_comma)std::cout<<',';dimension_comma=true;std::cout<<dim;}
        std::cout<<"]}";
    }
    std::cout<<"]}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
