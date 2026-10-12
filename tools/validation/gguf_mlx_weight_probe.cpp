#include "../../native/backends/mlx.hpp"
#include "../../native/core/gguf.hpp"
#include "../../native/core/gguf_affine.hpp"
#include <iostream>
#include <cstring>

int main(int argc,char **argv) {
    try {
        if(argc!=3)return 2;
        tc::configure_streams();tc::mx::set_cache_limit(0);
        const std::string name(argv[2]);
        tc::Weights weights;weights.load_gguf_file(argv[1]);
        const auto &q=weights.at(name);
        int fd=::open(argv[1],O_RDONLY);struct stat st{};::fstat(fd,&st);
        const auto directory=tc::gguf::read_directory(fd,uint64_t(st.st_size));
        const auto &d=directory.tensor(name);
        std::vector<std::byte> raw(d.bytes);
        if(::pread(fd,raw.data(),raw.size(),off_t(d.file_offset))!=ssize_t(raw.size()))return 3;
        ::close(fd);
        if(tc::gguf::type_info(d.type).elements==1) {
            tc::mx::eval(q);
            std::cout<<"float_dtype="<<q.dtype()<<" raw_bytes_exact="
                <<(q.nbytes()==raw.size()&&!std::memcmp(q.data<char>(),raw.data(),raw.size()))<<'\n';
            return 0;
        }
        const auto prefix=name.substr(0,name.size()-7);
        const auto &s=weights.at(prefix+".scales"),&b=weights.at(prefix+".biases");
        const tc::gguf::PackedMatrix source{raw,d.type,d.rows(),d.columns()};
        const uint32_t bits=d.type==8?8:4;
        std::vector<std::byte> codes(d.elements*bits/8),scales(d.elements/32*2),biases(scales.size());
        tc::gguf::pack_native_affine(source,tc::gguf::AffinePart::codes,codes);
        tc::gguf::pack_native_affine(source,tc::gguf::AffinePart::scales,scales);
        tc::gguf::pack_native_affine(source,tc::gguf::AffinePart::biases,biases);
        tc::mx::eval({q,s,b});
        std::cout<<"dtype q/s/b="<<q.dtype()<<'/'<<s.dtype()<<'/'<<b.dtype()<<" codes_exact="
            <<(q.nbytes()==codes.size()&&!std::memcmp(q.data<char>(),codes.data(),codes.size()))
            <<" scales_exact="<<(s.nbytes()==scales.size()&&!std::memcmp(s.data<char>(),scales.data(),scales.size()))
            <<" biases_exact="<<(b.nbytes()==biases.size()&&!std::memcmp(b.data<char>(),biases.data(),biases.size()))<<'\n';
        std::vector<float> original(d.elements);
        tc::gguf::decode_cpu_into(source,{0,d.rows(),0,d.columns()},
            {{reinterpret_cast<std::byte *>(original.data()),original.size()*4},tc::gguf::DecodeDType::f32,d.columns()*4,4});
        auto reference=tc::Tensor(original.data(),{int(d.rows()),int(d.columns())},tc::mx::float32);
        auto actual=tc::mx::dequantize(q,tc::mx::astype(s,tc::mx::float32),tc::mx::astype(b,tc::mx::float32),32,bits,"affine",std::nullopt,tc::mx::float32);
        auto error=tc::mx::max(tc::mx::abs(reference-actual));
        std::cout<<"weight_max_abs="<<error.item<float>()<<'\n';
        return 0;
    } catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
