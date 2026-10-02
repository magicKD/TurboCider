#include "runtime/streaming/gguf_gpu_affine.hpp"
#include "core/gguf_affine.hpp"
#include <cstring>
#include <iostream>

using namespace tc;
void insist(bool value,const char *reason) { if (!value) throw std::runtime_error(reason); }
int main() {
    try {
        configure_streams();mx::set_cache_limit(0);
        for (uint32_t type:{2u,3u,8u}) for (uint64_t rows:{1u,7u,33u}) for (uint64_t cols:{32u,64u,256u}) {
            const auto &info=gguf::type_info(type);const uint64_t groups=rows*cols/32;
            std::vector<uint8_t> raw(groups*info.bytes);
            for (uint64_t g=0;g<groups;++g) {
                const float scales[]={.125f,-.0625f,0.f,-0.f,0.000000059604644775390625f};
                const auto scale=gguf::float_to_fp16_rne(scales[g%5]);
                std::memcpy(raw.data()+g*info.bytes,&scale,2);
                const size_t offset=type==3 ? 4 : 2;
                if (type==3) {const auto bias=gguf::float_to_fp16_rne(-.25f);std::memcpy(raw.data()+g*info.bytes+2,&bias,2);}
                for (size_t i=offset;i<info.bytes;++i) raw[g*info.bytes+i]=uint8_t(g*13+i*37);
            }
            MemoryLedger ledger(1ull<<20);
            Tensor source(raw.data(),{int(rows),int(cols/32*info.bytes)},mx::uint8);
            std::optional<Tensor> escaped;
            {
                auto output=streaming::gguf_gpu_affine(source,type,rows,cols,ledger,41);
                for (size_t part=0;part<3;++part) {
                    std::vector<std::byte> expected(output[part].nbytes());
                    gguf::pack_native_affine({{reinterpret_cast<const std::byte *>(raw.data()),raw.size()},type,rows,cols},
                        gguf::AffinePart(part),expected);
                    insist(std::memcmp(output[part].data<std::byte>(),expected.data(),expected.size())==0,
                           "GPU raw affine field differs from CPU oracle");
                }
                if (ledger.snapshot().storage_count!=3 || ledger.snapshot().reserved_bytes!=0)
                    std::cerr<<"claims="<<ledger.snapshot().storage_count<<" reserved="<<ledger.snapshot().reserved_bytes<<" bytes="<<ledger.snapshot().storage_bytes<<'\n';
                insist(ledger.snapshot().storage_count==3 && ledger.snapshot().reserved_bytes==0,"GPU output admission/claims mismatch");
                escaped=output[0];
            }
            insist(ledger.snapshot().storage_count==1,"escaped GPU output lost its claim");
            auto lazy=mx::sum(mx::astype(*escaped,mx::float32));escaped.reset();
            insist(ledger.snapshot().storage_count==1,"lazy GPU reader lost claim");
            mx::eval(lazy);lazy=Tensor(0.f);mx::synchronize();mx::clear_cache();
            insist(ledger.snapshot().storage_bytes==0 && ledger.snapshot().reserved_bytes==0,"GPU affine claim leak");
            for (const char *fault: {"budget","type","columns","metadata"}) {
                MemoryLedger bad(std::strcmp(fault,"budget")==0 ? 1 : 1ull<<20);bool rejected=false;
                try { (void)streaming::gguf_gpu_affine(source,std::strcmp(fault,"type")==0 ? 12 : type,
                    rows,std::strcmp(fault,"columns")==0 ? cols-1 : std::strcmp(fault,"metadata")==0 ? cols*2 : cols,bad,42); }
                catch (const std::exception &) {rejected=true;}
                insist(rejected && bad.snapshot().storage_bytes==0 && bad.snapshot().reserved_bytes==0,"bad GPU plan admitted/leaked");
            }
        }
        for (uint16_t bits:{uint16_t(0x7c00),uint16_t(0x6000)}) {
            std::vector<uint8_t> raw(34,0);std::memcpy(raw.data(),&bits,2);
            Tensor source(raw.data(),{1,34},mx::uint8);MemoryLedger ledger(1ull<<20);bool rejected=false;
            try {(void)streaming::gguf_gpu_affine(source,8,1,32,ledger,43);} catch (const std::exception &) {rejected=true;}
            insist(rejected && !ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"GPU nonfinite/consumer overflow accepted or leaked");
        }
        {
            MemoryLedger ledger(1ull<<20);
            for (bool fail:{true,false}) {
                std::vector<streaming::GgufGpuAffinePending> pending;
                for (uint32_t type:{8u,2u,3u}) {
                    const auto &info=gguf::type_info(type);std::vector<uint8_t> bytes(info.bytes,0);
                    uint16_t scale=gguf::float_to_fp16_rne(.125f);
                    if (fail && type==2) scale=0x7c00;
                    std::memcpy(bytes.data(),&scale,2);
                    Tensor raw(bytes.data(),{1,int(info.bytes)},mx::uint8);
                    pending.push_back(streaming::prepare_gguf_gpu_affine(raw,type,1,32,ledger,55));
                }
                bool rejected=false;
                try {auto output=finish_gguf_gpu_affine(pending);insist(output.size()==3,"GPU batch count changed");}
                catch (const std::invalid_argument &) {rejected=true;}
                catch (const std::runtime_error &) {rejected=true;}
                insist(rejected==fail,"GPU batch failure/recovery verdict mismatch");
                pending.clear();mx::synchronize();mx::clear_cache();
                insist(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"GPU batch rollback leaked");
            }
        }
        {
            mx::set_cache_limit(512ull<<20);
            auto large=mx::zeros({1<<20},mx::float32);mx::eval(large);large=Tensor(0.f);mx::synchronize();
            std::vector<uint8_t> raw(34,0);const uint16_t scale=gguf::float_to_fp16_rne(.125f);std::memcpy(raw.data(),&scale,2);
            Tensor source(raw.data(),{1,34},mx::uint8);MemoryLedger ledger(65536);
            {auto output=streaming::gguf_gpu_affine(source,8,1,32,ledger,61);}
            mx::synchronize();
            insist(!ledger.snapshot().storage_bytes,"cached oversized backing escaped GPU capacity contract");
            const auto old=mx::set_cache_limit(0);insist(old==(512ull<<20),"GPU exact allocation did not restore cache hint");mx::clear_cache();
        }
        std::cout<<"PASS raw GPU affine: 27 geometries, CPU fields exact, admission, escaped/lazy claims, invalid input\n";
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
