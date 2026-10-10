#include "runtime/streaming/gguf_packed_bank.hpp"
#include "core/gguf_decode.hpp"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace tc;
using namespace tc::streaming;

namespace {
void read_exact(int fd,std::span<std::byte> target,uint64_t offset) {
    size_t done=0;
    while(done<target.size()) {
        const auto n=::pread(fd,target.data()+done,target.size()-done,off_t(offset+done));
        if(n<0 && errno==EINTR)continue;
        require(n>0,"real K source slice read failed");done+=size_t(n);
    }
}
double relative(const Tensor &a,const Tensor &b) {
    auto af=mx::astype(a,mx::float32),bf=mx::astype(b,mx::float32);
    return mx::sqrt(mx::sum(mx::square(af-bf))/mx::maximum(mx::sum(mx::square(bf)),Tensor(1e-20f))).item<float>();
}
}

int main(int argc,char **argv) {try {
    require(argc==3,"usage: qwen21-gguf-import-probe original-denoiser original-text-encoder");
    configure_streams();mx::set_cache_limit(0);std::cout<<std::setprecision(12);
    std::cout<<"{\"schema\":\"tc-qwen21-real-k-import-v1\",\"scope\":\"native full-source proof, bounded selected packed banks and real Metal projections; not model/performance/ANE qualification\",\"qualification_passed\":false,\"cases\":[";
    bool comma=false;
    const std::vector<std::vector<std::string>> selected{{
        "model.diffusion_model.transformer_blocks.0.attn.to_k.weight",
        "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
        "model.diffusion_model.transformer_blocks.27.attn.to_v.weight",
        "model.diffusion_model.img_in.weight",
        "model.diffusion_model.norm_out.linear.weight"},{
        "blk.0.attn_q.weight","blk.0.ffn_down.weight"}};
    for(int component=0;component<2;++component) {
        SourceFileIdentity file;file.logical_id="component";file.path=argv[component+1];
        auto lease=SourceLease::capture_verified({file});auto fd=lease->duplicate_fd("component");
        for(const auto &name:selected[component])for(auto dtype:{mx::float16,mx::bfloat16}) {
            MemoryLedger ledger(256ull<<20);Weights weights;
            GgufKImportOptions options;options.enabled=true;options.floating_dtype=dtype;
            options.include_tensor=[name](std::string_view key){return key==name;};
            GgufPackedBank bank(lease,"component",ledger,1ull<<20,true,0,6,options);
            const auto tensor=bank.directory().tensor(name);
            const auto start=Clock::now();bank.load(weights);
            const double load=std::chrono::duration<double>(Clock::now()-start).count();
            const auto metrics=bank.metrics();
            require(metrics.tensor_count==1 && metrics.source_read_bytes==tensor.bytes,
                    "component filter reread/skipped original tensor or published another source");
            const auto &type=gguf::type_info(tensor.type);
            const uint64_t rows=std::min<uint64_t>(8,tensor.rows()),columns=tensor.columns();
            const auto row_bytes=columns/type.elements*type.bytes;
            std::vector<std::byte> raw(rows*row_bytes);read_exact(fd.get(),raw,tensor.file_offset);
            std::vector<float> decoded(rows*columns);
            gguf::decode_cpu_into({raw,tensor.type,rows,columns},{0,rows,0,columns},
                {{reinterpret_cast<std::byte *>(decoded.data()),decoded.size()*4},gguf::DecodeDType::f32,columns*4,4});
            Tensor reference(decoded.data(),{int(rows),int(columns)},mx::float32);mx::eval(reference);
            if(comma)std::cout<<',';comma=true;
            std::cout<<"{\"component\":\""<<(component ? "text" : "denoiser")<<"\",\"name\":\""<<name<<"\",\"source_type\":"<<tensor.type
                <<",\"typed_recipe\":\""<<(dtype==mx::bfloat16 ? "bf16" : "fp16")<<"\",\"source_sha256\":\""<<lease->file("component").content_digest
                <<"\",\"import_seconds\":"<<load<<",\"source_tensor_bytes\":"<<tensor.bytes<<",\"managed_packed_capacity_bytes\":"<<metrics.packed_capacity_bytes
                <<",\"plan_digest\":\""<<metrics.plan_digest<<"\",\"projections\":[";
            bool projection_comma=false;
            for(int m:{1,33,145}) {
                auto x=mx::astype(mx::reshape(mx::sin(mx::arange(m*int(columns),mx::float32)*.013f)*.125f,{1,m,int(columns)}),dtype);
                mx::eval(x);
                auto oracle=mx::matmul(mx::astype(x,mx::float32),mx::transpose(reference));
                auto actual=linear(x,weights,name.substr(0,name.size()-7));mx::eval({oracle,actual});
                auto prefix=mx::slice(actual,{0,0,0},{1,m,int(rows)});const double error=relative(prefix,oracle);
                require(actual.shape()==mx::Shape{1,m,int(tensor.rows())} && actual.dtype()==dtype &&
                    mx::all(mx::isfinite(actual)).item<bool>() && std::isfinite(error) && error<=.05,
                    "real typed K projection disagrees with independent original-source F32 oracle");
                if(projection_comma)std::cout<<',';projection_comma=true;
                std::cout<<"{\"M\":"<<m<<",\"checked_output_rows\":"<<rows<<",\"relative_l2\":"<<error<<'}';
            }
            std::cout<<"]}";bank.check_unchanged();weights.clear();mx::synchronize();mx::clear_cache();
            require(!ledger.snapshot().storage_bytes && !ledger.snapshot().reserved_bytes,"real K packed bank leaked escaped claims");
        }
        lease->revalidate_after_drain();
    }
    std::cout<<"],\"passed_real_metal_projections\":42,\"default_route_changed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
