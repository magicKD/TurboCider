#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include "../../native/backends/dense_gpu_lora_b.hpp"
#include "qwen_lora_upload.hpp"
#include <iomanip>
#include <iostream>

using namespace tc;
using namespace tc::ane;
DeviceMatrixView matrix(const Tensor &a) {
    require(a.ndim()==3 || a.ndim()==2,"probe source rank geometry");
    return {const_cast<void *>(a.buffer().ptr()),a.buffer_size(),size_t(a.offset()),a.shape(-2),a.shape(-1),
        size_t(a.strides(-2))*a.itemsize(),a.dtype()==mx::float32?DType::FP32:DType::BF16,
        std::make_shared<Tensor>(a),a.data_shared_ptr()};
}
int main(int argc,char **argv){@autoreleasepool {try {
    require(argc==2,"usage: qwen-lora-upload-probe original-local-adapter");configure_streams();
    Weights adapter;adapter.load_file(argv[1],"transformer.transformer_blocks.0.img_mlp.");adapter.materialize();
    const auto &ag=adapter.at("gate_layer.lora_A.weight"),&bg=adapter.at("gate_layer.lora_B.weight"),
               &au=adapter.at("proj.lora_A.weight"),&bu=adapter.at("proj.lora_B.weight");
    require(ag.shape()==mx::Shape{256,4096} && au.shape()==ag.shape() && bg.shape()==mx::Shape{12288,256} && bu.shape()==bg.shape() &&
            ag.dtype()==mx::bfloat16 && au.dtype()==mx::bfloat16 && bg.dtype()==mx::bfloat16 && bu.dtype()==mx::bfloat16,
            "original layer0 BF16 r256 gate/up source mismatch");
    gpu::Device device(false,false);
    using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;
    Fn ranks=mx::compile([&](const std::vector<Tensor> &v){return std::vector<Tensor>{
        dense_gpu::projection_range(v[0],ag,0,256,0,4096,16,true,64),
        dense_gpu::projection_range(v[0],au,0,256,0,4096,16,true,64)};});
    std::vector<std::unique_ptr<research::LoraUpload>> producers;
    for (int bm:{16,32})for(bool swapped:{false,true})
        producers.push_back(std::make_unique<research::LoraUpload>(256,DType::BF16,2112,swapped,bm,128));
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-lora-rank-upload-v1\",\"scope\":\"real layer0 original BF16 gate/up A/B, synthetic input, scale0.75; every timed invocation recomputes joint F32 ranks, both B projections/delta boundaries, padding and joined FP16 correction IOSurface uploads; no FFN/ANE execution, read disk, model/visual or physical overlap proof\",\"cases\":[";
    bool comma=false;
    for(int m:{1024,2096,3144})for(int n:{5120,7168}) {
        const int chunk=m==2096?2112:1056,padded=(m+chunk-1)/chunk*chunk,first=12288-n;
        auto x=mx::reshape(mx::astype(mx::sin(mx::arange(m*4096,mx::float32)*.001f)*.125f,mx::bfloat16),{1,m,4096});mx::eval(x);
        auto control=mx::compile([&,n,first,padded,m](const std::vector<Tensor> &v){
            auto low=ranks(v);std::vector<Tensor> result;
            for(size_t i=0;i<2;++i) {
                auto delta=dense_gpu::lora_b_epilogue(low[i],i?bu:bg,first,first+n,.75f,mx::bfloat16);
                result.push_back(mx::contiguous(padded==m?delta:mx::concatenate({delta,mx::zeros({1,padded-m,n},mx::bfloat16)},1)));
            }
            return result;
        });
        gpu::Surface gate(device,n,chunk,gpu::Element::FP16),up(device,n,chunk,gpu::Element::FP16);
        auto expected=control({x});mx::eval(expected);
        auto verify=[&](int begin) {
            for(int c=0;c<n;++c)for(int r=0;r<chunk;++r)for(int half=0;half<2;++half) {
                const auto &surface=half?up:gate;
                const float actual=float(reinterpret_cast<const _Float16 *>(static_cast<const char *>(surface.data())+size_t(c)*surface.pitch())[r]);
                const auto bits=expected[half].data<mx::bfloat16_t>()[(begin+r)*n+c];
                const float reference=float(_Float16(float(bits)));
                require(std::isfinite(actual) && actual==reference,"rank-to-surface real B typed oracle mismatch");
            }
        };
        auto run=[&](size_t recipe,bool check) {
            if(recipe==0) {
                auto delta=control({x});mx::eval(delta);
                for(int begin=0;begin<padded;begin+=chunk) {
                    auto io=device.prepare_upload({{matrix(delta[0]),gate,begin},{matrix(delta[1]),up,begin}});
                    io.submit();const auto done=io.finish();require(done.ok && !io.validation_flags(),"control correction upload failed");
                    if(check)verify(begin);
                }
            } else {
                auto low=ranks({x});mx::eval(low);
                for(int begin=0;begin<padded;begin+=chunk) {
                    uint32_t flags=0;
                    auto done=producers[recipe-1]->run({
                        {matrix(low[0]),matrix(bg),gate,begin,first,.75f,DType::BF16},
                        {matrix(low[1]),matrix(bu),up,begin,first,.75f,DType::BF16}},flags);
                    require(done.ok && !flags,done.error);if(check)verify(begin);
                }
            }
        };
        std::vector<std::vector<double>> samples(5);
        for(size_t i=0;i<samples.size();++i){run(i,true);for(int warm=0;warm<3;++warm)run(i,false);}
        for(int hot=0;hot<15;++hot)for(size_t visit=0;visit<samples.size();++visit) {
            const auto recipe=(hot+visit)%samples.size();const auto start=Clock::now();run(recipe,false);
            samples[recipe].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"M\":"<<m<<",\"channels\":"<<n<<",\"first\":"<<first<<",\"chunk\":"<<chunk
                 <<",\"padded\":"<<padded<<",\"candidate_rank_scratch_bytes\":"<<producers[0]->scratch_bytes()
                 <<",\"control_global_correction_output_logical_bytes\":"<<uint64_t(padded)*n*4<<",\"typed_surface_exact\":true,\"recipes\":[";
        for(size_t i=0;i<samples.size();++i) {
            if(i)std::cout<<',';
            std::cout<<"{\"name\":\""<<(i?"direct_rank_B_surface":"joint_B_then_upload")<<"\",\"BM\":"<<(i?i<=2?16:32:0)
                     <<",\"BN\":"<<(i?128:0)<<",\"swapped\":"<<(i && i%2==0?"true":"false")<<",\"seconds\":[";
            for(size_t j=0;j<samples[i].size();++j){if(j)std::cout<<',';std::cout<<samples[i][j];}std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}}
