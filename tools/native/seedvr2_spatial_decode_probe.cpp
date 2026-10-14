#include "../../native/models/seedvr2/vae.hpp"
#include "../../native/models/seedvr2/memory.hpp"
#include <mach/mach.h>
#include <thread>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
namespace mx=mlx::core;
using tc::Tensor;
namespace {
struct Sampler {
    std::atomic<bool> stop{false};std::atomic<uint64_t> peak{0};std::thread worker;
    Sampler():worker([&]{while(!stop){task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
        if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS){
            auto old=peak.load();while(old<info.phys_footprint && !peak.compare_exchange_weak(old,info.phys_footprint)){};}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));}}){}
    ~Sampler(){stop=true;if(worker.joinable())worker.join();}
};
}
int main(int argc,char **argv){
    try {
        tc::require(argc==7,"usage: spatial-decode-probe MODEL FIXTURE OUTPUT baseline|spatial THRESHOLD_BYTES TILE_ROWS");
        const std::filesystem::path model(argv[1]),output(argv[3]);const std::string route(argv[4]);
        tc::require(!std::filesystem::exists(output) && (route=="baseline" || route=="spatial"),"fresh output and known route required");
        std::filesystem::create_directories(output);mx::set_default_device(mx::Device::gpu);
        auto tensors=mx::load_safetensors(argv[2]).first;auto latent=tensors.at("latent");
        auto reference=tensors.at(tensors.count("decoded")?"decoded":"pixels");
        const int frames=reference.shape(2),h=reference.shape(3),w=reference.shape(4);
        tc::require(reference.dtype()==mx::float16 && reference.shape(0)==1 && reference.shape(1)==3,"saved FP16 full decoder reference required");
        auto flat=mx::contiguous(mx::transpose(reference,{0,2,3,4,1}));mx::eval(flat);
        std::vector<uint16_t> saved(flat.size());std::memcpy(saved.data(),flat.data<uint16_t>(),flat.nbytes());
        tensors.clear();flat=Tensor(0);reference=Tensor(0);mx::synchronize();mx::clear_cache();
        std::atomic<bool> cancelled{false};tc::seedvr2::AllocatorCacheScope allocator(1ull<<30);
        tc::seedvr2::VAE vae;vae.load(model/"vae/seedvr2_ema_vae_fp16.safetensors");
        tc::SeedVR2VAEConfig config;config.history="frames";vae.configure(config);mx::eval(latent);
        tc::seedvr2::SpatialTileOptions options;options.threshold_bytes=std::stoull(argv[5]);options.rows=std::stoi(argv[6]);
        std::shared_ptr<tc::seedvr2::SpatialFileStats> stats;
        std::ofstream pixels_file(output/"decoded.fp16.bin",std::ios::binary);
        tc::require(bool(pixels_file),"cannot create bounded frame export");
        uint64_t compared=0,unequal=0;int emitted=0;double nx=0,ny=0,dot=0,es=0,max_abs=0;
        auto sink=[&](int index,const Tensor &pixels){
            tc::require(index==emitted && pixels.shape()==mx::Shape{1,3,1,h,w} && pixels.dtype()==mx::float16,"spatial frame index/shape changed");
            auto current=mx::contiguous(mx::transpose(pixels,{0,2,3,4,1}));mx::eval(current);
            const auto *actual=current.data<mx::float16_t>();const auto *expected=reinterpret_cast<const mx::float16_t *>(saved.data()+size_t(index)*h*w*3);
            pixels_file.write(reinterpret_cast<const char *>(actual),size_t(h)*w*3*2);
            tc::require(bool(pixels_file),"bounded spatial frame export failed");
            for(size_t i=0;i<size_t(h)*w*3;++i){const double a=float(actual[i]),b=float(expected[i]);
                tc::require(std::isfinite(a) && std::isfinite(b),"nonfinite spatial/reference pixels");
                nx+=b*b;ny+=a*a;dot+=a*b;es+=(a-b)*(a-b);max_abs=std::max(max_abs,std::abs(a-b));
                unequal+=std::memcmp(actual+i,expected+i,2)!=0;++compared;}
            ++emitted;
        };
        mx::reset_peak_memory();Sampler sampler;auto start=tc::Clock::now();
        if(route=="baseline")vae.decode_stream(latent,frames,sink,{},cancelled);
        else vae.decode_spatial_stream(latent,frames,sink,options,stats,{},cancelled);
        const double seconds=std::chrono::duration<double>(tc::Clock::now()-start).count();const auto active=mx::get_peak_memory();
        const double l2=std::sqrt(es/std::max(nx,1e-20)),cosine=dot/std::max(std::sqrt(nx*ny),1e-20);
        const bool quality=emitted==frames && l2<=.05 && cosine>=.999;
        tc::require(!stats || (stats->disk_live_bytes==0 && stats->host_live_bytes==0),"spatial scratch/buffer live accounting did not drain");
        std::ofstream out(output/"report.json");tc::require(bool(out),"cannot write spatial decode report");
        out<<std::setprecision(17)<<"{\"schema\":\"seedvr2-spatial-decode-screen-v1\",\"route\":\""<<route
            <<"\",\"vae_upscale_recipe\":\""<<vae.upscale_recipe()<<"\",\"frames\":"<<emitted<<",\"height\":"<<h<<",\"width\":"<<w
            <<",\"seconds_with_host_quality\":"<<seconds<<",\"mlx_active_peak_bytes\":"<<active<<",\"sampled_process_footprint_peak_bytes\":"<<sampler.peak
            <<",\"bitwise_equal\":"<<(unequal==0?"true":"false")<<",\"unequal_values\":"<<unequal<<",\"compared_values\":"<<compared
            <<",\"relative_l2\":"<<l2<<",\"cosine\":"<<cosine<<",\"max_abs\":"<<max_abs<<",\"quality_passed\":"<<(quality?"true":"false")
            <<",\"file_written_bytes\":"<<(stats?stats->written_bytes:0)<<",\"file_read_bytes\":"<<(stats?stats->read_bytes:0)
            <<",\"file_peak_bytes\":"<<(stats?stats->disk_peak_bytes:0)<<",\"host_window_peak_bytes\":"<<(stats?stats->host_peak_bytes.load():0)
            <<",\"tile_convolutions\":"<<(stats?stats->tile_convolutions:0)<<",\"global_normalizations\":"<<(stats?stats->global_normalizations:0)
            <<",\"qualified\":false,\"scope\":\"decoder-only saved FP16 witness, host quality scan included; not original-FP32/full-request/paired speed or low-RAM qualification\"}\n";
        out.close();tc::require(quality,"spatial decoder quality gate failed");
        if(route=="baseline")tc::require(unequal==0,"baseline changed saved decoder bytes");
        std::cout<<"PASS "<<route<<" frames="<<emitted<<" l2="<<l2<<" cosine="<<cosine<<" peak="<<active<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
