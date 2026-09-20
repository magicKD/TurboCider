// Research-only finalization; does not grant public streaming result authority.
#include "models/z_image/vae.hpp"
#include "runtime/streaming/source_lease.hpp"
#include "media/image.hpp"
#include <fstream>
#include <iostream>
using namespace tc;
namespace fs=std::filesystem;
int main(int argc,char **argv) {
    try {
        require(argc==4,"expected Comfy VAE checkpoint, latent.npy, output directory");
        const fs::path output=argv[3];require(!fs::exists(output),"output already exists");
        streaming::SourceFileIdentity file;file.logical_id="vae";file.path=argv[1];
        auto lease=streaming::SourceLease::capture_verified({file});
        mx::set_default_device(mx::Device(mx::Device::gpu,0));
        auto latent=mx::load(argv[2]);
        require(latent.dtype()==mx::float32 && latent.shape()==mx::Shape({16,1,64,64}),"expected 512-square F32 latent");
        require(mx::all(mx::isfinite(latent)).item<bool>(),"nonfinite latent");
        Weights weights;std::atomic<bool> cancelled{false};
        const Event event=[](const std::string &,int,int){};
        weights.load_lease(lease,{"vae"},event,cancelled);
        auto decoded=z_image::decode_vae(latent,weights);mx::eval(decoded);
        require(decoded.shape()==mx::Shape({1,3,512,512}) && mx::all(mx::isfinite(decoded)).item<bool>(),"invalid decoded image");
        auto pixels=mx::transpose(decoded,{0,2,3,1});mx::eval(pixels);mx::synchronize();
        lease->revalidate_after_drain();
        fs::create_directories(output);
        mx::save((output/"decoded.npy").string(),mx::astype(decoded,mx::float32));
        save_png(pixels,(output/"image.png").string());
        std::ofstream report(output/"decode.json");
        report<<"{\"width\":512,\"height\":512,\"finite\":true,\"vae_artifact_digest\":\""<<lease->artifact_digest()<<"\",\"scope\":\"research decode, not verified request completion\"}\n";
        report.flush();require(bool(report),"cannot write decode report");
        std::cout<<"PASS shared native VAE decode 512x512\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
