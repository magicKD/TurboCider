#include "../../native/backends/mlx.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <iostream>

using namespace tc;
using namespace tc::streaming;
int main(int argc,char **argv){try {
    require(argc==2,"temporary fixture directory required");configure_streams();
    const auto root=std::filesystem::path(argv[1]);std::filesystem::create_directories(root);
    const auto path=root/"adapter.safetensors",other=root/"other.safetensors",alias=root/"alias.safetensors";
    constexpr int h=128,f=512,rank=8;const std::string prefix="transformer_blocks.0.img_mlp.";
    auto a=mx::reshape(mx::sin(mx::arange(rank*h,mx::float32)*.013f)*.05f,{rank,h});
    auto b=mx::reshape(mx::cos(mx::arange(f*rank,mx::float32)*.017f)*.04f,{f,rank});
    auto save=[&](const auto &target,float strength) {
        mx::save_safetensors(target.string(),{{prefix+"gate_layer.lora_A.weight",a},
            {prefix+"gate_layer.lora_B.weight",b*Tensor(strength)},
            {prefix+"gate_layer.alpha",Tensor(4.f,mx::float16)},
            {prefix+"proj.lora_A.weight",a*.75f},{prefix+"proj.lora_B.weight",b*.5f*Tensor(strength)},
            {prefix+"proj.alpha",Tensor(12.f,mx::bfloat16)}});
    };
    save(path,1.f);save(other,2.f);
    auto capture=[&](const auto &name,bool verified=true) {
        SourceFileIdentity file;file.logical_id="adapter";file.path=name;
        return verified?SourceLease::capture_verified({file}):SourceLease::capture({file});
    };
    auto lease=capture(path),warm=capture(path);
    require(lease->verification_bytes_read()==std::filesystem::file_size(path) && !lease->verification_cache_hits() &&
            warm->verification_bytes_read()==0 && warm->verification_cache_hits()==1 &&
            lease->file("adapter").content_digest==warm->file("adapter").content_digest,
            "native generation-bound proof did not distinguish cold hash from warm reuse");
    auto initialize=[&](Weights &weights) {
        weights.bind_arrays({prefix+"gate_up.weight"},{mx::astype(mx::reshape(mx::cos(mx::arange(2*f*h,mx::float32)*.007f)*.04f,{2*f,h}),mx::bfloat16)});
        weights.materialize();
    };
    Weights original,actual;initialize(original);initialize(actual);
    const auto master=actual.at(prefix+"gate_up.weight").id();std::atomic<bool> cancelled{false};
    const Event event=[](const std::string &,int,int){};
    const std::vector<LoRAAsset> assets{{path.string(),.75f,"transformer"},{path.string(),-.25f,"transformer"}};
    require(original.apply_loras(assets,"transformer",event,cancelled,true)==4,"original fixture bindings");
    require(actual.apply_loras_leased(assets,"transformer",event,cancelled,true,lease,{"adapter","adapter"})==4,
            "leased adapter dropped stacked gate/up bindings");
    auto input=mx::astype(mx::reshape(mx::cos(mx::arange(33*h,mx::float32)*.01f)*.25f,{1,33,h}),mx::bfloat16);
    auto expected=original.project(input,prefix+"gate_up"),observed=actual.project(input,prefix+"gate_up");mx::eval({expected,observed});
    require(mx::all(expected==observed).item<bool>() && actual.at(prefix+"gate_up.weight").id()==master,
            "leased source loader changed projection/alpha/stack order or the base master");
    struct stat before{};require(::stat(path.c_str(),&before)==0,"fixture stat");save(path,2.f);
    const struct timespec restored[]{before.st_atimespec,before.st_mtimespec};
    require(::utimensat(AT_FDCWD,path.c_str(),restored,0)==0,"fixture mtime restore");
    auto changed=capture(path);
    require(changed->file("adapter").bytes==lease->file("adapter").bytes && changed->verification_bytes_read()>0 &&
            changed->verification_cache_hits()==0 && changed->file("adapter").content_digest!=lease->file("adapter").content_digest,
            "same-size mutation with restored mtime reused stale proof");
    auto still_bound=actual.project(input,prefix+"gate_up");mx::eval(still_bound);
    require(mx::all(still_bound==expected).item<bool>(),"successful leased bind left mutable file reads lazy");
    int rejected=0;
    auto reject=[&](auto &&operation){try{operation();}catch(const std::exception&){++rejected;return;}
        throw std::runtime_error("invalid leased adapter accepted");};
    reject([&]{lease->revalidate_after_drain();});
    Weights stale;initialize(stale);
    reject([&]{stale.apply_loras_leased(assets,"transformer",event,cancelled,true,lease,{"adapter","adapter"});});
    require(stale.bytes()==0,"stale source failure did not clear the component");
    Weights metadata;initialize(metadata);auto unverified=capture(path,false);
    reject([&]{metadata.apply_loras_leased(assets,"transformer",event,cancelled,true,unverified,{"adapter","adapter"});});
    require(metadata.bytes()>0,"pre-bind unverified rejection changed untouched base");
    Weights invalid;initialize(invalid);
    reject([&]{invalid.apply_loras_leased(assets,"transformer",event,cancelled,true,changed,{"adapter"});});
    require(invalid.bytes()==0,"logical id failure did not clear the component");
    std::filesystem::create_symlink(path.filename(),alias);auto alias_lease=capture(alias);
    Weights raced;initialize(raced);
    const Event replace=[&](const std::string &phase,int step,int){if(phase=="load_lora" && step==0) {
        std::filesystem::remove(alias);std::filesystem::create_symlink(other.filename(),alias);
    }};
    reject([&]{raced.apply_loras_leased({{alias.string(),1.f,"transformer"}},"transformer",replace,cancelled,true,alias_lease,{"adapter"});});
    require(raced.bytes()==0,"alias replacement race published a partial adapter");
    Weights stopped;initialize(stopped);cancelled=true;
    reject([&]{stopped.apply_loras_leased(assets,"transformer",event,cancelled,true,changed,{"adapter","adapter"});});
    require(stopped.bytes()==0,"cancelled bind published a partial adapter");
    std::cout<<"PASS leased LoRA stacked_bindings=4 rejected="<<rejected
             <<": native cold/full-hash and warm proof, exact original alpha/projection, immutable master, held-fd/materialized sources, same-size/restored-mtime mutation, alias race, cancellation and fail-closed cleanup\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
