#include "runtime.hpp"
namespace tc {
static ModelModule make_flux(const char *id, const char *name, int width, int height,
                             int memory_gb, const char *variant, bool hybrid) {
    std::string model(id), display(name), variant_name(variant);
    return {model,
        [model] { return Recipe{model, {{"text_encode",{}},{"denoise",{"text_encode"},4},{"vae_decode",{"denoise"}},{"export",{"vae_decode"}}},true}; },
        [model,variant_name](const Request& r) {
            require(r.model==model,"request model differs from FLUX session");
            require(r.width%16==0 && r.height%16==0,"FLUX dimensions must be multiples of 16");
            require(r.frames==1,"FLUX requires frames=1");
            require(r.operation=="image.generate" || r.operation=="image.transform" || r.operation=="image.edit","unsupported FLUX operation");
            require(r.model_variant=="auto" || r.model_variant==variant_name,"model_variant does not match the selected FLUX module");
            if(r.operation=="image.generate") require(r.inputs.empty(),"image.generate does not accept image inputs");
            else {
                require(!r.inputs.empty() && r.inputs.size()<=8,"FLUX requires 1...8 images");
                if(r.operation=="image.transform") require(r.inputs.size()==1,"image.transform requires one init_image");
                for(auto& input:r.inputs) require(input.kind=="image" && input.role==(r.operation=="image.transform"?"init_image":"reference"),"incorrect FLUX image role");
            }
            require(r.residency=="resident" || r.residency=="component_staged","FLUX block streaming is not supported");
            require(r.loras.size()<=8,"at most eight LoRA adapters may be active");
            require(r.loras.empty() || r.execution!="gpu_ane",
                    "FLUX LoRA currently requires GPU execution; base Core ML artifacts cannot represent merged LoRA MLP weights");
            for(const auto& l:r.loras) {
                require(!l.path.empty(),"LoRA path is required");
                require(l.strength>=-8.f&&l.strength<=8.f,"LoRA strength must be -8...8");
                require(l.role=="transformer"||l.role=="text_encoder","unsupported FLUX LoRA role");
            }
        },
        [model](const std::filesystem::path& root) { return std::make_unique<Flux>(root,model); },
        [model,display,width,height,memory_gb,variant_name,hybrid] { return @{
            @"id":@(model.c_str()),@"name":@(display.c_str()),@"executor":@YES,
            @"variant":@(variant_name.c_str()),@"operations":@[@"image.generate",@"image.transform",@"image.edit"],
            @"inputs":@[@"text",@"image"],@"roles":@[@"init_image",@"reference"],@"max_images":@8,
            @"output":@"image",@"default_steps":@4,@"default_frames":@1,@"default_width":@(width),@"default_height":@(height),
            @"supports_lora":@YES,@"supports_gpu_ane":@(hybrid),@"memory_estimate_gb":@(memory_gb)}; }};
}
ModelModule flux4_module(){return make_flux("flux2-klein-4b","FLUX.2 Klein 4B",512,512,16,"flux2-klein-4b",true);}
ModelModule flux9_module(){return make_flux("flux2-klein-9b","FLUX.2 Klein 9B",512,512,28,"flux2-klein-9b",false);}
}
