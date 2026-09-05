#include "runtime.hpp"
namespace tc {
ModelModule ltx_module() {
    return {"ltx-2.5-distilled",
        [] { return Recipe{"ltx-2.5-distilled",{{"text_encode",{}},{"connector",{"text_encode"}},{"av_stage1",{"connector"},8},{"latent_upsample",{"av_stage1"}},{"av_stage2",{"latent_upsample"},3},{"video_vae",{"av_stage2"}},{"audio_vae_vocoder",{"av_stage2"}},{"mux",{"video_vae","audio_vae_vocoder"}}},false}; },
        [](const Request& r) {
            require(r.width%64==0&&r.height%64==0,"LTX two-stage dimensions must be multiples of 64");
            require(r.steps==11,"LTX requires the 8+3 schedule");
            require(r.frames>=9 && r.frames%8==1,"LTX requires frames=8n+1");
            require(r.operation=="video.generate" || r.operation=="video.image","unsupported LTX operation");
            if(r.operation=="video.generate") require(r.inputs.empty(),"video.generate does not accept media inputs");
            else require(r.inputs.size()==1 && r.inputs[0].kind=="image" && r.inputs[0].role=="first_frame","LTX image-to-video requires one first_frame");
            require(r.fps==24,"LTX distilled contract requires 24 fps");
            require(r.residency=="resident"||r.residency=="component_staged","LTX block streaming is not yet supported");
        }, {},
        [] { return @{@"id":@"ltx-2.5-distilled",@"name":@"LTX 2.5 Distilled",@"executor":@NO,@"operations":@[@"video.generate",@"video.image"],@"inputs":@[@"text",@"image"],@"roles":@[@"first_frame"],@"max_images":@1,@"output":@"video",@"default_steps":@11,@"default_frames":@97,@"default_width":@704,@"default_height":@448}; }};
}
}
