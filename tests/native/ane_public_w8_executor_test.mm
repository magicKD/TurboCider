#include "../../native/backends/ane_public_w8.hpp"
#include "../../native/backends/ane_runtime_convert.hpp"
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace tc::ane;
namespace {
void check(bool ok, const std::string &reason) { if (!ok) throw std::runtime_error(reason); }
struct Storage {
    id<MTLBuffer> buffer;
    DeviceMatrixView view;
    Storage(id<MTLDevice> gpu, int rows, int cols, DType dtype = DType::FP32) {
        const size_t item = dtype == DType::FP32 ? 4 : 2, pitch = size_t(cols) * item + 16;
        buffer = [gpu newBufferWithLength:256 + size_t(rows) * pitch + 256 options:MTLResourceStorageModeShared];
        check(buffer != nil, "test allocation failed");
        auto owner = std::shared_ptr<void>((__bridge_retained void *)buffer, [](void *p) { CFRelease(p); });
        view = {(__bridge void *)buffer, buffer.length, 256, rows, cols, pitch, dtype, owner};
        std::memset(buffer.contents, 0x5a, buffer.length);
    }
    void *row(int r) { return static_cast<char *>(buffer.contents) + view.offset_bytes + size_t(r) * view.row_stride_bytes; }
    DeviceWeightView weight() {
        return {view.buffer, view.buffer_bytes, view.offset_bytes, view.row_stride_bytes, view.rows, view.cols,
                DeviceWeightEncoding::Dense, view.dtype, 32, {}, {}, view.owner};
    }
    void guard() {
        const auto *p = static_cast<const uint8_t *>(buffer.contents);
        for (size_t i = 0; i < view.offset_bytes; ++i) check(p[i] == 0x5a, "prefix overwrite");
        const size_t item = view.dtype == DType::FP32 ? 4 : 2;
        for (int r = 0; r < view.rows; ++r) for (size_t c = size_t(view.cols) * item; c < view.row_stride_bytes; ++c)
            check(p[view.offset_bytes + size_t(r) * view.row_stride_bytes + c] == 0x5a, "row padding overwrite");
        for (size_t i = view.offset_bytes + size_t(view.rows) * view.row_stride_bytes; i < buffer.length; ++i)
            check(p[i] == 0x5a, "suffix overwrite");
    }
};
float bf16(const void *row, int c) { return std::bit_cast<float>(uint32_t(static_cast<const uint16_t *>(row)[c]) << 16); }
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    @autoreleasepool {
      try {
        constexpr int chunk = 33, rows = chunk * 3, h = 128, f = 512;
        const bool lookahead=false;
        setenv("TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD",lookahead?"1":"0",1);
        setenv("TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE","1",1);
        PublicW8Graph graph(argv[1],256u<<20);
        check(graph.supports_device_weights() && graph.supports_device_io() && graph.data_path() == "w8a8_hadamard" &&
              graph.weight_recipe() == "sylvester-dh-b128-b512-rne-norm-f16-v2+public-int8-io-v1", "W8 capabilities missing");
        check(graph.slot_bytes() <= graph.estimated_bytes(), "memory estimate understated");
        check(graph.device_submission_fence_enabled(),"first request submission policy not enabled");
        check(graph.activation_lookahead_enabled()==lookahead,"A8 lookahead policy not reported");
        std::string error;
        check(graph.self_test(error), error);
        auto gpu = MTLCreateSystemDefaultDevice();
        Storage g(gpu,f,h), u(gpu,f,h), d(gpu,h,f), x(gpu,rows,h,DType::BF16), y(gpu,rows,h,DType::BF16),
                dg(gpu,rows,f), du(gpu,rows,f), hidden(gpu,rows,f,DType::BF16);
        for (int r=0;r<rows;++r) {
            for (int c=0;c<h;++c) static_cast<uint16_t*>(x.row(r))[c]=round_bf16(((r*3+c*7)%17-8)/8.f);
            for (int c=0;c<f;++c) { static_cast<float*>(dg.row(r))[c]=.02f*((r+c)%3-1); static_cast<float*>(du.row(r))[c]=.03f*((r*2+c)%3-1); }
        }
        auto fill_weights=[&](float gate,float up,float down) {
            for(int r=0;r<f;++r) {
                std::fill_n(static_cast<float*>(g.row(r)),h,0); std::fill_n(static_cast<float*>(u.row(r)),h,0);
                static_cast<float*>(g.row(r))[r%h]=gate; static_cast<float*>(u.row(r))[r%h]=up;
            }
            for(int r=0;r<h;++r) { std::fill_n(static_cast<float*>(d.row(r)),f,0); static_cast<float*>(d.row(r))[r]=down; }
        };
        auto stage=[&] {
            graph.stage_device_weights({g.weight(),u.weight(),d.weight()});
            const auto result=graph.wait_stage(); check(result.ok,result.error); check(result.stage_seconds>0,"missing stage timing");
        };
        auto run=[&](bool adapter,float gate,float up,float down) {
            graph.launch_device(x.view,y.view,adapter?std::optional<DeviceAdapterInput>({dg.view,du.view,hidden.view}):std::nullopt);
            const auto result=graph.finish(); check(result.ok,result.error); check(result.calls>=2,"multichunk not exercised");
            check(result.activation_prefetches==(lookahead?2u:0u) && result.activation_wait_seconds>0,
                  "bounded A8 staging lookahead/consumer-reuse telemetry mismatch");
            check(result.input_seconds>0 && result.prediction_seconds>0 && result.output_seconds>0 && result.total_seconds>0,
                  "W8 exposed timing telemetry missing");
            double diff=0,norm=0,hdiff=0,hnorm=0;
            for(int r=0;r<rows;++r) for(int c=0;c<h;++c) {
                const float input=bf16(x.row(r),c), gv=input*gate+(adapter?static_cast<float*>(dg.row(r))[c]:0),
                    uv=input*up+(adapter?static_cast<float*>(du.row(r))[c]:0), hv=gv/(1+std::exp(-gv))*uv, expected=hv*down, actual=bf16(y.row(r),c);
                check(std::isfinite(actual),"nonfinite W8 output"); diff+=double(actual-expected)*(actual-expected); norm+=double(expected)*expected;
                if(adapter) { const float actual_h=bf16(hidden.row(r),c); hdiff+=double(actual_h-hv)*(actual_h-hv); hnorm+=double(hv)*hv; }
            }
            check(norm>0&&std::sqrt(diff/norm)<.06,"W8 dense-source output oracle mismatch");
            check(!adapter||(hnorm>0&&std::sqrt(hdiff/hnorm)<.05),"W8 hidden dtype/scale oracle mismatch");
            return result;
        };
        for(bool adapter:{false,true,false}) { fill_weights(.125f,-.25f,.25f); stage(); run(adapter,.125f,-.25f,.25f); }
        fill_weights(-.25f,.125f,-.5f); stage(); run(true,-.25f,.125f,-.5f);
        fill_weights(.125f,-.25f,.25f); stage(); run(false,.125f,-.25f,.25f);
        if(graph.weight_code_cache_stats().enabled) {
            auto epoch=std::make_shared<int>(1);
            std::vector<DeviceWeightView> immutable{g.weight(),u.weight(),d.weight()};
            for(auto &source:immutable){source.allocation_identity=epoch;source.immutable_generation=true;}
            const auto before=graph.weight_code_cache_stats();
            for(int repeat=0;repeat<2;++repeat) {
                graph.stage_device_weights(immutable);check(graph.wait_stage().ok,"Public cached weight stage failed");
                run(repeat!=0,.125f,-.25f,.25f);
            }
            const auto cached=graph.weight_code_cache_stats();
            check(cached.fills>=before.fills+3 && cached.hits>=before.hits+3 && cached.ready_entries>=3 &&
                  cached.live_capacity_bytes<=cached.budget_bytes && cached.peak_capacity_bytes<=cached.budget_bytes,
                  "Public full executor lacks real bounded converted-code reuse");
            if(cached.native_surface_storage)
                check(cached.surface_bind_hits>=before.surface_bind_hits+3 && cached.copy_hits==0,
                      "Public native weight hit still copied the cache");
            std::cout<<"PASS Public converted weight cache: completed hits, base/real corrections, unchanged output oracle, bounded capacity\n";
            // The epoch dies before mutable fixture weights are changed.
            // Subsequent staging must purge these weak generation entries.
        }
        Storage fp32_partial(gpu,rows,h,DType::FP32),wrong_hidden(gpu,rows,f,DType::FP32);
        check(graph.supports_fp32_device_output(),"Sylvester F32 partial capability missing");
        for(bool adapter:{false,true,false}) {
            run(adapter,.125f,-.25f,.25f);
            std::vector<uint8_t> hidden_snapshot(hidden.buffer.length);
            if(adapter)std::memcpy(hidden_snapshot.data(),hidden.buffer.contents,hidden_snapshot.size());
            graph.launch_device(x.view,fp32_partial.view,adapter?std::optional<DeviceAdapterInput>({dg.view,du.view,hidden.view}):std::nullopt);
            const auto result=graph.finish();check(result.ok,result.error);
            check(result.copied_output_bytes==size_t(rows)*(h*4+(adapter?f*2:0)),"F32 partial plus model-dtype hidden byte receipt wrong");
            for(int r=0;r<rows;++r)for(int c=0;c<h;++c)
                check(round_bf16(static_cast<float*>(fp32_partial.row(r))[c])==static_cast<uint16_t*>(y.row(r))[c],
                    "Sylvester F32 restore changed normalized result/final BF16 boundary");
            if(adapter)check(!std::memcmp(hidden_snapshot.data(),hidden.buffer.contents,hidden_snapshot.size()),
                "F32 base partial changed BF16 LoRA hidden ABI/padding");
            fp32_partial.guard();hidden.guard();
        }
        graph.launch_device(x.view,fp32_partial.view,DeviceAdapterInput{dg.view,du.view,wrong_hidden.view});
        check(!graph.finish().ok,"F32 partial silently widened LoRA hidden ABI");
        stage();run(true,.125f,-.25f,.25f);
        std::cout<<"PASS W8 F32 GPU partial restore: base/LoRA/base, exact BF16 boundary, unchanged hidden ABI, bytes/guards and recovery\n";
        // Prepare the other bank while current A is in flight. Current output
        // must remain A, and source leases survive until explicit activation.
        Storage bg(gpu,f,h),bu(gpu,f,h),bd(gpu,h,f);
        for(int r=0;r<f;++r) {
            std::fill_n(static_cast<float*>(bg.row(r)),h,0);std::fill_n(static_cast<float*>(bu.row(r)),h,0);
            static_cast<float*>(bg.row(r))[r%h]=.25f;static_cast<float*>(bu.row(r))[r%h]=.125f;
        }
        for(int r=0;r<h;++r){std::fill_n(static_cast<float*>(bd.row(r)),f,0);static_cast<float*>(bd.row(r))[r]=-.5f;}
        auto lease=std::make_shared<int>(1);std::weak_ptr<void> leased=lease;
        std::vector<DeviceWeightRegion> future{{bg.weight(),{0,f,0,h,128}},{bu.weight(),{0,f,0,h,128}},{bd.weight(),{0,h,0,f,512}}};
        future[0].source.owner=lease;
        graph.launch_device(x.view,y.view);graph.prefetch_device_weight_regions(future);
        lease.reset();future[0].source.owner.reset();
        try{graph.prefetch_device_weight_regions(future);throw std::runtime_error("third bank request accepted");}catch(const std::runtime_error&e){check(std::string(e.what())!="third bank request accepted",e.what());}
        auto current=graph.finish();check(current.ok,current.error);check(!leased.expired(),"future source lease released before activation");
        for(int r=0;r<rows;++r)for(int c=0;c<h;++c){const float xv=bf16(x.row(r),c),gv=xv*.125f;
            const float expected=gv/(1+std::exp(-gv))*(xv*-.25f)*.25f;
            check(std::abs(bf16(y.row(r),c)-expected)<.0002f+.06f*std::abs(expected),"future bank overwrote in-flight current A");}
        auto mismatch=future;mismatch[2].selection.rotation_seed++;
        check(!graph.activate_prefetched_weights(mismatch),"wrong future source/selection identity accepted");
        auto activated=graph.activate_prefetched_weights(future);check(activated&&activated->ok,"future B activation failed");
        check(leased.expired(),"completed future source lease not released");run(false,.25f,.125f,-.5f);
        auto broken=future;broken[0].source.owner=bg.view.owner;broken[0].source.buffer_bytes=broken[0].source.offset_bytes+1;
        graph.launch_device(x.view,y.view);graph.prefetch_device_weight_regions(broken);
        check(graph.finish().ok,"failed future producer corrupted current B");
        auto rejected=graph.activate_prefetched_weights(broken);check(rejected&&!rejected->ok,"failed future bank published as ready");
        graph.launch_device(x.view,y.view);check(!graph.finish().ok,"failed future activation reused stale current weights");
        stage();run(false,.125f,-.25f,.25f);
        graph.prefetch_device_weight_regions({{bg.weight(),{0,f,0,h,128}},{bu.weight(),{0,f,0,h,128}},{bd.weight(),{0,h,0,f,512}}});
        graph.discard_prefetched_weights();check(!graph.activate_prefetched_weights(future),"discarded prefetch remained activatable");
        // Normalized up/hidden graph must recover BF16-range values, retain
        // headroom across future chunks/layers, and return the same hidden ABI.
        for(int r=0;r<rows;++r) std::fill_n(static_cast<uint16_t*>(x.row(r)),h,round_bf16(8.f));
        fill_weights(32.f,32.f,.25f); stage();
        const auto high=run(true,32.f,32.f,.25f);
        check(high.overflow_retries>0&&high.headroom_scale>=4,"W8 headroom retry not exercised");
        check(high.headroom_start_scale==1,"W8 retry did not report actual launch headroom");
        const auto reused=run(false,32.f,32.f,.25f);
        check(!reused.overflow_retries && reused.headroom_start_scale==high.headroom_scale &&
            reused.headroom_scale==high.headroom_scale,"W8 headroom not retained/reported");
        graph.launch_device(x.view,x.view); const auto alias=graph.finish();
        check(!alias.ok && alias.headroom_start_scale==high.headroom_scale &&
            alias.headroom_scale==high.headroom_scale,"W8 alias accepted or failed launch lost actual headroom");
        *static_cast<uint16_t*>(x.row(0))=0x7f80;
        graph.launch_device(x.view,y.view); check(!graph.finish().ok,"W8 nonfinite accepted");
        *static_cast<uint16_t*>(x.row(0))=round_bf16(8.f);
        run(false,32.f,32.f,.25f);
        *static_cast<uint16_t*>(x.row(chunk))=0x7f80;
        graph.launch_device(x.view,y.view);
        const auto partial=graph.finish();
        check(!partial.ok && partial.calls==1,"late failed W8 chunk reported partial success");
        *static_cast<uint16_t*>(x.row(chunk))=round_bf16(8.f);
        run(false,32.f,32.f,.25f);
        *static_cast<uint16_t*>(x.row(chunk*2))=0x7f80;
        graph.launch_device(x.view,y.view);
        const auto last_partial=graph.finish();
        check(!last_partial.ok && last_partial.calls==2,"last failed A8 slot reuse reported partial success");
        *static_cast<uint16_t*>(x.row(chunk*2))=round_bf16(8.f);
        run(false,32.f,32.f,.25f);
        auto short_weight=g.weight(); short_weight.buffer_bytes=short_weight.offset_bytes+1;
        graph.stage_device_weights({short_weight,u.weight(),d.weight()}); check(!graph.wait_stage().ok,"short W8 source accepted");
        graph.launch_device(x.view,y.view); check(!graph.finish().ok,"failed staging published old bank");
        stage(); run(false,32.f,32.f,.25f);
        for(auto*storage:{&g,&u,&d,&x,&y,&dg,&du,&hidden}) storage->guard();
        std::cout<<"PASS Public W8 shared executor: three chunks, real correction ABI/base-A-base, two banks/prefetch identity, FP32 restore, persistent headroom, late failures/guards/refill\n";
      }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
