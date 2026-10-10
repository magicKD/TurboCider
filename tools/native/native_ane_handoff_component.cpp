// One synthetic FP16 request through W0 or split admission. No model load.
#include "../../native/backends/metal_graph/native_ane_handoff.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>

using namespace tc::metal_graph;
namespace pa=tc::ane::private_api;
namespace ag=tc::ane::gpu;
namespace {
void check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
constexpr std::string_view source=R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void native_ane_producer(device half* x [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i<256) x[i]=half(0.25f+float(i%16)/32.0f);
}
kernel void native_ane_consumer(device const half* y [[buffer(0)]],device float* z [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    if(i<256) z[i]=float(y[i])+1.0f;
}
)metal";
constexpr std::string_view mil=R"mil(program(1.3)
{
 func main_ane<ios18>(tensor_buffer<fp16, shape=[1, 1, 1, 256], strides=[256, 256, 256, 1], interleave_factors=[1, 1, 1, 1]> x) {
  tensor<fp16, [1, 1, 1, 256]> xt = tensor_buffer_to_tensor<ios17>(input = x);
  tensor<fp16, [1, 1, 1, 256]> shifted = add(x = xt, y = fp16(0.5));
  tensor_buffer<fp16, shape=[1, 1, 1, 256], strides=[256, 256, 256, 1], interleave_factors=[1, 1, 1, 1]> y = tensor_to_tensor_buffer<ios17>(input = shifted, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([256, 256, 256, 1]));
 } -> (y);
}
)mil";
}
int main(int argc,char** argv) {
    if(argc!=3 || (std::string_view(argv[1])!="--tiny-device" && std::string_view(argv[1])!="--tiny-device-split")) {
        std::cout<<"usage: native-ane-handoff-component --tiny-device[\u002dsplit] <fresh-owned-cache>\n";return 2;
    }
    try {
        const auto cache=std::filesystem::absolute(argv[2]).lexically_normal();
        check(!std::filesystem::exists(cache),"component cache must be fresh and independently owned");
        size_t allocations=0;
        BackendOptions options;
        options.allocation.reserve=[](size_t,std::string_view){return std::make_shared<int>(0);};
        options.allocation.allocated=[&](const std::shared_ptr<void>&,void*,size_t){++allocations;};
        Backend backend(options);ag::Device device;
        check(device.native_device()==backend.native_device(),"native/ANE surfaces are on different Metal devices");
        auto library=backend.compile_library(source,"native-ane-synthetic-producer-consumer");
        auto producer=backend.prepare_pipeline(library,"native_ane_producer");
        auto consumer=backend.prepare_pipeline(library,"native_ane_consumer");
        const auto started=std::chrono::steady_clock::now();
        pa::Program program(device,mil,{},cache);
        const double compile_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        ag::Surface x(device,1,256,ag::Element::FP16),y(device,1,256,ag::Element::FP16);
        const auto original=x.metal_buffer_lease(),slice=x.slice_rows(0,1).metal_buffer_lease();
        original.validate(backend.native_device());slice.validate(backend.native_device());
        check(original.native_buffer==slice.native_buffer && original.owner==slice.owner,
              "surface view created a different backing/owner");
        NativeAneHandoff handoff(backend,program,{{"x",x}},{{"y",y}},{1,1,1,1,1});
        check(allocations==0,"ANE imports made a second charged Metal allocation");
        CommandGraph graph;graph.dispatch(producer,{handoff.input("x").view},{1,1,1},{256,1,1});
        handoff.submit_producer(std::move(graph));
        const bool split=std::string_view(argv[1])=="--tiny-device-split";
        NativeAneReadPermit permit;
        if(split) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            auto producer_permit=handoff.finish_producer_until(deadline);
            check(producer_permit.is_valid() && detail::same_ane_request(producer_permit.provenance(),handoff.provenance()),
                  "producer permit does not identify captured producer");
            check(!handoff.status().ane_submission_attempted,"producer finish unexpectedly submitted ANE");
            handoff.submit_ane_until(std::move(producer_permit),deadline);
            check(!producer_permit.is_valid(),"consumed producer permit remained valid");
            check(handoff.status().ane_submission_attempted && !handoff.status().ok,
                  "submit ANE alone incorrectly granted output success");
            permit=handoff.finish_ane_until(deadline);
        } else permit=handoff.finish(std::chrono::seconds(5),std::chrono::seconds(5));
        check(bool(permit),"native handoff failed to grant a result read permit");
        check(detail::same_ane_request(permit.provenance(),handoff.provenance()),"permit does not match actual request");
        auto status=handoff.status();
        check(status.ok && !status.retired && status.all_work_physically_complete &&
              status.ready_value==1 && status.done_value==1,"native handoff completion evidence is incomplete");
        bool rejected=false;try{(void)handoff.finish();}catch(const tc::ane::CapabilityError&){rejected=true;}
        check(rejected,"result permit could be issued twice");
        const auto output=backend.allocate(256*sizeof(float),"native-ane-restore-check");
        CommandGraph restore;restore.dispatch(consumer,{permit.output("y").view,
            output.view(0,output.size_bytes(),ScalarType::FP32)},{1,1,1},{256,1,1});
        const auto completed=backend.submit(std::move(restore)).wait(std::chrono::seconds(5));
        check(completed.ok && completed.physically_complete,"native output consumer did not complete");
        size_t finite=0;double maximum=0;
        for(size_t i=0;i<256;++i) {
            const float actual=static_cast<const float*>(output.contents())[i];
            check(std::isfinite(actual),"native bridge produced nonfinite output");++finite;
            maximum=std::max(maximum,std::abs(double(actual)-(1.75+double(i%16)/32)));
        }
        check(maximum==0,"GPU->private ANE->GPU arithmetic mismatch");
        check(allocations==1,"unexpected native output/import allocation count");
        std::cout<<"native_ane_handoff_component=PASS device="<<backend.device_name()
                 <<" requests=1 finite="<<finite<<" max_abs="<<maximum
                 <<" independent_events=1 imported_allocations=0 fresh_program="<<program.compiled_now()
                 <<" compile_seconds="<<compile_seconds<<" consumer_gpu_seconds="<<completed.gpu_seconds
                 <<" admission="<<(split ? "split" : "W0")
                 <<" models=0 full_model=0 overlap=0 e2e=0\n";
    } catch(const std::exception& error) {
        std::cerr<<"native_ane_handoff_component=FAIL "<<error.what()<<'\n';return 1;
    }
}
