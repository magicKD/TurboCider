#include "../../native/backends/ane_gpu.hpp"
#include "../../native/backends/private/native_handoff_types.hpp"
#include "../../native/backends/metal_graph/native_ane_handoff.hpp"
#include "../../native/backends/metal_graph/native_ane_handoff_state.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace tc::ane;
using namespace tc::ane::private_api;

namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F body) {
    try { body(); } catch (const CapabilityError &) { return; }
    throw std::runtime_error("invalid handoff metadata accepted");
}
struct Fixture {
    // CPU tokens only. No Metal object is created, messaged or dereferenced.
    int device = 1, other_device = 2, ready = 3, done = 4, buffer = 5;
    std::shared_ptr<void> owner = std::make_shared<int>(6);
    AneRequestProvenance provenance() const { return {"program-key", 7, 8, 9, 10, 11, 12, 0}; }
    AneEventEndpoint ready_event() { return {&ready, &device, owner, 13, AneEventRole::GpuReady}; }
    AneEventEndpoint done_event() { return {&done, &device, owner, 1, AneEventRole::AneDone}; }
    AneEventBinding binding() { return {ready_event(), done_event(), provenance()}; }
    gpu::SurfaceBufferLease lease() {
        return {&buffer, &device, 4096, 4096, 64, 128, 64, 2, 3, gpu::Element::FP16, owner, owner};
    }
};
}

int main() {
    try {
        static_assert(!std::is_copy_constructible_v<AneEventBinding>);
        static_assert(std::is_nothrow_move_constructible_v<AneEventBinding>);
        static_assert(!std::is_copy_constructible_v<tc::metal_graph::NativeAneReadPermit>);
        static_assert(!std::is_constructible_v<tc::metal_graph::NativeAneReadPermit,
                      std::shared_ptr<tc::metal_graph::detail::NativeAneHandoffState>>);
        static_assert(!std::is_copy_constructible_v<tc::metal_graph::NativeAneProducerPermit>);
        static_assert(!std::is_copy_assignable_v<tc::metal_graph::NativeAneProducerPermit>);
        static_assert(std::is_nothrow_move_constructible_v<tc::metal_graph::NativeAneProducerPermit>);
        static_assert(!std::is_constructible_v<tc::metal_graph::NativeAneProducerPermit,
                      std::shared_ptr<tc::metal_graph::detail::NativeAneHandoffState>>);
        using Protocol = tc::metal_graph::detail::NativeAnePermitProtocol;
        const tc::metal_graph::Completion gpu_ok{true,true,false,1,1,0.,{}};
        const gpu::Completion ane_ok{true,false,{},true,false};
        {
            Protocol protocol;
            rejects([&] { protocol.consume_producer(true); });
            rejects([&] { protocol.submitted(); });
            rejects([&] { protocol.ane_complete(ane_ok,true,1,1); });
            protocol.producer_complete(gpu_ok,1,0);
            rejects([&] { protocol.producer_complete(gpu_ok,1,0); });
            rejects([&] { protocol.consume_producer(false); });
            check(protocol.phase()==Protocol::Phase::ProducerReady,"foreign permit mutated target admission");
            protocol.consume_producer(true);
            rejects([&] { protocol.consume_producer(true); });
            check(!protocol.readable(),"producer admission granted an output read");
            protocol.submitted();
            check(!protocol.readable(),"submission alone granted an output read");
            protocol.ane_complete(ane_ok,true,1,1);
            check(protocol.readable(),"complete request lost read qualification");
            rejects([&] { protocol.ane_complete(ane_ok,true,1,1); });
        }
        for (int fault=0;fault<8;++fault) {
            Protocol protocol;
            auto receipt=gpu_ok; uint64_t ready=1,done=0;
            switch(fault) {
            case 0: receipt.ok=false;break;
            case 1: receipt.physically_complete=false;break;
            case 2: receipt.timed_out=true;break;
            case 3: ready=0;break;
            case 4: ready=2;break;
            case 5: done=1;break;
            case 6: receipt.submitted_buffers=receipt.completed_buffers=0;break;
            case 7: receipt.completed_buffers=0;break;
            }
            rejects([&] { protocol.producer_complete(receipt,ready,done); });
            check(protocol.retired(),"failed producer admission did not retire");
            rejects([&] { protocol.producer_complete(gpu_ok,1,0); });
            check(!protocol.readable(),"late producer completion resurrected result");
        }
        for (int fault=0;fault<7;++fault) {
            Protocol protocol;
            protocol.producer_complete(gpu_ok,1,0);protocol.consume_producer(true);protocol.submitted();
            auto receipt=ane_ok;bool matches=true;uint64_t ready=1,done=1;
            switch(fault) {
            case 0: receipt.ok=false;break;
            case 1: receipt.physical_complete=false;break;
            case 2: receipt.retired=true;break;
            case 3: receipt.timed_out=true;break;
            case 4: matches=false;break;
            case 5: ready=2;break;
            case 6: done=0;break;
            }
            rejects([&] { protocol.ane_complete(receipt,matches,ready,done); });
            check(protocol.retired(),"failed ANE receipt did not retire");
            const auto original_error=protocol.error();
            protocol.retire("late replacement");
            rejects([&] { protocol.ane_complete(ane_ok,true,1,1); });
            check(protocol.retired() && !protocol.readable() && protocol.error()==original_error,
                  "late callback changed sticky result/error");
        }
        for (int stage=0;stage<4;++stage) {
            Protocol protocol;
            if(stage>0)protocol.producer_complete(gpu_ok,1,0);
            if(stage>1)protocol.consume_producer(true);
            if(stage>2)protocol.submitted();
            protocol.retire("shared deadline expired",true);
            rejects([&] { protocol.producer_complete(gpu_ok,1,0); });
            rejects([&] { protocol.consume_producer(true); });
            rejects([&] { protocol.submitted(); });
            rejects([&] { protocol.ane_complete(ane_ok,true,1,1); });
            check(protocol.timed_out() && !protocol.readable(),"expired protocol was revived");
        }
        {
            using namespace std::chrono;
            const auto now=steady_clock::time_point(seconds(100));
            using tc::metal_graph::detail::ane_remaining;
            using tc::metal_graph::detail::ane_deadline;
            check(ane_remaining(now,now)==milliseconds(0),"exact deadline is not expired");
            check(ane_remaining(now-milliseconds(1),now)==milliseconds(0),"past deadline wrapped");
            check(ane_remaining(now+microseconds(1),now)==milliseconds(1),"sub-ms wait truncated to zero");
            check(ane_deadline(milliseconds(10),now)==now+milliseconds(10),"relative end incorrect");
            check(ane_deadline(milliseconds(0),now)==now,"zero timeout extended");
            rejects([&] { (void)ane_deadline(milliseconds(-1),now); });
            check(ane_deadline(milliseconds::max(),now)==steady_clock::time_point::max(),"large timeout overflow");
        }
        Fixture f;
        // MTLSharedEvent.device may be nil. The creator-device provenance must
        // still match; a non-null actual device from another GPU is rejected.
        for (void *creator : {static_cast<void*>(nullptr), static_cast<void*>(&f.device), static_cast<void*>(&f.other_device)})
            for (void *reported : {static_cast<void*>(nullptr), static_cast<void*>(&f.device), static_cast<void*>(&f.other_device)}) {
                const auto validate = [&] { tc::ane::private_api::detail::validate_shared_event_device(creator, reported, &f.device); };
                if (creator == &f.device && (!reported || reported == &f.device)) validate();
                else rejects(validate);
            }
        rejects([&] { tc::ane::private_api::detail::validate_shared_event_device(&f.device, nullptr, nullptr); });
        auto good = f.binding();
        good.validate_metadata(&f.device, "program-key", 7);
        good.validate_w0_values(13, 0); // independent event values need not be ordered
        rejects([&] { good.validate_w0_values(12, 0); });
        rejects([&] { good.validate_w0_values(14, 0); });
        rejects([&] { good.validate_w0_values(13, 1); });
        rejects([&] { good.validate_metadata(&f.other_device, "program-key", 7); });
        rejects([&] { good.validate_metadata(&f.device, "other-program", 7); });
        rejects([&] { good.validate_metadata(&f.device, "program-key", 8); });

        for (int fault = 0; fault < 12; ++fault) {
            auto r = f.ready_event(), d = f.done_event();
            auto p = f.provenance();
            switch (fault) {
            case 0: r.owner.reset(); break;
            case 1: d.owner.reset(); break;
            case 2: r.role = AneEventRole::AneDone; break;
            case 3: d.role = AneEventRole::GpuReady; break;
            case 4: d.native_event = nullptr; break;
            case 5: r.value = 0; break;
            case 6: d.value = 0; break;
            case 7: p.domain = 0; break;
            case 8: p.epoch = 0; break;
            case 9: p.request = 0; break;
            case 10: p.slot_generation = 0; break;
            case 11: p.producer_generation = 0; break;
            }
            AneEventBinding invalid(std::move(r), std::move(d), std::move(p));
            rejects([&] { invalid.validate_metadata(&f.device, "program-key", 7); });
        }
        auto p = f.provenance(); p.function_index = 1;
        AneEventBinding wrong_function(f.ready_event(), f.done_event(), p);
        rejects([&] { wrong_function.validate_metadata(&f.device, "program-key", 7); });
        auto shared_done = f.done_event(); shared_done.native_event = &f.ready; shared_done.value = 14;
        AneEventBinding shared(f.ready_event(), shared_done, f.provenance());
        shared.validate_metadata(&f.device, "program-key", 7);
        shared.validate_w0_values(13, 13);
        rejects([&] { shared.validate_w0_values(13, 12); });
        shared_done.value = 13;
        AneEventBinding invalid_shared(f.ready_event(), shared_done, f.provenance());
        rejects([&] { invalid_shared.validate_metadata(&f.device, "program-key", 7); });
        auto moved = std::move(good);
        moved.validate_metadata(&f.device, "program-key", 7);
        rejects([&] { good.validate_metadata(&f.device, "program-key", 7); });

        const auto receipt = f.provenance();
        check(tc::metal_graph::detail::same_ane_request(receipt, receipt), "matching receipt rejected");
        for (int fault = 0; fault < 8; ++fault) {
            auto stale = receipt;
            switch (fault) {
            case 0: stale.program_key = "another-program"; break;
            case 1: ++stale.program_generation; break;
            case 2: ++stale.domain; break;
            case 3: ++stale.epoch; break;
            case 4: ++stale.request; break;
            case 5: ++stale.slot_generation; break;
            case 6: ++stale.producer_generation; break;
            case 7: ++stale.function_index; break;
            }
            check(!tc::metal_graph::detail::same_ane_request(receipt, stale), "stale receipt accepted");
        }

        f.lease().validate(&f.device);
        for (int fault = 0; fault < 12; ++fault) {
            auto lease = f.lease();
            switch (fault) {
            case 0: lease.owner.reset(); break;
            case 1: lease.allocation_identity.reset(); break;
            case 2: lease.native_device = &f.other_device; break;
            case 3: lease.native_buffer = nullptr; break;
            case 4: lease.row_stride_bytes = 0; break;
            case 5: lease.row_stride_bytes = 5; break;
            case 6: lease.offset_bytes = 1; break;
            case 7: lease.offset_bytes = 4090; break;
            case 8: lease.size_bytes = 126; break;
            case 9: lease.allocated_bytes = 2048; break;
            case 10: lease.element = static_cast<gpu::Element>(99); break;
            case 11: lease.row_stride_bytes = std::numeric_limits<size_t>::max() - 1; break;
            }
            rejects([&] { lease.validate(&f.device); });
        }
        auto unrelated = std::make_shared<int>(9);
        auto bad_owner = f.lease(); bad_owner.allocation_identity = unrelated;
        rejects([&] { bad_owner.validate(&f.device); });
        std::weak_ptr<void> live;
        {
            auto lease = f.lease();
            lease.owner = unrelated; lease.allocation_identity = unrelated;
            live = unrelated; unrelated.reset();
            lease.validate(&f.device);
            check(!live.expired(), "lease failed to retain original allocation");
        }
        check(live.expired(), "lease unexpectedly retained allocation after destruction");
        std::cout << "PASS native ANE bridge CPU contracts: retained surfaces/events, provenance, move-only permits, split admission, sticky retirement, shared deadlines; device_created=0\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
