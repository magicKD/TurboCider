#include "bf16_streaming.hpp"
#include "diagnostic_options.hpp"
#include "../../backends/ane_memory.hpp"
#include <cstdlib>

namespace tc::qwen21 {
bool bf16_streaming_enabled() {
    const auto *value=std::getenv("TURBOCIDER_QWEN21_BF16_STREAMING");
    require(binary_option_or_unset(value),"Qwen BF16 streaming requires0 or1");return option_enabled(value);
}
Bf16StreamBank::Bf16StreamBank(Bf16StreamPlan plan,uint64_t budget,std::atomic<bool> &cancel)
    :plan_(std::move(plan)),pager_(plan_.lease,plan_.descriptor,plan_.descriptor.stages[0],plan_.layout.stages[0]),cancel_(cancel) {
    const auto &layout=plan_.layout.stages[0];
    require(layout.resident_bytes<=UINT64_MAX-layout.peak_pool_bytes,"Qwen BF16 managed capacity overflow");
    const auto capacity=layout.resident_bytes+layout.peak_pool_bytes;
    require(capacity<=budget,"Qwen BF16 streaming managed weight budget floor");
    uint64_t largest_field=0;
    for(const auto &field:plan_.descriptor.stages[0].resident_fields)
        largest_field=std::max(largest_field,(field.bytes+16383)&~uint64_t(16383));
    for(const auto &block:plan_.descriptor.stages[0].blocks)for(const auto &field:block.fields)
        largest_field=std::max(largest_field,(field.bytes+16383)&~uint64_t(16383));
    const auto observed=ane::observe_runtime_memory(mx::get_active_memory());
    const auto admission=ane::admit_memory(observed,{uint64_t(4)<<30,observed.physical_bytes},0,largest_field);
    require(admission.allowed(),"Qwen BF16 streaming physical admission declined: "+ane::memory_denial_reason(admission.denial,observed));
    pager_.use_exact_allocations();pager_.load_resident(resident_,&cancel_);plan_.lease->revalidate_after_drain();
    pager_.create_pool(layout.pools[0]);
    std::vector<uint64_t> capacities;for(const auto &slot:layout.pools[0].slots)capacities.push_back(slot.capacity_bytes);
    safety_=std::make_unique<streaming::SlotSafetyTracker>(layout.pools[0].id,plan_.lease->generation(),capacities);
    io_=std::make_unique<streaming::IoExecutor>(1,2,mailbox_);
    const auto &file=plan_.lease->descriptor().files[0];
    metrics_.source_sha256=file.content_digest;metrics_.source_file_bytes=file.bytes;metrics_.layout_digest=plan_.layout.digest;
    metrics_.verification_bytes=plan_.lease->verification_bytes_read();metrics_.verification_cache_hits=plan_.lease->verification_cache_hits();
    metrics_.verification_seconds=plan_.verification_seconds;metrics_.managed_weight_capacity_bytes=capacity;metrics_.managed_weight_budget_bytes=budget;
    metrics_.layers=plan_.layers;metrics_.prefix=plan_.prefix;metrics_.slots=2;
}
Bf16StreamBank::~Bf16StreamBank() {
    if(io_)io_->shutdown_and_join();
    // Includes any partly submitted graph on exception/cancel. A failed drain
    // cannot safely release reusable storage; terminate instead of use-after-free.
    try {mx::synchronize();}catch(...){std::terminate();}
}
void Bf16StreamBank::schedule(uint32_t layer) {
    if(layer>=plan_.layers)return;
    const auto &group=plan_.layout.stages[0].groups.at(layer-plan_.prefix);
    if(pending_[group.slot]) {
        require(pending_[group.slot]->item.group==group.id && pending_[group.slot]->item.pass==pass_,
            "Qwen BF16 pending slot belongs to another layer/pass");return;
    }
    require(safety_->state(group.slot)==streaming::ContentState::Vacant,"Qwen BF16 source slot still has readers");
    auto ticket=safety_->begin_fill(group.slot,{0,pass_,pass_,group.id},group.bytes);
    jobs_[group.slot]={this,&group};pending_[group.slot]=ticket;
    streaming::FillJob job{ticket,&jobs_[group.slot],[](void *raw,const tc_stream_slot_ticket_v1 *ticket,
        const std::atomic<bool> *shutdown,uint64_t *bytes) {
        auto &job=*static_cast<Job *>(raw);
        try {
            if(shutdown->load() || job.bank->cancel_.load())return -1;
            *bytes=job.bank->pager_.fill(*job.group,*ticket,&job.bank->cancel_);return 0;
        }catch(...){return -1;}
    }};
    require(io_->enqueue(job),"Qwen BF16 bounded refill queue backpressure");
}
void Bf16StreamBank::consume() {
    require(!mailbox_.overflowed(),"Qwen BF16 completion mailbox overflow");
    tc_stream_completion_v1 completion;
    while(mailbox_.pop(completion)) {
        require(completion.kind==TC_STREAM_FILL_COMPLETE && completion.status==0,
            "Qwen BF16 refill failed/cancelled; no source bank published");
        safety_->accept_ready(completion.ticket,completion.bytes);++metrics_.fills;
    }
}
void Bf16StreamBank::begin_pass(uint32_t pass) {
    checkpoint(cancel_);require(!finished_ && !pass_live_ && pass==metrics_.completed_passes &&
        pass<plan_.descriptor.stages[0].pass_count && safety_->quiescent(),"Qwen BF16 invalid pass boundary");
    plan_.lease->revalidate_after_drain();pass_=pass;next_layer_=0;pass_live_=true;
    if(!plan_.prefix)schedule(0);
}
const Weights &Bf16StreamBank::acquire(int layer) {
    checkpoint(cancel_);require(pass_live_ && layer>=0 && uint32_t(layer)==next_layer_ && !current_ticket_,
        "Qwen BF16 layer acquire out of order");
    if(uint32_t(layer)<plan_.prefix) {
        if(uint32_t(layer)+1==plan_.prefix)schedule(plan_.prefix);
        return resident_;
    }
    schedule(uint32_t(layer));const auto &group=plan_.layout.stages[0].groups.at(uint32_t(layer)-plan_.prefix);
    const auto ticket=*pending_[group.slot];const auto begin=Clock::now();
    while(!safety_->ready(ticket)) {checkpoint(cancel_);consume();if(!safety_->ready(ticket))mailbox_.wait_for(std::chrono::milliseconds(10));}
    metrics_.wait_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
    plan_.lease->revalidate_after_drain();safety_->begin_use(ticket);current_ticket_=ticket;
    current_=pager_.bind(group,ticket);schedule(uint32_t(layer)+1);return current_;
}
void Bf16StreamBank::retire(int layer) {
    require(pass_live_ && layer>=0 && uint32_t(layer)==next_layer_,"Qwen BF16 retirement out of order");
    if(uint32_t(layer)>=plan_.prefix) {
        require(current_ticket_.has_value(),"Qwen BF16 retirement missing current ticket");
        const tc_stream_reader_fence_v1 fence{1,metrics_.reader_fences+1};
        safety_->seal_readers(*current_ticket_,{&fence,1});++metrics_.reader_fences;
        safety_->complete_reader(*current_ticket_,fence);++metrics_.completed_reader_fences;
        pending_[current_ticket_->slot].reset();current_ticket_.reset();current_.clear();++metrics_.completed_streamed_layers;
    }else ++metrics_.completed_prefix_layers;
    ++metrics_.completed_layers;++next_layer_;
}
void Bf16StreamBank::finish_pass() {
    consume();require(pass_live_ && next_layer_==plan_.layers && !current_ticket_ && safety_->quiescent(),
        "Qwen BF16 pass ended with incomplete layers/readers");
    plan_.lease->revalidate_after_drain();pass_live_=false;++metrics_.completed_passes;
}
void Bf16StreamBank::finish() {
    require(!finished_ && !pass_live_ && metrics_.completed_passes==plan_.descriptor.stages[0].pass_count,
        "Qwen BF16 missing complete pass coverage");
    io_->shutdown_and_join();consume();mx::synchronize();plan_.lease->revalidate_after_drain();
    require(safety_->quiescent() && metrics_.completed_reader_fences==metrics_.reader_fences &&
        metrics_.fills==uint64_t(plan_.layers-plan_.prefix)*metrics_.completed_passes,
        "Qwen BF16 incomplete source/reader coverage");
    finished_=true;metrics_.drained=true;
}
QwenBf16StreamStageMetrics Bf16StreamBank::metrics() const {
    require(finished_,"Qwen BF16 final metrics require drain");auto result=metrics_;const auto &pager=pager_.metrics();
    result.resident_source_bytes=pager.resident_bytes_loaded;result.streamed_source_bytes=pager.streamed_bytes_loaded;
    result.slot_arrays=pager.slot_arrays_allocated;result.resident_load_seconds=pager.resident_load_seconds;
    result.streamed_read_seconds=pager.streamed_load_seconds;return result;
}
}
