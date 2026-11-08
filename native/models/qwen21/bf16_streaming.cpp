#include "bf16_streaming.hpp"
#include "diagnostic_options.hpp"
#include "../../backends/convrot_rotation.hpp"
#include "../../backends/ane_memory.hpp"
#include "../../runtime/streaming/gguf_storage.hpp"
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <unistd.h>

namespace tc::qwen21 {
namespace {
using streaming::gguf_storage::capacity_upper;
constexpr uint64_t system_reserve = uint64_t(4) << 30;
constexpr int convrot_group = 128;

uint64_t float_layer_capacity(const streaming::BlockSpec &block) {
    uint64_t capacity = 0;
    for (const auto &field : block.fields) {
        if (field.materialization->format != "I8") continue;
        const auto plane_bytes = field.bytes / convrot_group * sizeof(float);
        capacity = gguf::checked_add(capacity, capacity_upper(field.bytes));
        capacity = gguf::checked_add(capacity, capacity_upper(plane_bytes));
        capacity = gguf::checked_add(capacity, capacity_upper(plane_bytes));
    }
    return capacity;
}

Tensor allocate_float_backing(const mx::Shape &shape, mx::Dtype type, uint64_t bytes) {
    const auto capacity = capacity_upper(bytes);
    ane::require_streaming_memory_headroom(mx::get_active_memory(), capacity,
        "Qwen ConvRot coefficient-bank admission declined");
    Tensor array(mx::allocator::malloc(bytes), shape, type);
    const auto actual = mx::allocator::allocator().size(array.data_shared_ptr()->buffer);
    require(actual >= bytes && actual <= capacity, "Qwen ConvRot packing backing exceeds admitted capacity");
    return array;
}
} // namespace

bool bf16_streaming_enabled() {
    const auto *value=std::getenv("TURBOCIDER_QWEN21_BF16_STREAMING");
    require(binary_option_or_unset(value),"Qwen BF16 streaming requires0 or1");return option_enabled(value);
}
uint64_t bf16_stream_managed_capacity(const Bf16StreamPlan &plan) {
    const auto &layout=plan.layout.stages.at(0);
    auto capacity=gguf::checked_add(layout.resident_bytes,layout.peak_pool_bytes);
    if(plan.mode==StreamWeightMode::ConvRotFloat)
        capacity=gguf::checked_add(capacity,float_layer_capacity(plan.descriptor.stages.at(0).blocks.front()));
    return capacity;
}
Bf16StreamPlan bf16_stream_cache_plan(Bf16StreamPlan base,uint64_t budget,
        const std::function<bool(uint64_t)> &admit) {
    require(base.prefix==0 && base.descriptor.stages.size()==1 &&
        base.descriptor.stages[0].id=="denoiser" && base.mode!=StreamWeightMode::Bf16,
        "Qwen weight caching requires the original zero-prefix streamed denoiser plan");
    const auto minimum=bf16_stream_managed_capacity(base);
    require(minimum<=budget,"Qwen streamed weight budget is below the required weights and two refill buffers; minimum MiB="+
        std::to_string((minimum+((uint64_t(1)<<20)-1))>>20));
    auto selected=base;
    uint32_t target=0;
    for(uint32_t prefix=1;prefix<=base.layers-2;++prefix) {
        auto candidate=base;
        auto &stage=candidate.descriptor.stages[0];
        for(uint32_t layer=0;layer<prefix;++layer) for(auto field:stage.blocks[layer].fields) {
            field.name="transformer_blocks."+std::to_string(layer)+"."+field.name;
            stage.resident_fields.push_back(std::move(field));
        }
        stage.blocks.erase(stage.blocks.begin(),stage.blocks.begin()+prefix);
        candidate.prefix=prefix;
        candidate.descriptor.workload["resident_prefix"]=std::to_string(prefix);
        candidate.descriptor.backend_revision+="-cached-prefix-v1";
        StreamingConfig config;config.enabled=true;config.schema_version=1;config.selection="manual";config.retention="request";
        config.stages[stage.id]={"streamed",1,2,0,1,1};
        candidate.layout=streaming::compile_layout(config,candidate.descriptor);
        const auto capacity=bf16_stream_managed_capacity(candidate);
        if(capacity>budget)break;
        target=prefix;
        if(!admit || admit(capacity))selected=std::move(candidate);
    }
    selected.cache_budget_prefix=target;
    selected.cache_limited_by_memory=selected.prefix<target;
    if(selected.prefix)selected.minimum_plan=std::make_shared<const Bf16StreamPlan>(std::move(base));
    return selected;
}
Bf16StreamPlan bf16_stream_recheck_cache_plan(Bf16StreamPlan plan,
        const ane::StreamingMemoryObservation &observation) {
    if (!plan.minimum_plan) return plan;
    const auto admission = ane::admit_streaming_memory(observation,
        {system_reserve, observation.memory.physical_bytes}, bf16_stream_managed_capacity(plan));
    if (admission.allowed() && observation.pressure == ane::StreamingMemoryPressure::Normal) return plan;
    auto minimum = *plan.minimum_plan;
    minimum.cache_budget_prefix = plan.cache_budget_prefix;
    minimum.cache_limited_by_memory = true;
    return minimum;
}
Bf16StreamBank::Bf16StreamBank(Bf16StreamPlan plan,uint64_t budget,std::atomic<bool> &cancel)
    :plan_(bf16_stream_recheck_cache_plan(std::move(plan), ane::observe_streaming_memory(mx::get_active_memory()))),pager_(plan_.lease,
        plan_.minimum_plan ? plan_.minimum_plan->descriptor : plan_.descriptor,
        plan_.minimum_plan ? plan_.minimum_plan->descriptor.stages[0] : plan_.descriptor.stages[0],
        plan_.minimum_plan ? plan_.minimum_plan->layout.stages[0] : plan_.layout.stages[0]),cancel_(cancel) {
    const auto &layout=plan_.layout.stages[0];
    const auto &stage = plan_.descriptor.stages[0];
    const bool convrot_request = plan_.mode != StreamWeightMode::Bf16;
    const bool convrot = plan_.mode == StreamWeightMode::ConvRotFloat || plan_.mode == StreamWeightMode::ConvRotW8A8;
    const bool tensorops = plan_.mode == StreamWeightMode::ConvRotW8A8;
    require(!convrot_request || stage.id=="denoiser" || plan_.prefix==0,
        "Qwen ConvRot encoder bank requires zero resident layer prefix");
    const auto capacity = gguf::checked_add(layout.resident_bytes, layout.peak_pool_bytes);
    const auto derived = plan_.mode == StreamWeightMode::ConvRotFloat ? float_layer_capacity(stage.blocks.front()) : 0;
    const auto managed = gguf::checked_add(capacity, derived);
    require(execution_layout().peak_pool_bytes<=layout.peak_pool_bytes,
        "Qwen cached layers require the same reusable refill capacity as the suffix");
    require(managed <= budget, "Qwen streaming managed weight budget floor");
    if (convrot_request) {
        ane::require_streaming_memory_headroom(mx::get_active_memory(), managed,
            "Qwen ConvRot streaming physical admission declined");
    } else {
        uint64_t largest_field=0;
        for(const auto &field : stage.resident_fields)
            largest_field = std::max(largest_field, capacity_upper(field.bytes));
        for(const auto &block : stage.blocks) for(const auto &field : block.fields)
            largest_field = std::max(largest_field, capacity_upper(field.bytes));
        const auto observed=ane::observe_runtime_memory(mx::get_active_memory());
        const auto admission=ane::admit_memory(observed,{system_reserve,observed.physical_bytes},0,largest_field);
        require(admission.allowed(),"Qwen BF16 streaming physical admission declined: "+ane::memory_denial_reason(admission.denial,observed));
    }
    pager_.use_exact_allocations(convrot_request);
    pager_.load_resident(resident_,&cancel_);plan_.lease->revalidate_after_drain();
    if (tensorops) resident_.set_convrot_w8a8(true);
    pager_.create_pool(execution_layout().pools[0]);
    if (derived) {
        // Each block in this pool shares the same validated geometry. Reserve
        // one reusable exact Q8 bank for the current layer. The raw embedding
        // is gathered directly and never duplicated as a packed vocabulary.
        streaming::gguf_storage::ExactCapacityCacheScope exact;
        for (const auto &field : stage.blocks.front().fields) {
            if (field.materialization->format != "I8") continue;
            checkpoint(cancel_);
            const auto &shape = field.materialization->shape;
            const int rows = int(shape[0]), columns = int(shape[1]);
            const uint64_t plane_bytes = field.bytes / convrot_group * sizeof(float);
            const mx::Shape coefficients{rows, columns / convrot_group};
            float_projections_.push_back({field.name.substr(0, field.name.size() - std::string_view(".weight").size()),
                allocate_float_backing({rows, columns / 4}, mx::uint32, field.bytes),
                allocate_float_backing(coefficients, mx::float32, plane_bytes),
                allocate_float_backing(coefficients, mx::float32, plane_bytes)});
        }
    }
    if(plan_.minimum_plan) {
        cache_plan_=plan_;
        auto &cache_stage=cache_plan_->descriptor.stages[0];
        std::erase_if(cache_stage.resident_fields,[](const auto &field) {
            return !field.name.starts_with("transformer_blocks.");
        });
        StreamingConfig config;config.enabled=true;config.schema_version=1;config.selection="manual";config.retention="request";
        config.stages[cache_stage.id]={"streamed",1,2,0,1,1};
        cache_plan_->layout=streaming::compile_layout(config,cache_plan_->descriptor);
        cache_pager_=std::make_unique<streaming::MlxWeightPager>(plan_.lease,cache_plan_->descriptor,
            cache_stage,cache_plan_->layout.stages[0]);
        cache_pager_->use_exact_allocations(true);
        try {
            cache_pager_->load_resident(cached_,&cancel_);
            if(tensorops)cached_.set_convrot_w8a8(true);
        } catch(const std::invalid_argument &error) {
            if(std::string_view(error.what()).find("incremental ConvRot source admission declined")==std::string_view::npos)throw;
            retired_cache_metrics_=cache_pager_->metrics();
            cached_.clear();cache_pager_.reset();cache_plan_.reset();
            plan_.prefix=0;plan_.cache_limited_by_memory=true;
            ++metrics_.cache_evictions;
        }
    }
    std::vector<uint64_t> capacities;for(const auto &slot:layout.pools[0].slots)capacities.push_back(slot.capacity_bytes);
    safety_=std::make_unique<streaming::SlotSafetyTracker>(layout.pools[0].id,plan_.lease->generation(),capacities);
    io_=std::make_unique<streaming::IoExecutor>(1,2,mailbox_);
    const auto &file=plan_.lease->descriptor().files[0];
    metrics_.source_sha256=file.content_digest;metrics_.source_file_bytes=file.bytes;
    metrics_.layout_digest=plan_.minimum_plan ? plan_.minimum_plan->layout.digest : plan_.layout.digest;
    if(cache_plan_)metrics_.cache_layout_digest=cache_plan_->layout.digest;
    metrics_.verification_bytes=plan_.lease->verification_bytes_read();metrics_.verification_cache_hits=plan_.lease->verification_cache_hits();
    metrics_.verification_seconds=plan_.verification_seconds;metrics_.managed_weight_capacity_bytes=capacity;metrics_.managed_weight_budget_bytes=budget;
    metrics_.derived_weight_capacity_bytes=derived;
    metrics_.source_precision=convrot ? "original_i8_convrot" : "original_bf16";
    metrics_.matmul_mode=tensorops ? "int8_tensorops_w8a8" : convrot ? "q8_g128_float_a" : "bf16";
    metrics_.layers=plan_.layers;metrics_.prefix=plan_.prefix;metrics_.slots=2;
    metrics_.cache_budget_prefix=plan_.cache_budget_prefix;
    metrics_.cache_limited_by_memory=plan_.cache_limited_by_memory;
    metrics_.final_prefix=plan_.prefix;
}
Bf16StreamBank::~Bf16StreamBank() {
    if(io_)io_->shutdown_and_join();
    // Includes any partly submitted graph on exception/cancel. A failed drain
    // cannot safely release reusable storage; terminate instead of use-after-free.
    try {mx::synchronize();}catch(...){std::terminate();}
}
int Bf16StreamBank::embedding_vocabulary() const {
    require(plan_.embedding.has_value(), "Qwen sparse embedding requires a ConvRot encoder source");
    return int(plan_.embedding->codes.materialization->shape.at(0));
}
Tensor Bf16StreamBank::embedding_rows(const Tokens &tokens) {
    checkpoint(cancel_);
    require(!finished_ && !pass_live_ && !layer_live_ && metrics_.gathered_embedding_rows == 0,
        "Qwen sparse embedding gather must precede the encoder pass exactly once");
    const int vocabulary = embedding_vocabulary();
    require(!tokens.ids.empty() && tokens.ids.size() <= size_t(Tokenizer::qwen21_limit),
        "Qwen sparse embedding requires a bounded nonempty token sequence");
    for (int id : tokens.ids)
        require(id >= 0 && id < vocabulary, "Qwen21 token ID out of vocabulary");
    const auto &source = *plan_.embedding;
    const int rows = int(tokens.ids.size()), width = int(source.codes.materialization->shape.at(1));
    const uint64_t code_bytes = uint64_t(rows) * width, scale_bytes = uint64_t(rows) * sizeof(float);
    const auto code_capacity = capacity_upper(code_bytes), scale_capacity = capacity_upper(scale_bytes);
    require(gguf::checked_add(metrics_.managed_weight_capacity_bytes,
        gguf::checked_add(metrics_.derived_weight_capacity_bytes,gguf::checked_add(code_capacity,scale_capacity))) <=
        metrics_.managed_weight_budget_bytes, "Qwen sparse embedding exceeds managed weight budget");
    streaming::gguf_storage::ExactCapacityCacheScope exact;
    auto codes = allocate_float_backing({rows,width},mx::int8,code_bytes);
    auto scales = allocate_float_backing({rows,1},mx::float32,scale_bytes);
    plan_.lease->revalidate_after_drain();
    const auto &file = plan_.lease->descriptor().files.at(0);
    auto fd = plan_.lease->duplicate_fd(file.logical_id);
    const uint64_t code_offset = source.codes.materialization->reads.at(0).offset;
    const uint64_t scale_offset = source.scales.materialization->reads.at(0).offset;
    auto read = [&](void *destination,uint64_t bytes,uint64_t offset) {
        uint64_t done=0;
        while(done<bytes) {
            checkpoint(cancel_);
            const auto count=::pread(fd.get(),static_cast<char *>(destination)+done,size_t(bytes-done),off_t(offset+done));
            if(count<0 && errno==EINTR)continue;
            require(count>0,"Qwen sparse embedding source read failed");
            done+=uint64_t(count);
        }
    };
    for(int row=0;row<rows;++row) {
        const uint64_t id=uint64_t(tokens.ids[size_t(row)]);
        read(codes.data<int8_t>()+uint64_t(row)*width,width,code_offset+id*width);
        read(scales.data<float>()+row,sizeof(float),scale_offset+id*sizeof(float));
        require(std::isfinite(scales.data<float>()[row]),"nonfinite Qwen sparse embedding scale");
    }
    plan_.lease->revalidate_after_drain();
    auto rotated=mx::astype(codes,mx::float32)*scales;
    auto embeddings=mx::astype(convrot_kernel::rotate(rotated,convrot_kernel::Rotation::SimdRegister),mx::bfloat16);
    mx::eval(embeddings);
    // Keep an admitted upper bound including the transient gathered codes.
    metrics_.managed_weight_capacity_bytes+=code_capacity+scale_capacity;
    metrics_.gathered_embedding_rows=uint64_t(rows);
    metrics_.gathered_embedding_source_bytes=code_bytes+scale_bytes;
    return embeddings;
}
void Bf16StreamBank::schedule(uint32_t layer) {
    if(layer>=plan_.layers)return;
    const auto &group=execution_layout().groups.at(plan_.minimum_plan ? layer : layer-plan_.prefix);
    if(pending_[group.slot]) {
        require(pending_[group.slot]->item.group==group.id && pending_[group.slot]->item.pass==pass_,
            "Qwen BF16 pending slot belongs to another layer/pass");return;
    }
    require(safety_->state(group.slot)==streaming::ContentState::Vacant,"Qwen weight stream source slot still has readers");
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
    metrics_.prefix_per_pass.push_back(plan_.prefix);
    if(!plan_.prefix)schedule(0);
}
void Bf16StreamBank::bind_float_layer(uint32_t layer) {
    const std::string prefix = (plan_.descriptor.stages[0].id == "encoder" ? "model.layers." : "transformer_blocks.") +
        std::to_string(layer) + ".";
    std::vector<std::string> keys;
    std::vector<Tensor> arrays;
    for (const auto &key : current_.sorted_keys()) {
        const auto &tensor = current_.at(key);
        if (tensor.dtype() == mx::int8 || key.ends_with(".weight_scale")) continue;
        keys.push_back(key);
        arrays.push_back(tensor);
    }
    for (auto &projection : float_projections_) {
        checkpoint(cancel_);
        const auto name = prefix + projection.name;
        const auto &source = current_.at(name + ".weight");
        const auto &row_scales = current_.at(name + ".weight_scale");
        const int groups = projection.scales.shape(1);
        // Signed q -> q + 128 for MLX affine Q8, exactly equivalent to
        // the original signed code with F32 bias -128 * row scale.
        const auto *from = source.data<uint8_t>();
        auto *to = projection.codes.data<uint8_t>();
        for (size_t i = 0; i < source.nbytes(); ++i) to[i] = from[i] ^ uint8_t(0x80);
        for (int row = 0; row < source.shape(0); ++row) {
            const auto scale = row_scales.data<float>()[row];
            const auto offset = size_t(row) * groups;
            std::fill_n(projection.scales.data<float>() + offset, groups, scale);
            std::fill_n(projection.biases.data<float>() + offset, groups, -128.f * scale);
        }
        keys.insert(keys.end(), {name + ".weight", name + ".scales", name + ".biases"});
        arrays.insert(arrays.end(), {projection.codes, projection.scales, projection.biases});
    }
    current_.bind_arrays(keys, arrays);
}

const Weights &Bf16StreamBank::acquire(int layer) {
    checkpoint(cancel_);require(pass_live_ && layer>=0 && uint32_t(layer)==next_layer_ && !layer_live_,
        "Qwen weight stream layer acquire out of order");
    if(uint32_t(layer)<plan_.prefix) {
        if(uint32_t(layer)+1==plan_.prefix)schedule(plan_.prefix);
        layer_live_=true;
        metrics_.peak_live_layer_owners=1;
        if(plan_.mode==StreamWeightMode::ConvRotFloat) {
            current_=plan_.minimum_plan ? cached_ : resident_;
            bind_float_layer(uint32_t(layer));
            return current_;
        }
        return plan_.minimum_plan ? cached_ : resident_;
    }
    schedule(uint32_t(layer));const auto &group=execution_layout().groups.at(plan_.minimum_plan ? uint32_t(layer) : uint32_t(layer)-plan_.prefix);
    const auto ticket=*pending_[group.slot];const auto begin=Clock::now();
    while(!safety_->ready(ticket)) {checkpoint(cancel_);consume();if(!safety_->ready(ticket))mailbox_.wait_for(std::chrono::milliseconds(10));}
    metrics_.wait_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
    plan_.lease->revalidate_after_drain();safety_->begin_use(ticket);current_ticket_=ticket;
    current_=pager_.bind(group,ticket);
    schedule(uint32_t(layer)+1);
    if (plan_.mode == StreamWeightMode::ConvRotW8A8) current_.set_convrot_w8a8(true);
    if (plan_.mode == StreamWeightMode::ConvRotFloat) bind_float_layer(uint32_t(layer));
    layer_live_=true;
    metrics_.peak_live_layer_owners=1;
    return current_;
}
void Bf16StreamBank::retire(int layer) {
    require(pass_live_ && layer_live_ && layer>=0 && uint32_t(layer)==next_layer_,"Qwen weight stream retirement out of order");
    if(uint32_t(layer)>=plan_.prefix) {
        require(current_ticket_.has_value(),"Qwen BF16 retirement missing current ticket");
        const tc_stream_reader_fence_v1 fence{1,metrics_.reader_fences+1};
        safety_->seal_readers(*current_ticket_,{&fence,1});++metrics_.reader_fences;
        safety_->complete_reader(*current_ticket_,fence);++metrics_.completed_reader_fences;
        pending_[current_ticket_->slot].reset();current_ticket_.reset();current_.clear();++metrics_.completed_streamed_layers;
    }else {current_.clear();++metrics_.completed_prefix_layers;}
    layer_live_=false;
    ++metrics_.completed_layers;++next_layer_;
}
void Bf16StreamBank::finish_pass() {
    consume();require(pass_live_ && next_layer_==plan_.layers && !layer_live_ && !current_ticket_ && safety_->quiescent(),
        "Qwen BF16 pass ended with incomplete layers/readers");
    plan_.lease->revalidate_after_drain();pass_live_=false;++metrics_.completed_passes;
}
void Bf16StreamBank::finish() {
    require(!finished_ && !pass_live_ && metrics_.completed_passes==plan_.descriptor.stages[0].pass_count,
        "Qwen BF16 missing complete pass coverage");
    io_->shutdown_and_join();consume();mx::synchronize();plan_.lease->revalidate_after_drain();
    require(safety_->quiescent() && metrics_.completed_reader_fences==metrics_.reader_fences &&
        metrics_.fills==metrics_.completed_streamed_layers &&
        metrics_.completed_prefix_layers+metrics_.completed_streamed_layers==uint64_t(plan_.layers)*metrics_.completed_passes,
        "Qwen BF16 incomplete source/reader coverage");
    finished_=true;metrics_.drained=true;
}
QwenBf16StreamStageMetrics Bf16StreamBank::metrics() const {
    require(finished_,"Qwen BF16 final metrics require drain");auto result=metrics_;const auto &pager=pager_.metrics();
    result.resident_source_bytes=pager.resident_bytes_loaded;result.streamed_source_bytes=pager.streamed_bytes_loaded;
    const auto cache=cache_pager_ ? cache_pager_->metrics() : retired_cache_metrics_;
    result.resident_source_bytes+=cache.resident_bytes_loaded;
    result.slot_arrays=pager.slot_arrays_allocated+cache.slot_arrays_allocated;
    result.resident_load_seconds=pager.resident_load_seconds+cache.resident_load_seconds;
    result.streamed_read_seconds=pager.streamed_load_seconds;return result;
}
const streaming::StageLayout &Bf16StreamBank::execution_layout() const {
    return plan_.minimum_plan ? plan_.minimum_plan->layout.stages[0] : plan_.layout.stages[0];
}
uint64_t Bf16StreamBank::release_fixed_fields(const std::vector<std::string> &names) {
    checkpoint(cancel_);
    require(!finished_ && !pass_live_ && !layer_live_ && !current_ticket_ &&
        !pending_[0] && !pending_[1] && safety_->quiescent() &&
        metrics_.reader_fences==metrics_.completed_reader_fences,
        "Qwen fixed-field release requires a drained pass boundary");
    const auto &stage=plan_.minimum_plan ? plan_.minimum_plan->descriptor.stages[0] :
        plan_.descriptor.stages[0];
    for(const auto &name:names) {
        require(!name.starts_with("transformer_blocks.") && !name.starts_with("model.layers."),
            "Qwen fixed-field release cannot retire a layer field");
        const auto found=std::find_if(stage.resident_fields.begin(),stage.resident_fields.end(),
            [&](const auto &field) {
                return field.materialization && field.materialization->reads.size()==1 &&
                    field.materialization->reads.front().tensor==name;
            });
        require(found!=stage.resident_fields.end(),"Qwen fixed-field release requires an original fixed field: "+name);
    }
    // The pager drains the last submitted GPU readers and revalidates the
    // held-fd source before atomically retiring its and resident_'s owners.
    // Optional-cache eviction never rebuilds resident_, so retired fields
    // cannot be brought back by later prefix or slot transitions.
    return pager_.release_resident_fields(resident_,names,&cancel_);
}
bool Bf16StreamBank::evict_cached_layers() {
    require(!finished_ && !pass_live_ && !layer_live_ && !current_ticket_ && safety_->quiescent(),
        "Qwen cache eviction requires a drained pass boundary");
    if(!cache_pager_)return false;
    checkpoint(cancel_);mx::synchronize();plan_.lease->revalidate_after_drain();
    retired_cache_metrics_=cache_pager_->metrics();
    cached_.clear();cache_pager_.reset();cache_plan_.reset();
    plan_.prefix=0;metrics_.final_prefix=0;metrics_.cache_limited_by_memory=true;
    ++metrics_.cache_evictions;
    mx::clear_cache();
    return true;
}
}
