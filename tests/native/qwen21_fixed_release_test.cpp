#include "models/qwen21/bf16_streaming.hpp"
#include "turbocider/turbocider.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

using namespace tc;
using namespace tc::qwen21;

namespace {
const std::vector<std::string> time_fields{
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "modulation.1.weight", "norm_out.linear.weight"};
const std::string retained_field="txt_in.in_layer.weight";
constexpr uint32_t layers=4,passes=3;

template<class Function>
void rejects(Function &&function,std::string_view reason) {
    try { function(); }
    catch(const std::exception &error) {
        require(std::string_view(error.what()).find(reason)!=std::string_view::npos,
            std::string("wrong release rejection: ")+error.what());
        return;
    }
    throw std::runtime_error("invalid fixed release was accepted");
}

uint64_t capacity(const Tensor &array) {
    return mx::allocator::allocator().size(array.buffer());
}
std::vector<std::weak_ptr<Tensor::Data>> weak_fields(const Weights &weights) {
    std::vector<std::weak_ptr<Tensor::Data>> result;
    for(const auto &name:time_fields)result.emplace_back(weights.at(name).data_shared_ptr());
    return result;
}
uint64_t time_capacity(const Weights &weights) {
    uint64_t result=0;
    for(const auto &name:time_fields)result+=capacity(weights.at(name));
    return result;
}
void retained(const Weights &weights,std::uintptr_t expected) {
    require(weights.at(retained_field).id()==expected,"release changed a retained fixed binding");
}
void all_present(const Weights &weights) {
    for(const auto &name:time_fields)require(weights.has(name),"failed release partly removed fixed owners");
}
void all_retired(const Weights &weights,const std::vector<std::weak_ptr<Tensor::Data>> &weak) {
    for(size_t index=0;index<time_fields.size();++index)
        require(!weights.has(time_fields[index]) && weak[index].expired(),
            "retired fixed source ownership survived or was restored");
}
struct HeldSource {
    streaming::OwnedSourceFd fd;
    struct stat before{};
    unsigned char byte=0;
    explicit HeldSource(const std::shared_ptr<const streaming::SourceLease> &lease)
        :fd(lease->duplicate_fd(lease->descriptor().files.at(0).logical_id)) {
        require(::fstat(fd.get(),&before)==0 && ::lseek(fd.get(),7,SEEK_SET)==7 &&
            ::pread(fd.get(),&byte,1,0)==1,"cannot capture held source witness");
    }
    void unchanged() const {
        struct stat after{};unsigned char observed=0;
        require(::fcntl(fd.get(),F_GETFD)>=0 && ::lseek(fd.get(),0,SEEK_CUR)==7 &&
            ::fstat(fd.get(),&after)==0 && ::pread(fd.get(),&observed,1,0)==1 && observed==byte &&
            before.st_dev==after.st_dev && before.st_ino==after.st_ino && before.st_size==after.st_size &&
            before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec &&
            before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec,
            "release changed the held source generation, bytes, descriptor or offset");
    }
};

void pager_cases(const std::filesystem::path &path) {
    std::atomic<bool> cancel{false};
    auto plan=bf16_stream_plan(path,false,layers,0,passes,cancel,StreamWeightMode::ConvRotFloat);
    const auto &stage=plan.descriptor.stages.at(0);
    const auto &layout=plan.layout.stages.at(0);
    HeldSource held(plan.lease);
    streaming::MlxWeightPager pager(plan.lease,plan.descriptor,stage,layout);
    pager.use_exact_allocations(true);
    Weights weights;
    pager.load_resident(weights,&cancel);
    pager.create_pool(layout.pools.at(0));
    weights.set_metal_convrot(true);
    const auto *owner=&weights;
    const auto kept=weights.at(retained_field).id();
    const auto initial_metrics=pager.metrics();
    auto weak=weak_fields(weights);
    require(pager.release_resident_fields(weights,{},&cancel)==0,"empty release changed capacity");
    rejects([&]{pager.release_resident_fields(weights,{time_fields[0],"unknown.weight"},&cancel);},"unknown or retired");
    all_present(weights);
    rejects([&]{pager.release_resident_fields(weights,{time_fields[0],time_fields[0]},&cancel);},"duplicate");
    all_present(weights);
    rejects([&]{pager.release_resident_fields(weights,{time_fields[0],"transformer_blocks.0.attn.to_q.weight"},&cancel);},"unknown or retired");
    all_present(weights);
    {
        auto other=weights;
        rejects([&]{pager.release_resident_fields(other,time_fields,&cancel);},"bound owner");
        all_present(other);all_present(weights);
    }
    {
        const auto keys=weights.sorted_keys();
        std::vector<Tensor> originals;
        for(const auto &key:keys)originals.push_back(weights.at(key));
        auto rebound=originals;
        const auto second=std::find(keys.begin(),keys.end(),time_fields[1])-keys.begin();
        rebound.at(size_t(second))=mx::zeros(originals.at(size_t(second)).shape(),mx::bfloat16);
        mx::eval(rebound.at(size_t(second)));
        weights.bind_arrays(keys,rebound);
        const auto first_id=weights.at(time_fields[0]).id();
        const auto second_id=weights.at(time_fields[1]).id();
        rejects([&]{pager.release_resident_fields(weights,{time_fields[0],time_fields[1]},&cancel);},"binding differs");
        require(weights.at(time_fields[0]).id()==first_id && weights.at(time_fields[1]).id()==second_id,
            "mismatched selection partly changed bindings");
        weights.bind_arrays(keys,originals);
        weights.set_metal_convrot(true);
    }
    cancel.store(true);
    rejects([&]{pager.release_resident_fields(weights,time_fields,&cancel);},"cancelled");
    cancel.store(false);all_present(weights);held.unchanged();

    const auto bytes=time_capacity(weights),alias_bytes=capacity(weights.at(time_fields[0]));
    std::optional<Tensor> external(weights.at(time_fields[0]));
    mx::synchronize();mx::clear_cache();
    const auto before=mx::get_active_memory();
    const auto retired=pager.release_resident_fields(weights,time_fields,&cancel);
    mx::clear_cache();
    const auto with_alias=mx::get_active_memory();
    require(retired==bytes && owner==&weights && weights.metal_convrot(),
        "release changed destination identity/mode or reported logical bytes");
    require(!weak[0].expired(),"release destroyed a caller-owned alias");
    for(size_t index=1;index<weak.size();++index)require(weak[index].expired(),"pager retained a removed field owner");
    require(before>=with_alias && before-with_alias==bytes-alias_bytes,
        "physical active delta did not exclude retained external ownership");
    external.reset();mx::clear_cache();
    const auto without_alias=mx::get_active_memory();
    require(before>=without_alias && before-without_alias==bytes,
        "released buffers survived their last external owner and idle-cache clear");
    all_retired(weights,weak);retained(weights,kept);held.unchanged();
    rejects([&]{pager.release_resident_fields(weights,time_fields,&cancel);},"unknown or retired");
    require(pager.metrics().resident_bytes_loaded==initial_metrics.resident_bytes_loaded &&
        pager.metrics().slot_arrays_allocated==initial_metrics.slot_arrays_allocated,
        "release changed historical read/allocation accounting");

    // The reusable pool is independent of released fixed storage.
    const auto &group=layout.groups.at(0);
    tc_stream_slot_ticket_v1 ticket{};
    ticket.pool=group.pool;ticket.slot=group.slot;
    require(pager.fill(group,ticket,&cancel)==group.bytes,"release damaged pool reads");
    {
        const auto bound=pager.bind(group,ticket);
        const auto &field=stage.blocks.at(0).fields.at(0);
        const auto &source=field.materialization->reads.at(0);
        std::vector<char> expected(size_t(source.bytes));
        require(::pread(held.fd.get(),expected.data(),expected.size(),off_t(source.offset))==ssize_t(expected.size()) &&
            std::memcmp(bound.at(source.tensor).data<char>(),expected.data(),expected.size())==0,
            "release changed reusable pool payload bytes");
    }
    held.unchanged();

    // A submitted GPU reader is drained before the pager drops its owners.
    const auto &source=weights.at(retained_field);
    double sum=0;
    for(size_t index=0;index<source.size();++index)
        sum+=std::bit_cast<float>(uint32_t(source.data<uint16_t>()[index])<<16);
    std::weak_ptr<Tensor::Data> reader_source(source.data_shared_ptr());
    auto reader=mx::sum(mx::astype(source,mx::float32));
    mx::async_eval(reader);
    pager.release_resident_fields(weights,{retained_field},&cancel);
    require(reader_source.expired() && reader.item<float>()==float(sum),
        "release did not drain the final GPU reader correctly");
    std::cout<<"{\"pager_retired_capacity\":"<<retired<<",\"external_alias_capacity\":"<<alias_bytes
        <<",\"active_delta_with_alias\":"<<before-with_alias
        <<",\"active_delta_after_last_alias\":"<<before-without_alias<<"}\n";
    std::cout<<"PASS pager atomic validation, weak owners, retained aliases, reader drain and source/FD/offset\n";
}

void bank_case(const std::filesystem::path &path,uint32_t prefix,bool evict) {
    std::atomic<bool> cancel{false};
    auto base=bf16_stream_plan(path,false,layers,0,passes,cancel,StreamWeightMode::ConvRotFloat);
    const auto budget=bf16_stream_managed_capacity(
        bf16_stream_plan(path,false,layers,prefix,passes,cancel,StreamWeightMode::ConvRotFloat));
    auto plan=bf16_stream_cache_plan(std::move(base),budget);
    require(plan.prefix==prefix && plan.minimum_plan,"release fixture selected the wrong optional prefix");
    const auto original_source_bytes=plan.layout.stages.at(0).resident_source_read_bytes.value();
    HeldSource held(plan.lease);
    Bf16StreamBank bank(std::move(plan),budget,cancel);
    const auto *owner=&bank.resident();
    const auto kept=bank.resident().at(retained_field).id();
    auto weak=weak_fields(bank.resident());
    const auto bytes=time_capacity(bank.resident());
    rejects([&]{bank.release_fixed_fields({time_fields[0],"transformer_blocks.0.attn.norm_q.weight"});},"layer field");
    all_present(bank.resident());
    rejects([&]{bank.release_fixed_fields({time_fields[0],"unknown.weight"});},"original fixed field");
    all_present(bank.resident());
    cancel.store(true);
    rejects([&]{bank.release_fixed_fields(time_fields);},"cancelled");
    cancel.store(false);all_present(bank.resident());
    require(bank.release_fixed_fields(time_fields)==bytes && owner==&bank.resident(),
        "bank retired capacity or resident object identity changed");
    all_retired(bank.resident(),weak);retained(bank.resident(),kept);held.unchanged();
    auto result=mx::zeros({128},mx::bfloat16);mx::eval(result);
    for(uint32_t pass=0;pass<passes;++pass) {
        bank.begin_pass(pass);
        rejects([&]{bank.release_fixed_fields({retained_field});},"drained pass boundary");
        for(uint32_t layer=0;layer<layers;++layer) {
            const auto &weights=bank.acquire(int(layer));
            const auto key="transformer_blocks."+std::to_string(layer)+".attn.norm_q.weight";
            result=result+weights.at(key);
            mx::eval(result);bank.retire(int(layer));
        }
        bank.finish_pass();
        if(evict && pass==0)require(bank.evict_cached_layers(),"optional layer cache did not evict");
        all_retired(bank.resident(),weak);retained(bank.resident(),kept);held.unchanged();
    }
    bank.finish();
    const auto metrics=bank.metrics();
    const uint64_t cached=prefix*(evict ? 1 : passes),fills=layers*passes-cached;
    require(metrics.drained && metrics.slots==2 && metrics.completed_passes==passes &&
        metrics.completed_prefix_layers==cached && metrics.completed_streamed_layers==fills &&
        metrics.fills==fills && metrics.reader_fences==fills && metrics.completed_reader_fences==fills &&
        metrics.prefix_per_pass==std::vector<uint32_t>{prefix,evict ? 0u : prefix,evict ? 0u : prefix} &&
        metrics.resident_source_bytes==original_source_bytes && metrics.managed_weight_budget_bytes==budget,
        "release changed passes, prefix eviction, fences, original read bytes or budget");
    require(mx::all(result==Tensor(float(layers*passes),mx::bfloat16)).item<bool>(),
        "release changed layer reader values");
    rejects([&]{bank.release_fixed_fields({retained_field});},"drained pass boundary");
    std::cout<<"{\"bank_prefix\":"<<prefix<<",\"evicted\":"<<(evict?"true":"false")
        <<",\"retired_capacity\":"<<bytes<<",\"passes\":"<<passes<<",\"fills\":"<<fills<<"}\n";
}

void changed_source_and_cancelled_destruction(const std::filesystem::path &path,const std::filesystem::path &root) {
    const auto changed=root/"changed-source.safetensors";
    std::filesystem::copy_file(path,changed);
    std::atomic<bool> cancel{false};
    auto plan=bf16_stream_plan(changed,false,layers,0,passes,cancel,StreamWeightMode::ConvRotFloat);
    HeldSource held(plan.lease);
    std::vector<std::weak_ptr<Tensor::Data>> weak;
    {
        Bf16StreamBank bank(std::move(plan),16ull<<20,cancel);
        weak=weak_fields(bank.resident());
        std::filesystem::rename(changed,root/"changed-source-original.safetensors");
        std::filesystem::copy_file(path,changed);
        rejects([&]{bank.release_fixed_fields(time_fields);},"source");
        all_present(bank.resident());
        for(const auto &owner:weak)require(!owner.expired(),"source failure partly retired pager ownership");
        cancel.store(true);
        rejects([&]{bank.release_fixed_fields(time_fields);},"cancelled");
    }
    for(const auto &owner:weak)require(owner.expired(),"cancelled bank destruction retained fixed ownership");
    require(::fcntl(held.fd.get(),F_GETFD)>=0 && ::lseek(held.fd.get(),0,SEEK_CUR)==7,
        "cancelled destruction closed or consumed an external held source fd");
    std::cout<<"PASS source mutation rejects atomically; cancelled destruction drains fixed owners\n";
}
} // namespace

int main(int argc,char **argv) {
    try {
        require(argc==4,"expected owned DiT fixture, evidence directory and exact native library");
        char *raw=tc_runtime_build_identity();require(raw,"missing native runtime ID");
        const std::string identity(raw);tc_string_free(raw);
        require(identity==TC_TEST_BOUND_NATIVE_ID,"fixed-release fixture loaded another runtime build");
        Dl_info image{};
        require(::dladdr(reinterpret_cast<const void *>(&tc_runtime_build_identity),&image) && image.dli_fname &&
            std::filesystem::canonical(image.dli_fname)==std::filesystem::canonical(argv[3]),
            "fixed-release fixture loaded another library path");
        std::cout<<"NATIVE_RUNTIME_ID="<<identity<<"\nNATIVE_LIBRARY_PATH="<<image.dli_fname<<std::endl;
        configure_streams();mx::set_cache_limit(0);
        pager_cases(argv[1]);
        bank_case(argv[1],1,false);
        bank_case(argv[1],2,false);
        bank_case(argv[1],2,true);
        changed_source_and_cancelled_destruction(argv[1],argv[2]);
        std::cout<<"PASS Qwen fixed release: prefix1/2, three passes, 6/10 refill eviction, no fixed resurrection\n";
        return 0;
    }catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
