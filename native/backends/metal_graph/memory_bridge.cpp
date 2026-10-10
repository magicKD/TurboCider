#include "memory_bridge.hpp"
#include "../../runtime/memory_execution.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace tc::metal_graph {
namespace {
void require_bridge(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Claim {
    std::optional<MemoryReservation> reservation;
    std::optional<StorageLease> lease;
    StorageId storage;
};
struct ImportedOwner {
    std::shared_ptr<Claim> claim;
    std::shared_ptr<void> backing;
    ImportedOwner(std::shared_ptr<Claim> value,std::shared_ptr<void> owner)
        :claim(std::move(value)),backing(std::move(owner)) {}
    ~ImportedOwner() noexcept {
        // Return the external allocation before announcing budget as free.
        backing.reset();
        claim.reset();
    }
};
uint64_t handle(void* value) { return uint64_t(reinterpret_cast<uintptr_t>(value)); }
} // namespace

struct MemoryBridge::State {
    std::mutex mutex;
    MemoryExecutionContext* context;
    const std::thread::id owner = std::this_thread::get_id();
    const uint64_t domain, generation;
    const std::vector<AllocationSiteBinding> bindings;
    // Token provenance only: weak refs neither keep claims alive nor form a
    // second byte ledger. Validate before casting opaque callback arguments.
    std::map<const void*,std::weak_ptr<Claim>> issued;

    State(MemoryExecutionContext& value,uint64_t domain_value,uint64_t generation_value,
          std::vector<AllocationSiteBinding> sites)
        :context(&value),domain(domain_value),generation(generation_value),bindings(std::move(sites)) {}

    void require_owner_running() const {
        require_bridge(std::this_thread::get_id()==owner,
            "native memory bridge allocation/import requires its owner thread");
        require_bridge(context!=nullptr,"native memory bridge is closed");
        require_bridge(context->state()==MemoryExecutionState::Running,
            "native memory bridge allocation/import requires a running context");
    }
    std::shared_ptr<Claim> find_claim(const std::shared_ptr<void>& opaque,bool committed) {
        require_bridge(bool(opaque),"native memory bridge claim is missing");
        const auto found=issued.find(opaque.get());
        require_bridge(found!=issued.end(),"native memory bridge claim belongs to another binding");
        auto claim=found->second.lock();
        require_bridge(bool(claim) && !claim.owner_before(opaque) && !opaque.owner_before(claim),
            "native memory bridge claim has stale or foreign ownership");
        if (committed)
            require_bridge(claim->lease.has_value() && bool(*claim->lease) && !claim->reservation,
                "native memory bridge import requires an already committed claim");
        return claim;
    }
    std::shared_ptr<void> reserve(size_t upper,std::string_view label) {
        std::lock_guard lock(mutex);
        require_owner_running();
        require_bridge(upper>0,"native memory bridge reservation must be nonzero");
        const auto found=std::lower_bound(bindings.begin(),bindings.end(),label,
            [](const auto& binding,std::string_view value) { return binding.label<value; });
        require_bridge(found!=bindings.end() && found->label==label,
            "native memory bridge allocation label has no manifest site binding");
        // Deliberately no MemoryAdmission::try_reserve / generic fallback.
        auto reservation=context->try_reserve_site(found->site_id,found->memory_class,upper);
        if (!reservation) return {};
        auto claim=std::make_shared<Claim>();
        claim->reservation.emplace(std::move(*reservation));
        for (auto it=issued.begin();it!=issued.end();)
            if (it->second.expired()) it=issued.erase(it); else ++it;
        const auto [entry,inserted]=issued.emplace(claim.get(),claim);
        (void)entry;
        require_bridge(inserted,"native memory bridge duplicate live token identity");
        return claim;
    }
    void allocated(const std::shared_ptr<void>& opaque,void* native,size_t actual) {
        std::lock_guard lock(mutex);
        require_owner_running();
        auto claim=find_claim(opaque,false);
        require_bridge(native && actual,"native memory bridge commit requires actual backing capacity");
        require_bridge(claim->reservation.has_value() && bool(*claim->reservation) && !claim->lease,
            "native memory bridge token was already committed or cancelled");
        require_bridge(actual<=claim->reservation->reserved_bytes(),
            "native memory bridge actual capacity exceeds reservation");
        const StorageId storage{domain,handle(native),actual,generation};
        claim->lease.emplace(claim->reservation->commit(storage));
        claim->reservation.reset();
        claim->storage=storage;
    }
};

MemoryBridge::MemoryBridge(MemoryExecutionContext& context,uint64_t domain,uint64_t generation,
                           std::vector<AllocationSiteBinding> bindings) {
    require_bridge(domain && generation && !bindings.empty(),
        "native memory bridge requires a domain, generation and explicit allocation sites");
    require_bridge(context.state()==MemoryExecutionState::Admitted ||
                   context.state()==MemoryExecutionState::Running,
        "native memory bridge cannot bind a draining or terminal context");
    std::sort(bindings.begin(),bindings.end(),[](const auto& a,const auto& b) { return a.label<b.label; });
    const auto& sites=context.compiled_plan().allocation_sites;
    for (size_t i=0;i<bindings.size();++i) {
        const auto& b=bindings[i];
        require_bridge(!b.label.empty() && !b.site_id.empty() &&
                       b.memory_class!=MemoryClass::UnknownExternal &&
                       (!i || b.label!=bindings[i-1].label),
            "native memory bridge label/site/class is empty, unknown or duplicated");
        const auto site=std::find_if(sites.begin(),sites.end(),
            [&](const auto& item) { return item.site_id==b.site_id; });
        require_bridge(site!=sites.end() && site->memory_class==b.memory_class,
            "native memory bridge site/class differs from the compiled manifest");
    }
    state_=std::make_shared<State>(context,domain,generation,std::move(bindings));
}
MemoryBridge::~MemoryBridge() noexcept { close(); }
AllocationHooks MemoryBridge::hooks() const {
    return {[state=state_](size_t upper,std::string_view label) { return state->reserve(upper,label); },
            [state=state_](const std::shared_ptr<void>& claim,void* native,size_t actual) {
                state->allocated(claim,native,actual);
            }};
}
void MemoryBridge::close() noexcept {
    std::lock_guard lock(state_->mutex);
    state_->context=nullptr;
}
StorageId MemoryBridge::allocation_identity(const std::shared_ptr<void>& opaque) const {
    std::lock_guard lock(state_->mutex);
    state_->require_owner_running();
    return state_->find_claim(opaque,true)->storage;
}
std::shared_ptr<void> MemoryBridge::retain_import(const std::shared_ptr<void>& opaque,
    void* native,size_t actual,std::shared_ptr<void> owner) const {
    std::lock_guard lock(state_->mutex);
    state_->require_owner_running();
    auto claim=state_->find_claim(opaque,true);
    const auto& s=claim->storage;
    require_bridge(bool(owner) && native && actual && s.allocator_domain==state_->domain &&
        s.session_generation==state_->generation && s.handle==handle(native) && s.capacity==actual,
        "native memory bridge import owner/backing/capacity/provenance mismatch");
    require_bridge(owner.owner_before(claim) || claim.owner_before(owner),
        "native memory bridge accounting claim is not an external allocator owner");
    return std::make_shared<ImportedOwner>(std::move(claim),std::move(owner));
}

} // namespace tc::metal_graph
