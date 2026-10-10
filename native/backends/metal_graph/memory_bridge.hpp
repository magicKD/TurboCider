#pragma once

#include "backend.hpp"
#include "../../runtime/memory_accounting.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tc { class MemoryExecutionContext; }

namespace tc::metal_graph {

struct AllocationSiteBinding {
    std::string label;
    std::string site_id;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
};

// Scoped adapter to the existing request ledger, not another allocator/budget.
// Construct and allocate on the request owner thread. The context must outlive
// this binding (or close()); Session::unbind_memory_context must close it before
// destroying that context. Copies of hooks then reject new allocations safely.
// Issued tokens contain only ledger reservations/leases and may die on a Metal
// completion thread after close/context destruction; they never call Context.
class MemoryBridge {
public:
    MemoryBridge(MemoryExecutionContext&, uint64_t allocator_domain,
                 uint64_t generation, std::vector<AllocationSiteBinding>);
    ~MemoryBridge() noexcept;
    MemoryBridge(const MemoryBridge&) = delete;
    MemoryBridge& operator=(const MemoryBridge&) = delete;
    MemoryBridge(MemoryBridge&&) = delete;
    MemoryBridge& operator=(MemoryBridge&&) = delete;

    AllocationHooks hooks() const;
    void close() noexcept;

    // Reuse a claim issued AND committed by this binding for exactly this
    // native backing/capacity. Returns one owner retaining both the external
    // allocator owner and original claim; pass it as Backend::import_buffer's
    // storage_owner. No reservation, new StorageId or duplicate byte charge.
    // The caller still proves backing_owner owns that actual allocation, not
    // merely an Objective-C retain of its MTLBuffer. A claim cannot be its own
    // allocator owner; opaque ownership cannot authenticate arbitrary callers.
    // A plain MLX array owner/anonymous lease is not accepted as accounting
    // evidence. External allocators can use hooks before/after their original
    // allocation, then retain that same token with every escaped Data owner.
    std::shared_ptr<void> retain_import(const std::shared_ptr<void>& claim,
        void* native_buffer, size_t allocated_bytes,
        std::shared_ptr<void> backing_owner) const;

    // Metadata for a committed claim, not a lookup by raw address or a claim
    // manufacturing API. Same binding/thread/lifetime checks as retain_import.
    StorageId allocation_identity(const std::shared_ptr<void>& claim) const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace tc::metal_graph
