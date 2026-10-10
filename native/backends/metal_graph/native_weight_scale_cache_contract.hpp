#pragma once

#include "backend.hpp"
#include <array>
#include <optional>

namespace tc::metal_graph {

// Metadata only. These keys/transitions grant no device completion permit.
// The controller is responsible for source immutability and for calling
// publish only after its actual complete successful physical execution.
struct WeightScaleSourceIdentity {
    std::weak_ptr<const void> allocation;
    uintptr_t native_buffer = 0;
    size_t offset = 0, extent = 0;
    uint32_t rows = 0, columns = 0;
    size_t pitch = 0;
    ScalarType type = ScalarType::Bytes;
};
struct WeightScaleRecipe {
    uint32_t hidden = 0, width = 0, ane_channels = 0;
    uint32_t first_channel = 0, seed = 20260930, seed_high = 0, basis = 0;
    uint32_t gate_norm_bits = 0, down_norm_bits = 0;
    uint32_t gate_block = 128, down_block = 512;
    uint32_t version = 1;
    uint8_t staging = 0;
    bool operator==(const WeightScaleRecipe&) const = default;
};
struct WeightScaleCacheKey {
    std::array<WeightScaleSourceIdentity,3> sources;
    WeightScaleRecipe recipe;
    std::string immutable_source_revision;
    uint64_t source_generation = 0;
};
inline bool same_allocation_owner(const std::weak_ptr<const void>& a,
                                  const std::weak_ptr<const void>& b) noexcept {
    return !a.expired() && !b.expired() && !a.owner_before(b) && !b.owner_before(a);
}
inline bool same_scale_source(const WeightScaleSourceIdentity& a,
                              const WeightScaleSourceIdentity& b) noexcept {
    return same_allocation_owner(a.allocation,b.allocation) && a.native_buffer==b.native_buffer &&
        a.offset==b.offset && a.extent==b.extent && a.rows==b.rows && a.columns==b.columns &&
        a.pitch==b.pitch && a.type==b.type;
}
inline void validate_scale_cache_key(const WeightScaleCacheKey& key) {
    if (key.immutable_source_revision.empty() || !key.source_generation ||
        !key.recipe.hidden || !key.recipe.width || !key.recipe.ane_channels ||
        key.recipe.ane_channels>key.recipe.width || key.recipe.version!=1 || key.recipe.staging>1)
        throw std::invalid_argument("native scale cache requires an immutable source epoch and exact recipe");
    for (size_t i=0;i<key.sources.size();++i) {
        const auto& s=key.sources[i];
        const uint32_t rows=i==2?key.recipe.hidden:key.recipe.width;
        const uint32_t columns=i==2?key.recipe.width:key.recipe.hidden;
        const uint64_t last=uint64_t(columns)*2;
        if (s.allocation.expired() || !s.native_buffer || s.type!=ScalarType::BF16 ||
            s.rows!=rows || s.columns!=columns || s.pitch<last || s.pitch>UINT32_MAX || (s.pitch%2) ||
            (s.offset%2) || s.extent>SIZE_MAX-s.offset)
            throw std::invalid_argument("native scale cache source allocation/view identity is invalid");
        const uint64_t extent=uint64_t(rows-1)*s.pitch+last;
        if(extent>s.extent)
            throw std::invalid_argument("native scale cache source view extent is too small");
    }
}
inline bool same_scale_cache_key(const WeightScaleCacheKey& a,
                                 const WeightScaleCacheKey& b) noexcept {
    if (a.source_generation!=b.source_generation || a.immutable_source_revision!=b.immutable_source_revision ||
        a.recipe!=b.recipe) return false;
    for (size_t i=0;i<a.sources.size();++i)
        if (!same_scale_source(a.sources[i],b.sources[i])) return false;
    return true;
}
inline size_t compact_weight_scale_bytes(uint32_t hidden,uint32_t channels) {
    const uint64_t bytes=(uint64_t(channels)*2+hidden)*2;
    if (!hidden || !channels || bytes>SIZE_MAX)
        throw std::invalid_argument("native compact weight scale payload overflows");
    return size_t(bytes);
}
class WeightScaleEntryState {
public:
    enum class State : uint8_t { Empty, Pending, Valid, Retired };
    State state() const noexcept { return state_; }
    bool hit(const WeightScaleCacheKey& key) const noexcept {
        return state_==State::Valid && key_ && same_scale_cache_key(*key_,key);
    }
    void begin(WeightScaleCacheKey key,uint64_t invocation) {
        validate_scale_cache_key(key);
        if (!invocation || state_==State::Pending || state_==State::Retired)
            throw std::invalid_argument("native scale cache cannot fill a pending/retired entry");
        key_=std::move(key); invocation_=invocation; state_=State::Pending;
    }
    void publish(uint64_t invocation) {
        if (state_!=State::Pending || !invocation || invocation!=invocation_ || !key_)
            throw std::invalid_argument("native scale cache publication has a stale invocation");
        validate_scale_cache_key(*key_);
        state_=State::Valid;
    }
    void retire() noexcept { state_=State::Retired; key_.reset(); }
private:
    State state_=State::Empty;
    std::optional<WeightScaleCacheKey> key_;
    uint64_t invocation_=0;
};

} // namespace tc::metal_graph
