// CPU-only checks of the production metadata key and entry state. Opaque
// handles below are never dereferenced or submitted to Metal/private ANE.
#include "../../native/backends/metal_graph/native_weight_scale_cache_contract.hpp"

#include <bit>
#include <iostream>
#include <limits>

using namespace tc::metal_graph;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}

struct SourceFixture {
    // Fused gate/up genuinely share one ownership control block. The down
    // allocation has a separate owner; metadata describes the full BF16 view.
    std::shared_ptr<const std::array<int, 2>> fused =
        std::make_shared<const std::array<int, 2>>(std::array<int, 2>{1, 2});
    std::shared_ptr<const int> down = std::make_shared<const int>(3);
    std::shared_ptr<const void> fused_owner = fused, down_owner = down;

    WeightScaleCacheKey key() const {
        WeightScaleCacheKey result;
        result.immutable_source_revision = "request-owned-materialized-bf16-revision-1";
        result.source_generation = 7;
        auto& recipe = result.recipe;
        recipe.hidden = 4096; recipe.width = 12288; recipe.ane_channels = 6144;
        recipe.first_channel = 6144;
        recipe.gate_norm_bits = std::bit_cast<uint32_t>(0.0883883461356163f);
        recipe.down_norm_bits = std::bit_cast<uint32_t>(0.04419417306780815f);
        const size_t full_gate = size_t(12288) * 4096 * 2;
        const size_t full_down = size_t(4096) * 12288 * 2;
        result.sources[0] = {fused_owner, reinterpret_cast<uintptr_t>(fused.get()),
                            0, full_gate, 12288, 4096, 8192, ScalarType::BF16};
        result.sources[1] = {fused_owner, reinterpret_cast<uintptr_t>(fused.get()),
                            full_gate, full_gate, 12288, 4096, 8192, ScalarType::BF16};
        result.sources[2] = {down_owner, reinterpret_cast<uintptr_t>(down.get()),
                            0, full_down, 4096, 12288, 24576, ScalarType::BF16};
        return result;
    }
};

void owner_identity() {
    SourceFixture fixture;
    const auto key = fixture.key();
    validate_scale_cache_key(key);
    check(same_scale_cache_key(key, key), "live key must match itself");

    // Different get() addresses with the SAME allocation control block are
    // aliases, whereas a new control block over the same get() is not identity.
    const std::shared_ptr<const void> alias_a(fixture.fused, &(*fixture.fused)[0]);
    const std::shared_ptr<const void> alias_b(fixture.fused, &(*fixture.fused)[1]);
    check(alias_a.get() != alias_b.get(), "fixture aliases must have different get addresses");
    check(same_allocation_owner(alias_a, alias_b), "shared-owner aliasing pointers lost identity");
    auto alias_key = key;
    alias_key.sources[0].allocation = alias_b;
    check(same_scale_cache_key(key, alias_key), "alias owner changed otherwise identical source");

    const std::shared_ptr<const void> unrelated(alias_a.get(), [](const void*) {});
    check(unrelated.get() == alias_a.get(), "fixture must exercise same-address control blocks");
    check(!same_allocation_owner(alias_a, unrelated), "same address incorrectly grants ownership identity");
    auto foreign = key;
    foreign.sources[0].allocation = unrelated;
    validate_scale_cache_key(foreign); // valid metadata, a different allocation
    check(!same_scale_cache_key(key, foreign), "foreign same-address control block hit cache");

    const std::weak_ptr<const void> empty;
    check(!same_allocation_owner(empty, empty), "empty owners incorrectly match");
    std::weak_ptr<const void> expired;
    {
        std::shared_ptr<const void> temporary = std::make_shared<const int>(4);
        expired = temporary;
    }
    check(!same_allocation_owner(expired, expired), "expired owners incorrectly match");
    auto stale = key;
    stale.sources[1].allocation = expired;
    check(!same_scale_cache_key(stale, stale), "expired key matched itself");
    rejects([&] { validate_scale_cache_key(stale); }, "expired source key accepted");
}

void key_changes_invalidate() {
    SourceFixture fixture;
    const auto key = fixture.key();
    WeightScaleEntryState entry;
    entry.begin(key, 1); entry.publish(1);
    const auto miss = [&](const auto& mutate) {
        auto changed = key;
        mutate(changed);
        check(!same_scale_cache_key(key, changed), "changed cache dependency compared equal");
        check(!entry.hit(changed), "changed cache dependency reused old scales");
        check(entry.hit(key), "a miss invalidated the original immutable entry");
    };
    miss([](auto& k) { ++k.source_generation; });
    miss([](auto& k) { k.immutable_source_revision += "-rebound"; });
    for (size_t i = 0; i < key.sources.size(); ++i) {
        miss([i](auto& k) { k.sources[i].native_buffer ^= uintptr_t(16); });
        miss([i](auto& k) { k.sources[i].offset += 2; });
        miss([i](auto& k) { k.sources[i].extent += 2; });
        miss([i](auto& k) { k.sources[i].pitch += 2; });
        miss([i](auto& k) { --k.sources[i].rows; });
        miss([i](auto& k) { --k.sources[i].columns; });
        miss([i](auto& k) { k.sources[i].type = ScalarType::FP16; });
    }
    // The fused tensors share owner+handle but must retain their distinct view
    // offsets; swapping gate/up is not a hit even when shape/dtype are equal.
    check(same_allocation_owner(key.sources[0].allocation, key.sources[1].allocation),
          "fused fixture does not share real ownership");
    check(!same_scale_source(key.sources[0], key.sources[1]), "fused gate/up offsets collapsed");
    miss([](auto& k) { std::swap(k.sources[0], k.sources[1]); });

    miss([](auto& k) { ++k.recipe.hidden; });
    miss([](auto& k) { ++k.recipe.width; });
    miss([](auto& k) { --k.recipe.ane_channels; });
    miss([](auto& k) { --k.recipe.first_channel; });
    miss([](auto& k) { ++k.recipe.seed; });
    miss([](auto& k) { ++k.recipe.seed_high; });
    miss([](auto& k) { ++k.recipe.basis; });
    miss([](auto& k) { k.recipe.gate_norm_bits ^= 1U; });
    miss([](auto& k) { k.recipe.down_norm_bits ^= 1U; });
    miss([](auto& k) { k.recipe.gate_block = 512; });
    miss([](auto& k) { k.recipe.down_block = 128; });
    miss([](auto& k) { ++k.recipe.version; });
    miss([](auto& k) { k.recipe.staging = 1; });
}

void invalid_views() {
    SourceFixture fixture;
    const auto key = fixture.key();
    const auto invalid = [&](const auto& mutate) {
        auto changed = key;
        mutate(changed);
        rejects([&] { validate_scale_cache_key(changed); }, "invalid source view/epoch accepted");
    };
    invalid([](auto& k) { k.immutable_source_revision.clear(); });
    invalid([](auto& k) { k.source_generation = 0; });
    invalid([](auto& k) { k.recipe.hidden = 0; });
    invalid([](auto& k) { k.recipe.width = 0; });
    invalid([](auto& k) { k.recipe.ane_channels = 0; });
    invalid([](auto& k) { k.recipe.ane_channels = k.recipe.width + 1; });
    invalid([](auto& k) { k.recipe.version = 0; });
    invalid([](auto& k) { k.recipe.staging = 2; });
    for (size_t i = 0; i < key.sources.size(); ++i) {
        invalid([i](auto& k) { k.sources[i].allocation.reset(); });
        invalid([i](auto& k) { k.sources[i].native_buffer = 0; });
        invalid([i](auto& k) { k.sources[i].type = ScalarType::FP16; });
        invalid([i](auto& k) { --k.sources[i].rows; });
        invalid([i](auto& k) { --k.sources[i].columns; });
        invalid([i](auto& k) { k.sources[i].pitch = k.sources[i].columns * 2 - 2; });
        invalid([i](auto& k) { ++k.sources[i].pitch; });
        invalid([i](auto& k) { ++k.sources[i].offset; });
        invalid([i](auto& k) { --k.sources[i].extent; });
        invalid([i](auto& k) { k.sources[i].offset = std::numeric_limits<size_t>::max() - 1; });
    }
    // Full matrices with aligned padding are legal; compact-only assumptions
    // would incorrectly reject original down strides and retained full views.
    auto padded = key;
    padded.sources[2].pitch += 64;
    padded.sources[2].extent += size_t(padded.sources[2].rows - 1) * 64;
    validate_scale_cache_key(padded);
    check(!same_scale_cache_key(key, padded), "padding/layout changed without cache miss");
}

void entry_transaction() {
    SourceFixture fixture;
    const auto key = fixture.key();
    WeightScaleEntryState entry;
    using State = WeightScaleEntryState::State;
    check(entry.state() == State::Empty && !entry.hit(key), "empty entry has readable scales");
    rejects([&] { entry.publish(1); }, "empty entry published without a fill");
    rejects([&] { entry.begin(key, 0); }, "zero invocation accepted");
    entry.begin(key, 10);
    check(entry.state() == State::Pending && !entry.hit(key), "pending scales exposed as valid");
    rejects([&] { entry.begin(key, 11); }, "pending fill overwritten");
    rejects([&] { entry.publish(9); }, "stale invocation published another fill");
    check(entry.state() == State::Pending && !entry.hit(key), "stale publish mutated pending state");
    entry.publish(10);
    check(entry.state() == State::Valid && entry.hit(key), "matching fill did not publish");
    rejects([&] { entry.publish(10); }, "duplicate publication accepted");
    auto next = key;
    ++next.source_generation;
    entry.begin(next, 11);
    check(!entry.hit(key) && !entry.hit(next), "refill exposed old or pending scales");
    rejects([&] { entry.publish(10); }, "previous generation published a refill");
    entry.publish(11);
    check(entry.hit(next) && !entry.hit(key), "refill did not replace source generation");
    entry.retire();
    check(entry.state() == State::Retired && !entry.hit(next), "retired cache resurrected a hit");
    rejects([&] { entry.begin(key, 12); }, "retired entry allowed a new fill");
    rejects([&] { entry.publish(11); }, "late callback revived a retired entry");
    entry.retire(); // idempotent sticky retirement

    WeightScaleEntryState expired_pending, expired_valid;
    WeightScaleCacheKey escaped;
    {
        SourceFixture temporary;
        escaped = temporary.key();
        expired_pending.begin(escaped, 21);
        expired_valid.begin(escaped, 22); expired_valid.publish(22);
    }
    check(!expired_valid.hit(escaped), "valid entry outlived its actual allocation owner");
    rejects([&] { expired_pending.publish(21); }, "late publication accepted expired source owners");
    check(!expired_pending.hit(escaped), "expired pending entry became readable");
    expired_pending.retire(); expired_valid.retire();
}

void payload() {
    check(compact_weight_scale_bytes(4096, 6144) == 32768, "production layer scales must use 32 KiB");
    check(compact_weight_scale_bytes(4096, 6144) * 32 == 1024 * 1024,
          "production 32-layer scale payload must use 1024 KiB");
    check(compact_weight_scale_bytes(128, 512) * 2 == 4608,
          "generated two-layer scale payload must use 4608 bytes");
    rejects([] { (void)compact_weight_scale_bytes(0, 512); }, "zero hidden size accepted");
    rejects([] { (void)compact_weight_scale_bytes(128, 0); }, "zero channel count accepted");
}
} // namespace

int main() {
    try {
        owner_identity(); key_changes_invalidate(); invalid_views(); entry_transaction(); payload();
        std::cout << "native_weight_scale_cache_contract=PASS ownership=control_block "
                     "views_recipes_epochs=checked stale_publish=checked expired_owners=checked "
                     "production_payload=1048576 tiny_payload=4608 devices=0 models=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
