#include "../../native/runtime/streaming/preset_catalog.hpp"

#include <cassert>
#include <iostream>

using namespace tc::streaming;

namespace {
PresetWorkload workload() {
    return {"z-image-turbo", "image.generate", "gpu",
            "Apple Test GPU/16GiB", "embedded_app", 512, 512, 1, 9, false};
}

StreamingPresetRecord record(const char *id, uint32_t rank,
                             uint64_t calibrated, uint64_t reads) {
    StreamingPresetRecord value;
    value.id = id;
    value.revision = 1;
    value.catalog_revision = "test-r1";
    value.release_channel = "public-experimental";
    value.workload = workload();
    value.calibration = {true, calibrated,
                         "execution_process_tree_v1", "observed_tree_max_v1"};
    value.minimum_physical_memory_bytes = 16 * gib;
    value.maximum_physical_memory_bytes = 16 * gib;
    value.logical_read_bytes = reads;
    value.performance_rank = rank;
    value.evidence_digest = "sha256:test-only";
    return value;
}

PresetResolveQuery query(uint64_t target) {
    PresetResolveQuery value;
    value.workload = workload();
    value.target_request_memory_bytes = target;
    value.physical_memory_bytes = 16 * gib;
    return value;
}
} // namespace

int main() {
    assert(streaming_target_margin_bytes(8 * gib) == 858993460ull);
    assert(streaming_target_margin_bytes(10 * gib) == gib);
    assert(supported_streaming_target(12 * gib));
    assert(!supported_streaming_target(14 * gib));

    StreamingPresetCatalog catalog{"test-r1", {
        record("slower-small", 2, 7 * gib, 100),
        record("fast-fit", 1, 8 * gib, 200),
        record("fast-too-large", 0, 10 * gib, 50),
    }};
    auto ten = resolve_streaming_preset(query(10 * gib), catalog);
    assert(ten.selected && ten.selected->id == "fast-fit");
    auto twelve = resolve_streaming_preset(query(12 * gib), catalog);
    assert(twelve.selected && twelve.selected->id == "fast-too-large");

    auto exact_query = query(10 * gib);
    exact_query.preset_id = "slower-small";
    exact_query.preset_revision = 1;
    exact_query.catalog_revision = "test-r1";
    auto exact = resolve_streaming_preset(exact_query, catalog);
    assert(exact.selected && exact.selected->id == "slower-small");

    exact_query.catalog_revision = "old-r0";
    assert(resolve_streaming_preset(exact_query, catalog).rejection_code ==
           "streaming_resolution_stale");

    auto unsupported = query(14 * gib);
    assert(resolve_streaming_preset(unsupported, catalog).rejection_code ==
           "unsupported_memory_target");

    auto wrong_memory = query(10 * gib);
    wrong_memory.physical_memory_bytes = 12 * gib;
    assert(resolve_streaming_preset(wrong_memory, catalog).rejection_code ==
           "unvalidated_workload");

    auto incomplete = record("incomplete", 0, 0, 0);
    incomplete.calibration.complete = false;
    StreamingPresetCatalog incomplete_catalog{"test-r1", {incomplete}};
    assert(resolve_streaming_preset(query(10 * gib), incomplete_catalog)
               .rejection_code == "memory_calibration_incomplete");

    auto revoked = record("revoked", 0, 7 * gib, 0);
    revoked.revoked = true;
    StreamingPresetCatalog revoked_catalog{"test-r1", {revoked}};
    assert(resolve_streaming_preset(query(10 * gib), revoked_catalog)
               .rejection_code == "preset_not_public");

    assert(production_streaming_preset_catalog().records.empty());
    std::cout << "PASS public preset resolver: exact integer margins, deterministic "
                 "rank, replay identity, device/calibration/revocation fail-closed\n";
}
