#pragma once

#include "native_ffn.hpp"
#include "native_ane_handoff.hpp"
#include "native_ane_stage_contract.hpp"

namespace tc::metal_graph {

using AneStageGeometry = native_ane_stage_contract::Geometry;
enum class AneWeightStaging : uint8_t { Generic, SylvesterRegister };
class NativeAneProducerPermit;

struct AneWeightBankViews {
    ImportedAneSurface g, sg, u, su, d, sd;
};
struct AneActivationViews {
    ImportedAneSurface x, tx;
};
// Exactly rows*2 FP16 bytes per view. The three slices may share one backing
// allocation only when their byte ranges are disjoint. Cache readiness and
// immutable source identity belong to the controller, not to these views.
struct AneCompactWeightScales {
    BufferView gate, up, down;
};

// Fresh job-owned Shared UInt32. Never CPU reset/reused after recording. The
// controller owns graph construction and proves these encoders belong to the
// handoff's captured producer; this is not an arbitrary graph write-set proof.
class AneStageStatus {
public:
    static AneStageStatus producer(Backend&, NativeAneJobIdentity);
    static AneStageStatus restore(Backend&, const NativeAneReadPermit&, const AneStageStatus& producer);
    AneStageStatus(const AneStageStatus&)=delete;
    AneStageStatus& operator=(const AneStageStatus&)=delete;
    AneStageStatus(AneStageStatus&&) noexcept=default;
    AneStageStatus& operator=(AneStageStatus&&) noexcept=default;
    const BufferView& validation_view() const noexcept { return status_; }
    const NativeAneJobIdentity& identity() const noexcept { return identity_; }
    uint32_t encoded_mask() const noexcept { return encoded_; }
    bool validated() const noexcept { return validated_; }
    // The actual sealed permit validates its captured producer, not a caller's
    // self-reported Completion. Read flags only after successful physical join.
    void validate_after(const NativeAneProducerPermit&);
    // Controller passes its ACTUAL captured restore/join submission. Tickets
    // are sealed native objects; an unrelated ticket must never be substituted.
    void validate_after(const Ticket& actual_consumer);
private:
    AneStageStatus()=default;
    native_ane_stage_contract::StatusRole role_{};
    BufferView status_;
    NativeAneJobIdentity identity_{};
    uint32_t encoded_=0;
    bool validated_=false;
    void* device_=nullptr;
    std::optional<AneStageGeometry> geometry_;
    std::optional<ImportedAneSurface> down_scales_;
    std::optional<tc::ane::private_api::AneRequestProvenance> provenance_;
    // Metadata only, for cross-stage aliases in either recording order. The
    // recorded graph, and sd above, retain the actual allocation owners.
    std::array<native_ane_stage_contract::Layout,3> weight_sources_{};
    std::array<native_ane_stage_contract::Layout,6> weight_outputs_{};
    std::array<native_ane_stage_contract::Layout,3> activation_{};
    std::optional<std::array<native_ane_stage_contract::Layout,3>> compact_scales_;
    friend class PreparedAneStages;
};

// Base-only, two-pass Sylvester W8/A8 and FP32 restore. No S1, LoRA,
// lookahead/prefetch, single-pass candidate, source copies or implicit eval.
class PreparedAneStages {
public:
    static PreparedAneStages prepare(Backend&, AneStageGeometry = {},
        AneWeightStaging = AneWeightStaging::Generic, bool scale_reuse = false);
    AneStageGeometry geometry() const noexcept { return geometry_; }
    AneWeightStaging weight_staging() const noexcept { return weight_staging_; }
    // reuse requires a controller-qualified immutable-source cache snapshot.
    // fill only records raw FP16 copies; it never publishes cache readiness.
    // Both modes require scale_reuse preparation and cannot be combined.
    void encode_weight_bank(CommandGraph&, const MatrixView& gate,
        const MatrixView& up, const MatrixView& down, const AneWeightBankViews&,
        AneStageStatus&, const AneCompactWeightScales* reuse = nullptr,
        const AneCompactWeightScales* fill = nullptr) const;
    void encode_activation(CommandGraph&, const MatrixView& normalized_input,
        const AneActivationViews&, AneStageStatus&) const;
    // Uses the SAME sd snapshot captured by this producer status; y can be
    // accessed only through this successful permit. No prequeued restore.
    void encode_restore(CommandGraph&, const NativeAneReadPermit&,
        const AneStageStatus& producer, const MatrixView& fp32_partial,
        AneStageStatus& restore) const;
private:
    AneStageGeometry geometry_;
    AneWeightStaging weight_staging_=AneWeightStaging::Generic;
    void* device_=nullptr;
    Pipeline scales_, codes_, restore_;
    Pipeline scale_copy_;
    Pipeline register_scales128_, register_codes128_, register_scales512_, register_codes512_;
    BufferView signs_, zero_;
    void encode_quant(CommandGraph&, const MatrixView&, const ImportedAneSurface& codes,
        const ImportedAneSurface& scales, const native_ane_stage_contract::W8Parameters&,
        const BufferView& status, bool skip_scales = false) const;
    void encode_scale_copy(CommandGraph&, const BufferView& source,
        const BufferView& destination, uint32_t rows,
        uint32_t source_half_pitch, uint32_t destination_half_pitch) const;
};

} // namespace tc::metal_graph
