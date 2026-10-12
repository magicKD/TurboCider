#pragma once

#include "../../runtime/session.hpp"
#include "../../runtime/streaming/source_lease.hpp"
#include "../../backends/mlx.hpp"
#include "../../backends/metal_graph/backend.hpp"

namespace tc { struct MemoryExecutionReport; struct NativeHybridDenoiserMetrics; }
namespace tc::qwen21 {

// Boundary adapter only. The text encoder and VAE keep their existing MLX
// executor; every operation between native begin and final latents belongs to
// the independent native graph. This owner is never cached across requests.
class NativeSession {
public:
    // Optional adapter lease must be the exact verified object used by
    // Weights::apply_loras_lease. Pipeline materializes its immutable factors
    // before this owner captures the native phase baseline. The explicit
    // candidate route verifies its content pin separately from model qualification.
    NativeSession(const Request&, std::shared_ptr<const streaming::SourceLease> dit_source,
                  std::shared_ptr<const streaming::SourceLease> conditioning_source,
                  const Weights&, const Tensor& text, const Tensor& latents,
                  const Tensor& schedule, MemoryExecutionContext* parent_context,
                  std::shared_ptr<const streaming::SourceLease> adapter_source = {});
    ~NativeSession() noexcept;
    NativeSession(const NativeSession&) = delete;
    NativeSession& operator=(const NativeSession&) = delete;

    metal_graph::Completion step();
    metal_graph::Completion physical_status() const;
    metal_graph::Completion drain(std::chrono::milliseconds timeout);
    // Copy actual native hybrid counters before finish destroys the run. The
    // result is separate from memory certification and hardware placement.
    NativeHybridDenoiserMetrics hybrid_metrics() const;
    NativeDenoiserLoRAMetrics lora_metrics() const;
    // Requires a complete, finite native result and source revalidation. A
    // bounded 128-KiB copy publishes a produced MLX latent only at this boundary.
    Tensor finish_latents();
    // Destroy graph/source wrappers and finish the native-only memory scope.
    // Successful source/input MLX arrays remain owned by the enclosing Session.
    MemoryExecutionReport finish();
    // True only after physical drain AND a clean terminal memory phase.
    // Releasing native wrappers alone never authorizes Session reuse.
    bool abort(std::chrono::milliseconds timeout, const std::string& reason) noexcept;
    bool cleanup_quarantined() const noexcept;
    std::string_view cleanup_failure() const noexcept;
    void close_memory_binding() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tc::qwen21
