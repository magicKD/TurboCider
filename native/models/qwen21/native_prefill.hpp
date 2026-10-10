#pragma once
#include "../../backends/mlx.hpp"
#include "../../runtime/session.hpp"
namespace tc::qwen21 {
// Idle Session-owned operators/program/scratch only. Each call stages current
// immutable checkpoint views again; no conditioning/KV/LoRA source cache.
class NativePrefill {
public:
    explicit NativePrefill(const std::filesystem::path&);
    ~NativePrefill();
    bool healthy() const;
    const std::filesystem::path& cache() const;
    double cold_preparation_seconds() const;
    void begin_request(bool prepared);
    Tensor execute(const Tensor&,const Tensor& gate_up,const Tensor& down,std::atomic<bool>&);
    NativePrefillExecutionMetrics metrics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
