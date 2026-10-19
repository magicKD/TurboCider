#pragma once
#include "../../backends/mlx.hpp"
#include "../../runtime/session.hpp"
#include <array>

namespace tc::qwen21 {
// Fixed512-row Private W8 QKV owner. Header/layout stays the same in public
// and private builds; Objective-C and the private implementation stay out of it.
class RuntimePrivateQkv {
  public:
    using Gpu=std::function<Tensor(const Tensor&)>;
    RuntimePrivateQkv(size_t budget,std::atomic<bool>&);
    // Failed peer: no Device/Program allocation or retry at all.
    explicit RuntimePrivateQkv(const std::string &failure_reason);
    ~RuntimePrivateQkv();
    RuntimePrivateQkv(const RuntimePrivateQkv&)=delete;
    RuntimePrivateQkv&operator=(const RuntimePrivateQkv&)=delete;
    void begin_request(const std::string &source_generation);
    void begin_request(const std::string &source_generation,bool prefetch);
    bool eligible(int rows,bool permit_private=true) const;
    bool plan(int rows);
    bool plan(int rows,bool permit_private); // explicit policy decline, same GPU credit, no stage/submission
    void stage(int layer,int rows,std::array<Tensor,3>);
    bool prefetch(int layer,int rows,std::array<Tensor,3>); // one future, only after current QKV has completed
    QwenPrivateQkvPrefetchMetrics prefetch_metrics() const;
    Tensor run(int layer,const Tensor&,const Gpu&,std::atomic<bool>&);
    void drain();
    void disable(const std::string &failure_reason);
    bool available()const;
    const QwenPrivateQkvMetrics &metrics()const{return metrics_;}
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    QwenPrivateQkvMetrics metrics_;
    size_t budget_;
    std::string source_generation_;
    void fail(const std::string&);
};
} // namespace tc::qwen21
