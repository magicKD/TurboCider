#pragma once
#include "../../backends/mlx.hpp"
#include <algorithm>

namespace tc::qwen21 {
// Request-owned, GPU-only two-step submission window. Preserve the original
// state evaluation boundary, then enqueue its own finite check separately.
// A successful drain validates EVERY submitted step before progress/publish.
class AsyncStepPair {
    struct Pending { int step; Tensor state; std::optional<Tensor> finite; };
    int expected_;
    std::vector<Pending> pending_;
    uint64_t submitted_=0,checked_=0,accepted_=0,batches_=0,eval_calls_=0,peak_=0,peak_bytes_=0,host_ahead_=0;
    bool failed_=false;
    std::vector<Tensor> roots() const {
        std::vector<Tensor> values;values.reserve(pending_.size()*2);
        for(auto it=pending_.rbegin();it!=pending_.rend();++it){values.push_back(it->state);if(it->finite)values.push_back(*it->finite);}
        return values;
    }
  public:
    explicit AsyncStepPair(int steps):expected_(steps) {
        require(steps>=1&&steps<=6,"bounded async steps require a one-to-six-step horizon");pending_.reserve(2);
    }
    AsyncStepPair(const AsyncStepPair&)=delete;
    AsyncStepPair& operator=(const AsyncStepPair&)=delete;
    ~AsyncStepPair() noexcept {
        if(pending_.empty())return;
        // This object dies BEFORE the request Transformer/weights. No timeout
        // is treated as completion; use MLX's own result-dependent wait.
        try{mx::eval(roots());}catch(...){try{mx::synchronize();}catch(...){}}
    }
    bool full()const{return pending_.size()==2;}
    bool empty()const{return pending_.empty();}
    size_t pending()const{return pending_.size();}
    bool any_not_ready()const {
        for(const auto &p:pending_)if(!p.state.is_available()||(p.finite&&!p.finite->is_available()))return true;
        return false;
    }
    void note_next_graph(){if(!pending_.empty()&&any_not_ready())++host_ahead_;}
    void submit(int step,const Tensor &state) {
        require(!failed_&&pending_.size()<2&&step==int(submitted_)&&step<expected_,"async step order/capacity/horizon differs");
        require(state.ndim()==3&&state.shape(0)==1&&state.size()>0&&state.dtype()==mx::bfloat16,"async step state must be a nonempty BF16 batch-one tensor");
        pending_.push_back({step,state,{}}); // own roots before any submission
        try{
            mx::async_eval({state});++eval_calls_;
            pending_.back().finite=mx::all(mx::isfinite(state));
            mx::async_eval({*pending_.back().finite});++eval_calls_;
        }catch(...){failed_=true;throw;}
        ++submitted_;peak_=std::max(peak_,uint64_t(pending_.size()));
        uint64_t bytes=0;for(const auto &p:pending_)bytes+=p.state.nbytes()+(p.finite?p.finite->nbytes():0);
        peak_bytes_=std::max(peak_bytes_,bytes);
    }
    std::vector<int> drain() {
        require(!failed_,"cannot publish a failed async batch");
        if(pending_.empty())return {};
        mx::eval(roots());++batches_;
        std::vector<int> done;done.reserve(pending_.size());int bad=-1;
        for(const auto &p:pending_){require(p.finite.has_value(),"missing per-step finite mask");++checked_;
            if(p.finite->item<bool>()){++accepted_;done.push_back(p.step);}else if(bad<0)bad=p.step;}
        pending_.clear(); // all dependency waits completed, including bad masks
        if(bad>=0){failed_=true;throw std::runtime_error("nonfinite Qwen21 latent at step "+std::to_string(bad));}
        return done;
    }
    bool complete()const{return !failed_&&pending_.empty()&&submitted_==uint64_t(expected_)&&checked_==submitted_&&accepted_==submitted_;}
    uint64_t submitted()const{return submitted_;}
    uint64_t checked()const{return checked_;}
    uint64_t accepted()const{return accepted_;}
    uint64_t batches()const{return batches_;}
    uint64_t evaluation_calls()const{return eval_calls_;}
    uint64_t peak_pending()const{return peak_;}
    uint64_t peak_logical_bytes()const{return peak_bytes_;}
    uint64_t host_ahead_observations()const{return host_ahead_;}
};
}
