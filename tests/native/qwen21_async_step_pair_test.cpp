#include "../../native/models/qwen21/async_step_pair.hpp"
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace tc;
bool same(const Tensor &a,const Tensor &b){
    require(a.shape()==b.shape()&&a.dtype()==b.dtype(),"raw state type/geometry differs");
    auto x=mx::view(mx::contiguous(a),mx::uint8),y=mx::view(mx::contiguous(b),mx::uint8);mx::eval(x,y);
    return std::equal(x.data<uint8_t>(),x.data<uint8_t>()+x.size(),y.data<uint8_t>());
}
void parity(int steps){
    auto start=mx::astype(mx::reshape(mx::sin(mx::arange(33*64,mx::float32)*.017f),{1,33,64}),mx::bfloat16);
    auto weight=mx::eye(64,mx::bfloat16);mx::eval(start,weight);
    auto reference=start;std::vector<Tensor> expected;
    for(int i=0;i<steps;++i){reference=reference+mx::matmul(reference,weight)*Tensor(-.125f,mx::bfloat16);mx::eval(reference);
        require(mx::all(mx::isfinite(reference)).item<bool>(),"nonfinite ordinary fixture");expected.push_back(reference);}
    auto state=start;qwen21::AsyncStepPair pair(steps);std::vector<int> progress;
    for(int i=0;i<steps;++i){pair.note_next_graph();state=state+mx::matmul(state,weight)*Tensor(-.125f,mx::bfloat16);pair.submit(i,state);
        if(pair.full()||i==steps-1){auto done=pair.drain();progress.insert(progress.end(),done.begin(),done.end());
            require(same(state,expected[i]),"bounded async changed BF16 state bits");}}
    require(pair.complete()&&pair.submitted()==uint64_t(steps)&&pair.checked()==uint64_t(steps)&&pair.accepted()==uint64_t(steps)&&
        pair.evaluation_calls()==uint64_t(2*steps)&&pair.batches()==uint64_t((steps+1)/2)&&pair.peak_pending()==2,"async full-step/check/batch coverage differs");
    require(pair.peak_logical_bytes()==2*(start.nbytes()+1),"logical state/mask ownership differs");
    require(progress.size()==size_t(steps),"completed progress missing");for(int i=0;i<steps;++i)require(progress[i]==i,"progress order differs");
}
void nonfinite(){
    auto finite=mx::ones({1,33,64},mx::bfloat16);
    auto bad=finite*Tensor(std::numeric_limits<float>::infinity(),mx::bfloat16);
    qwen21::AsyncStepPair pair(2);pair.submit(0,bad);
    auto recovered=mx::where(mx::isfinite(bad),bad,mx::zeros_like(bad));pair.submit(1,recovered);
    bool rejected=false;try{(void)pair.drain();}catch(const std::exception &e){rejected=std::string(e.what()).find("step 0")!=std::string::npos;}
    require(rejected&&!pair.complete()&&pair.empty()&&pair.checked()==2&&pair.accepted()==1,"recovered final state hid an earlier nonfinite step");
    mx::eval(recovered);require(mx::all(mx::isfinite(recovered)).item<bool>(),"negative-control final state was not actually finite");
}
void scope_exit(){
    auto state=mx::full({1,1024,2048},.125f,mx::bfloat16),w=mx::eye(2048,mx::bfloat16);mx::eval(state,w);
    auto result=state;for(int i=0;i<12;++i)result=mx::matmul(result,w);
    bool interrupted=false;
    try{qwen21::AsyncStepPair pair(1);pair.submit(0,result);require(pair.pending()==1,"scope-exit fixture did not retain its pending roots");
        throw std::runtime_error("controlled host abort");}catch(const std::exception &e){interrupted=std::string(e.what())=="controlled host abort";}
    require(interrupted&&result.is_available(),"scope exit returned without result-dependent completion");
    require(same(result,state),"scope-exit drain changed dependency chain");
}
void invalid(){
    auto state=mx::ones({1,33,64},mx::bfloat16);bool rejected=false;
    try{qwen21::AsyncStepPair bad(7);}catch(const std::exception&){rejected=true;}require(rejected,"unbounded horizon admitted");
    qwen21::AsyncStepPair pair(3);
    auto deny=[&](int step,const Tensor &value){bool denied=false;try{pair.submit(step,value);}catch(const std::exception&){denied=true;}require(denied,"invalid async submission admitted");};
    deny(1,state);deny(0,mx::astype(state,mx::float32));require(pair.empty()&&pair.submitted()==0,"failed admission mutated pending state");
    pair.submit(0,state);pair.submit(1,state);deny(2,state);require(pair.pending()==2&&pair.submitted()==2,"capacity refusal mutated prior roots");
    (void)pair.drain();pair.submit(2,state);(void)pair.drain();require(pair.complete(),"valid retry after host admission refusal failed");
}
}
int main(){try{tc::configure_streams();parity(5);parity(6);nonfinite();scope_exit();invalid();
    std::cout<<"{\"five_six_state_bits_exact\":true,\"per_step_nonfinite_recovery_rejected\":true,\"scope_exit_drained\":true,\"bounded_capacity_and_progress\":true}\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
