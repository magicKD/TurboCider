#include "../../native/backends/mlx.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {try {
    using namespace tc;configure_streams();
    const std::vector<std::string> names{"fp32_repeat","fp32_gqa","native_repeat","native_gqa"};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-encoder-attention-component-v1\","
        "\"scope\":\"synthetic original-dtype post-norm Q/K/V, language heads32/kv8/d128; causal and padded-key mask, complete compiled attention/casts/repeat/eval; not original weights/full encoder/model/physical trace\",\"cases\":[";
    bool comma=false;
    for(auto dtype:{mx::bfloat16,mx::float16})for(int rows:{33,318,574,1024}) {
        auto q=mx::astype(mx::random::normal({1,32,rows,128},mx::float32,mx::random::key(123))*.25f,dtype);
        auto k=mx::astype(mx::random::normal({1,8,rows,128},mx::float32,mx::random::key(124))*.25f,dtype);
        auto v=mx::astype(mx::random::normal({1,8,rows,128},mx::float32,mx::random::key(125))*.25f,dtype);
        auto index=mx::arange(rows,mx::int32);auto key=mx::reshape(index,{1,rows}),query=mx::reshape(index,{rows,1});
        auto mask=mx::reshape(mx::where(mx::logical_or(key>query,key>=Tensor(rows-1)),Tensor(-INFINITY),Tensor(0.f)),{1,1,rows,rows});
        mx::eval({q,k,v,mask});
        using Fn=std::function<std::vector<Tensor>(const std::vector<Tensor>&)>;std::vector<Fn> recipes;
        for(int i=0;i<4;++i)recipes.push_back(mx::compile([i](const std::vector<Tensor>&a) {
            auto kk=(i%2)?a[1]:mx::repeat(a[1],4,1),vv=(i%2)?a[2]:mx::repeat(a[2],4,1);
            auto mm=i<2?a[3]:mx::astype(a[3],a[0].dtype());
            return std::vector<Tensor>{attend(a[0],kk,vv,i<2,mm)};
        }));
        const std::vector<Tensor> args{q,k,v,mask};auto expected=recipes[0](args)[0];mx::eval(expected);
        std::vector<double> errors;std::vector<std::vector<double>> times(4);
        for(auto &fn:recipes) {
            auto y=fn(args)[0];mx::eval(y);
            require(y.dtype()==dtype && y.shape()==expected.shape() && mx::all(mx::isfinite(y)).item<bool>(),"attention finite/shape/dtype mismatch");
            const auto error=mx::sqrt(mx::sum(mx::square(mx::astype(y,mx::float32)-mx::astype(expected,mx::float32)))/
                mx::sum(mx::square(mx::astype(expected,mx::float32)))).item<float>();
            require(std::isfinite(error) && error<.05f,"attention candidate exceeds5% component budget");errors.push_back(error);
            for(int warm=0;warm<3;++warm)mx::eval(fn(args));
        }
        for(int iteration=0;iteration<15;++iteration)for(size_t j=0;j<4;++j) {
            const auto i=(iteration+j)%4;const auto start=Clock::now();mx::eval(recipes[i](args));
            times[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        if(comma)std::cout<<',';comma=true;std::cout<<"{\"rows\":"<<rows<<",\"dtype\":\""<<(dtype==mx::bfloat16?"bf16":"fp16")<<"\",\"recipes\":[";
        for(size_t i=0;i<4;++i) {
            if(i)std::cout<<',';std::cout<<"{\"name\":\""<<names[i]<<"\",\"relative_l2\":"<<errors[i]<<",\"seconds\":[";
            for(size_t j=0;j<times[i].size();++j){if(j)std::cout<<',';std::cout<<times[i][j];}std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"qualification_passed\":false}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;} }
