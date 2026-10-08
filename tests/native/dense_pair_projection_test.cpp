#include "../../native/backends/dense_gpu_pair.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iostream>

namespace mx=mlx::core;
int main(){try {
    int cases=0;
    for(auto dtype:{mx::bfloat16,mx::float16})for(int rows:{1,33,67})for(bool strided:{false,true}) {
        auto w=mx::astype(mx::reshape(mx::sin(mx::arange(1792*768,mx::float32)*.003f)*.04f,{1792,768}),dtype);
        auto source=mx::astype(mx::reshape(mx::cos(mx::arange(rows*512,mx::float32)*.017f)*.2f,
            strided?mx::Shape{512,rows}:mx::Shape{rows,512}),dtype);
        auto x=mx::expand_dims(strided?mx::transpose(source):source,0);mx::eval({x,w});
        for(int bm:{16,32,64})for(int bn:{64,128})for(bool fp32:{false,true}) {
            auto first=tc::dense_gpu::projection_range(x,w,128,768,128,640,bm,fp32,bn);
            auto second=tc::dense_gpu::projection_range(x,w,1024,1664,128,640,bm,fp32,bn);
            auto expected=mx::concatenate({first,second},-1);
            auto got=tc::dense_gpu::projection_row_pair(x,w,128,1024,640,128,640,bm,bn,fp32);mx::eval({expected,got});
            if(got.shape()!=expected.shape() || got.dtype()!=expected.dtype() ||
                !mx::all(mx::view(got,mx::uint8)==mx::view(expected,mx::uint8)).item<bool>())
                throw std::runtime_error("paired physical projection changed original range results");
            ++cases;
        }
        for(int bad:{-1,0,63,129,896,2048}) {
            bool rejected=false;
            try{tc::dense_gpu::projection_row_pair(x,w,128,1024,bad,128,640);}
            catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("invalid paired extent/seam admitted");
        }
    }
    std::cout<<"PASS paired dense projection cases="<<cases<<": two dtypes, tails/strided input, physical nonzero row/column offsets, all six tile recipes, F32/narrow output and seam/extent guards\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
