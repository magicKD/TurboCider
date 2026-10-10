#include "../../native/backends/mlx.hpp"
#include "../../native/backends/dense_gpu_projection.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

// Real checkpoint down weight, synthetic finite hidden. No model download,
// learned-weight copy on disk, activation dump or timing qualification.
int main(int argc,char **argv) {
    try {
        using namespace tc;
        require(argc==2,"usage: qwen-down-kernel-probe local-diffusion-checkpoint");
        configure_streams();Weights weights;
        weights.load_file(argv[1],"transformer_blocks.0.img_mlp.out.");weights.materialize();
        const auto &down=weights.at("weight");
        require(down.shape()==mx::Shape{4096,12288} && down.dtype()==mx::bfloat16,"local Qwen BF16 down fixture mismatch");
        struct Recipe{int bm,bn;bool static_tiles;};
        const std::vector<Recipe> recipes{{32,128,false},{16,64,false},{32,64,false},{64,64,false},
            {64,128,false},{32,128,true},{64,64,true},{64,128,true}};
        std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen-down-kernel-screen-v1\","
            "\"scope\":\"serial operator host span; same physical weight view; synthetic hidden; not full FFN/request/physical GPU trace\","
            "\"warmup_per_recipe\":3,\"measured_per_recipe\":9,\"cases\":[";
        bool comma=false;
        for(int rows:{1024,1056,3137})for(int columns:{3072,7168,10752}) {
            auto input=mx::astype(mx::sin(mx::arange(rows*columns,mx::float32)*.001f)*.125f,mx::bfloat16);
            input=mx::reshape(input,{1,rows,columns});mx::eval(input);
            std::vector<double> first_error;std::vector<std::vector<double>> seconds(recipes.size());
            auto baseline=dense_gpu::projection_range(input,down,0,4096,0,columns,32,true);mx::eval(baseline);
            for(size_t i=0;i<recipes.size();++i) {
                const auto r=recipes[i];
                auto value=dense_gpu::projection_range(input,down,0,4096,0,columns,r.bm,true,r.bn,r.static_tiles);mx::eval(value);
                const float error=mx::sqrt(mx::sum(mx::square(value-baseline))/mx::sum(mx::square(baseline))).item<float>();
                require(std::isfinite(error) && error<.002f,"down kernel numerical screen failed");first_error.push_back(error);
                for(int warm=0;warm<3;++warm)mx::eval(dense_gpu::projection_range(input,down,0,4096,0,columns,r.bm,true,r.bn,r.static_tiles));
            }
            // Cyclic start offset prevents one recipe from always going first.
            for(int iteration=0;iteration<9;++iteration)for(size_t visit=0;visit<recipes.size();++visit) {
                const size_t i=(visit+iteration)%recipes.size();const auto r=recipes[i];const auto start=Clock::now();
                auto value=dense_gpu::projection_range(input,down,0,4096,0,columns,r.bm,true,r.bn,r.static_tiles);mx::eval(value);
                seconds[i].push_back(std::chrono::duration<double>(Clock::now()-start).count());
            }
            if(comma)std::cout<<',';comma=true;
            std::cout<<"{\"rows\":"<<rows<<",\"columns\":"<<columns<<",\"output_columns\":4096,\"physical_weight_pitch\":12288,\"recipes\":[";
            for(size_t i=0;i<recipes.size();++i) {
                if(i)std::cout<<',';const auto r=recipes[i];
                std::cout<<"{\"bm\":"<<r.bm<<",\"bn\":"<<r.bn<<",\"static_requested\":"<<(r.static_tiles?"true":"false")
                    <<",\"static_selected\":"<<(r.static_tiles && rows%r.bm==0 ? "true":"false")<<",\"rel_l2_vs_original\":"<<first_error[i]<<",\"seconds\":[";
                for(size_t sample=0;sample<seconds[i].size();++sample){if(sample)std::cout<<',';std::cout<<seconds[i][sample];}
                std::cout<<"]}";
            }
            std::cout<<"]}";
        }
        std::cout<<"],\"qualification_passed\":false}\n";
    } catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
