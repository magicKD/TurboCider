// CPU-only processor lifecycle screen. No model weights or GPU evaluation.
#include "../../native/models/qwen21/encoder_prefill.hpp"
#include "../../native/platform/apple/bridge.hpp"
#include "../../native/platform/apple/platform.hpp"
#include "turbocider/turbocider.h"
#include <fstream>
#include <iostream>

using namespace tc;
namespace {
double elapsed(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
NSString *ns(const std::string &s) {return [NSString stringWithUTF8String:s.c_str()];}
const std::array<std::string,6> prompts{
    "A blue ceramic teapot on a wooden table, soft daylight.",
    "A quiet lake in a mountain valley at sunrise.",
    "Change the teapot to bright red. Preserve the handle and background.",
    "Add exactly one yellow lemon to the left of the blue teapot.",
    "一张白底海报，清晰写着“秋日茶会”，下方放着蓝色陶瓷茶壶。",
    "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|> Make the teapot glass.<|im_end|>"};
}
int main(int argc,const char **argv) {@autoreleasepool {
    try {
        require(argc==3,"usage: tokenizer-reuse-probe PROCESSOR FRESH_OUTPUT_JSON");
        const auto root=std::filesystem::canonical(argv[1]),out=std::filesystem::absolute(argv[2]);
        require(!std::filesystem::exists(out),"fresh CPU receipt required");
        const auto source=root/"tokenizer.json";const auto source_hash=sha256_file(source);
        std::atomic<bool> cancelled{false};
        const auto setup_start=Clock::now();qwen21::VerifiedEncoderTokenizer retained(root,cancelled);
        const double cold=elapsed(setup_start);
        std::vector<Tokens> expected;
        { Tokenizer ordinary(root);for(const auto &prompt:prompts)expected.push_back(ordinary.raw_bounded(prompt,2048)); }
        NSMutableArray *samples=[NSMutableArray array];
        for(int iteration=-1;iteration<6;++iteration)for(int visit=0;visit<2;++visit) {@autoreleasepool {
            const bool reuse=((iteration+visit+2)%2)!=0;
            const auto &prompt=prompts[(iteration+6)%6];
            const auto start=Clock::now();Tokens tokens;
            if(reuse) {
                retained.check_unchanged();tokens=retained.tokenizer().raw_bounded(prompt,2048);retained.check_unchanged();
            } else {
                Tokenizer tokenizer(root);tokens=tokenizer.raw_bounded(prompt,2048);
            }
            const double wall=elapsed(start);
            const auto &reference=expected[(iteration+6)%6];
            require(tokens.ids==reference.ids && tokens.valid==reference.valid,"retained tokenizer changed IDs");
            [samples addObject:@{@"iteration":@(iteration),@"reuse":@(reuse),@"seconds":@(wall),
                                 @"token_count":@(tokens.valid)}];
        }}
        require(sha256_file(source)==source_hash && retained.sha256()==source_hash,"processor identity changed");
        char *runtime=tc_runtime_build_identity();require(runtime,"runtime identity missing");
        NSDictionary *report=@{@"schema":@"tc-qwen-tokenizer-reuse-cpu-v1",@"status":@"complete",
            @"qualification_passed":@NO,@"device_calls":@0,@"source_sha256":ns(source_hash),
            @"source_bytes":@(std::filesystem::file_size(source)),@"loaded_runtime_id":ns(runtime),
            @"retained_cold_setup_seconds":@(cold),@"samples":samples,@"all_token_ids_equal":@YES,
            @"scope":@"CPU constructor+tokenization+destruction vs verified retained tokenizer with pre/post source-generation checks; not full request/GPU timing"};
        tc_string_free(runtime);std::ofstream file(out);file<<json(report)<<'\n';std::cout<<json(report)<<'\n';
        return 0;
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}}
