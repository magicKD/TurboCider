// Native end-to-end smoke/performance entry, not installed as the App executor.
#include "../../native/models/wan/pipeline.hpp"
#include "../../native/core/unigram_tokenizer.hpp"
#include "../../native/media/video.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        bool compiled = false;
        int repeat = 1;
        bool repeat_set = false;
        std::filesystem::path hybrid_manifest;
        while (argc > 1) {
            if (std::string(argv[argc - 1]) == "--compiled") {
                tc::require(!compiled, "duplicate --compiled");
                compiled = true;
                --argc;
            } else if (argc > 2 && std::string(argv[argc - 2]) == "--repeat") {
                tc::require(!repeat_set, "duplicate --repeat");
                std::string count = argv[argc - 1];
                tc::require(count.size() == 1 && count[0] >= '1' && count[0] <= '4',
                            "repeat must be 1...4");
                repeat = count[0] - '0';
                repeat_set = true;
                tc::require(repeat >= 1 && repeat <= 4, "repeat must be 1...4");
                argc -= 2;
            } else if (argc > 2 && std::string(argv[argc - 2]) == "--hybrid") {
                tc::require(hybrid_manifest.empty(), "duplicate --hybrid");
                hybrid_manifest = std::filesystem::absolute(argv[argc - 1]);
                argc -= 2;
            } else break;
        }
        tc::require(argc == 5 || argc == 8,
            "usage: wan-generate-probe MODEL TAEHV PROMPT OUTPUT.mp4 [WIDTH HEIGHT FRAMES] [--compiled | --hybrid MANIFEST] [--repeat 1...4]");
        const auto root = std::filesystem::absolute(argv[1]);
        const auto decoder = std::filesystem::absolute(argv[2]);
        tc::wan::GenerateOptions options;
        if (argc == 8) { options.width = std::stoi(argv[5]); options.height = std::stoi(argv[6]); options.frames = std::stoi(argv[7]); }
        options.compile_dit = compiled;
        options.hybrid_manifest = hybrid_manifest;
        std::atomic<bool> cancelled{false};
        bool prompt_cache_hit = false;
        std::unordered_map<std::string, tc::Clock::time_point> phase_start;
        tc::Event progress = [&](const std::string &phase, int done, int total) {
            if (phase == "umt5_prompt_cache") prompt_cache_hit = true;
            if (done == 0) phase_start[phase] = tc::Clock::now();
            if (done == total && phase_start.count(phase))
                std::cerr << phase << " seconds=" << std::chrono::duration<double>(
                    tc::Clock::now() - phase_start.at(phase)).count() << '\n';
            if (done == 0 || done == total) std::cerr << phase << ' ' << done << '/' << total << '\n';
        };
        tc::UnigramTokenizer tokenizer(root / "tokenizer");
        const auto tokens = tokenizer.prompt(argv[3]);
        std::vector<std::filesystem::path> outputs;
        for (int run = 0; run < repeat; ++run) {
            auto output = std::filesystem::absolute(argv[4]);
            if (run) output.replace_filename(output.stem().string() + "-repeat" + std::to_string(run) + output.extension().string());
            tc::require(!std::filesystem::exists(output) && !std::filesystem::is_symlink(output),
                        "probe output already exists: " + output.string());
            outputs.push_back(output);
        }
        tc::wan::Pipeline pipeline(root, root / "text_encoder", decoder, progress, cancelled);
        for (int run = 0; run < repeat; ++run) {
            prompt_cache_hit = false;
            phase_start.clear();
            tc::mx::reset_peak_memory();
            const auto start = tc::Clock::now();
            auto pixels = pipeline.generate(tokens, options, progress, cancelled);
            tc::require(pixels.shape() == tc::mx::Shape{1, options.frames, 3, options.height, options.width},
                        "Wan generated incorrect output dimensions");
            auto rgb = tc::mx::astype(tc::mx::contiguous(tc::mx::transpose(pixels, {0, 1, 3, 4, 2})) *
                                      tc::Tensor(255.f), tc::mx::uint8);
            tc::mx::eval(rgb);
            const auto &output = outputs.at(run);
            tc::write_video_rgb24(output, rgb.data<uint8_t>(), options.frames, options.width, options.height, options.fps);
            auto info = tc::probe_video(output);
            tc::require(info.frames == options.frames && info.width == options.width && info.height == options.height,
                        "Wan encoded video failed geometry verification");
            std::cout << "Native Wan video written: " << output << "; seconds="
                      << std::chrono::duration<double>(tc::Clock::now() - start).count()
                      << "; run=" << run << "; prompt_cache_hit=" << prompt_cache_hit
                      << "; compiled=" << compiled << "; mlx_peak_bytes=" << tc::mx::get_peak_memory() << std::endl;
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
