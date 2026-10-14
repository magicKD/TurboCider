#include "../../native/media/video.hpp"
#include "../../native/core/common.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <thread>

namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    check(rejected,"invalid streaming writer operation was accepted");
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 2,"a fresh output directory is required");
        const std::filesystem::path root(argv[1]);
        check(!std::filesystem::exists(root),"writer test output already exists");
        std::filesystem::create_directories(root);
        constexpr int width = 64, height = 64;
        std::vector<uint8_t> pixels(size_t(width)*height*3,127);
        for (int fps : {16,24}) {
            const auto path = root / ("fps"+std::to_string(fps)+".mp4");
            tc::RGB24VideoWriter writer(path,9,width,height,fps);
            for (int frame = 0; frame < 9; ++frame) {
                std::fill(pixels.begin(),pixels.end(),uint8_t(frame*20));
                writer.append(frame,pixels.data(),pixels.size());
            }
            check(!std::filesystem::exists(path),"streaming writer published incomplete video");
            writer.finish();
            const auto info = tc::probe_video(path);
            check(info.frames == 9 && info.fps == fps && info.width == width && info.height == height,"writer geometry mismatch");
            rejects([&] { writer.finish(); });
            rejects([&] { writer.append(9,pixels.data(),pixels.size()); });
        }
        {
            tc::RGB24VideoWriter writer(root/"slow-fps24.mp4",9,width,height,24);
            for (int frame = 0; frame < 9; ++frame) {
                if (frame >= 1 && frame <= 3) std::this_thread::sleep_for(std::chrono::milliseconds(2200));
                std::fill(pixels.begin(),pixels.end(),uint8_t(frame*20));
                writer.append(frame,pixels.data(),pixels.size());
            }
            writer.finish();
            check(tc::probe_video(root/"slow-fps24.mp4").frames == 9,"slow producer truncated frames");
        }
        const auto existing = root / "preserve.mp4";
        { std::ofstream file(existing); file << "existing user artifact"; }
        {
            tc::RGB24VideoWriter writer(existing,2,width,height,24);
            rejects([&] { writer.append(1,pixels.data(),pixels.size()); });
            rejects([&] { writer.append(0,pixels.data(),pixels.size()-1); });
            writer.append(0,pixels.data(),pixels.size());
            rejects([&] { writer.finish(); });
        }
        { std::ifstream file(existing); std::string text; std::getline(file,text); check(text == "existing user artifact","failed writer damaged destination"); }
        const auto cancelled_path = root / "cancelled.mp4";
        std::atomic<bool> cancelled{false};
        {
            tc::RGB24VideoWriter writer(cancelled_path,2,width,height,24,&cancelled);
            writer.append(0,pixels.data(),pixels.size());
            cancelled = true;
            bool typed = false;
            try { writer.append(1,pixels.data(),pixels.size()); } catch (const tc::Cancelled &) { typed = true; }
            check(typed,"writer cancellation lost native cancellation type");
            rejects([&] { writer.finish(); });
        }
        check(!std::filesystem::exists(cancelled_path),"cancelled writer published output");
        const auto long_path = root / "long1000.mp4";
        {
            tc::RGB24VideoWriter writer(long_path,1000,width,height,24);
            for (int frame = 0; frame < 1000; ++frame) writer.append(frame,pixels.data(),pixels.size());
            writer.finish();
        }
        check(tc::probe_video(long_path).frames == 1000,"long producer truncated frames");
        for (const auto &entry : std::filesystem::directory_iterator(root))
            check(entry.path().filename().string().find(".mp4.") == std::string::npos,"writer leaked staging file");
        std::cout << "PASS: streaming writer FPS/frame count, order/size validation, backpressure, cancellation, preserved destination and 1000-frame producer\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
