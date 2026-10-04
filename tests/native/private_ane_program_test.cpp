#include "../../native/backends/private/ane_program.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace tc::ane;
using namespace tc::ane::private_api;
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) return 2;
    try {
        Device device;
        GraphShape shape{Kind::Matmul, 64, 64, 64, 64, 64, false};
        const auto mil = fp16_program(shape);
        Program program(device, mil, {}, argv[1]);
        Surface x(device, 64, 64, Element::FP16), w(device, 64, 64, Element::FP16), y(device, 64, 64, Element::FP16);
        Surface i8(device, 3, 65, Element::I8);
        if (i8.pitch() != 128 || i8.bytes() < i8.pitch() * i8.rows()) throw std::runtime_error("INT8 surface layout");
        auto at = [](Surface &s, int row, int col) -> _Float16 & {
            return reinterpret_cast<_Float16 *>(static_cast<char *>(s.data()) + size_t(row) * s.pitch())[col];
        };
        for (int trial = 0; trial < 3; ++trial) {
            const float scale = trial == 1 ? -.25f : .125f;
            std::memset(w.data(), 0, w.rows() * w.pitch());
            for (int r = 0; r < 64; ++r) { at(w, r, r) = _Float16(scale); for (int c = 0; c < 64; ++c) at(x, r, c) = _Float16(((r * 3 + c * 7) % 17 - 8) / 8.f); }
            std::pair<std::string, Surface> inputs[]{{"w", w}, {"x", x}}, outputs[]{{"y", y}};
            const uint64_t start = uint64_t(trial) * 2 + 1, done = start + 1;
            // Queue ANE before the GPU signal. Without a hardware wait, this
            // could read unready input; the output must remain at its poison.
            std::memset(y.data(), 0x7f, y.rows() * y.pitch());
            auto ticket = program.enqueue(inputs, outputs, start, done);
            device.signal(start);
            if (!device.wait(done, std::chrono::seconds(30))) throw std::runtime_error("GPU shared-event wait failed");
            auto result = ticket.finish();
            if (!result.ok) throw std::runtime_error(result.error);
            for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
                if (std::abs(float(at(y, r, c)) - float(at(x, r, c)) * scale) > .0002f) throw std::runtime_error("ANE numerical/A-B-A mismatch");
        }
        Program cached(device, mil, {}, argv[1]);
        if (cached.cache_key() != program.cache_key() || cached.compiled_now()) throw std::runtime_error("program cache reuse failed");
        GraphShape ffn_shape{Kind::SwiGLU, 33, 64, 96, 32, 48, false};
        Program ffn(device, fp16_program(ffn_shape), {}, argv[1]);
        Surface fx(device, 64, 33, Element::FP16), wg(device, 96, 64, Element::FP16),
            wu(device, 96, 64, Element::FP16), wd(device, 64, 96, Element::FP16), fy(device, 64, 33, Element::FP16);
        for (auto *surface : {&wg, &wu, &wd}) {
            std::memset(surface->data(), 0, surface->rows() * surface->pitch());
            for (uint32_t r = 0; r < surface->rows(); ++r) at(*surface, r, r % surface->columns()) = _Float16(.125f);
        }
        for (int c = 0; c < 64; ++c) for (int r = 0; r < 33; ++r) at(fx, c, r) = _Float16(((r * 3 + c * 7) % 17 - 8) / 8.f);
        std::pair<std::string, Surface> ffn_inputs[]{{"x", fx}, {"wg", wg}, {"wu", wu}, {"wd", wd}}, ffn_outputs[]{{"y", fy}};
        auto ft = ffn.enqueue(ffn_inputs, ffn_outputs, 7, 8); device.signal(7);
        if (!ft.finish().ok || !device.wait(8, std::chrono::seconds(30))) throw std::runtime_error("FFN event failure");
        for (int c = 0; c < 64; ++c) for (int r = 0; r < 33; ++r) {
            const float gate = float(at(fx, c, r)) * .125f;
            const float expected = gate / (1 + std::exp(-gate)) * gate * .125f;
            const float actual = float(at(fy, c, r));
            if (std::abs(actual - expected) > .0002f + .02f * std::abs(expected))
                throw std::runtime_error("FFN numerical mismatch r=" + std::to_string(r) + " c=" + std::to_string(c) +
                    " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
        }
        if (argc == 3 && std::string(argv[2]) == "timeout") {
            // No GPU producer for value9. Deadline releases GPU wait10, never
            // returns successful output, and disables further private work.
            auto stalled = ffn.enqueue(ffn_inputs, ffn_outputs, 9, 10);
            auto timeout = stalled.finish(std::chrono::milliseconds(1));
            if (timeout.ok || !timeout.timed_out || Program::healthy() || !device.wait(10, std::chrono::seconds(30)))
                throw std::runtime_error("timeout safety-release/health failure");
            try { Program disabled(device, mil, {}, argv[1]); throw std::runtime_error("unhealthy program accepted"); }
            catch (const CapabilityError &) {}
            std::cout << "PASS private timeout releases GPU wait, no success, process disabled, async owners retained\n";
        }
        std::cout << "PASS private ANE GPU-signal/ANE/GPU-wait, named bindings, A/B/A, cache; device=" << device.name()
                  << " compiled_now=" << program.compiled_now() << " slot_bytes=" << x.bytes() + w.bytes() + y.bytes() << "\n";
    } catch (const std::exception &error) { std::cerr << error.what() << "\n"; return 1; }
}
