#include "../../native/backends/private/ane_calibration.hpp"
#include "../../native/backends/private/ane_mil.hpp"
#include "../../native/core/gguf_decode.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>

namespace mx = mlx::core;
using namespace tc::ane;
using namespace tc::ane::private_api;
namespace {
void check(bool value, const char *reason) { if (!value) throw std::runtime_error(reason); }
DeviceWeightView weight(const mx::array &value) {
    return {const_cast<void *>(value.buffer().ptr()), value.buffer_size(), size_t(value.offset()), size_t(value.shape(1)) * 2,
            value.shape(0), value.shape(1), DeviceWeightEncoding::Dense, DType::BF16, 32, {}, {},
            std::make_shared<mx::array>(value)};
}
struct Bank {
    Surface g, sg, u, su, d, sd;
    Bank(Device &device, int h, int f)
        : g(device, f, h, Element::I8), sg(device, f, 1, Element::FP16),
          u(device, f, h, Element::I8), su(device, f, 1, Element::FP16),
          d(device, h, f, Element::I8), sd(device, h, 1, Element::FP16) {}
};
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    @autoreleasepool {
      try {
        constexpr int rows = 33, hidden = 128, width = 2560;
        // Fixture versions differ with depth; the fifth source is the future
        // weight set. These are NOT model checkpoints or performance claims.
        constexpr std::array<int, 5> depth{0, 7, 15, 23, 31};
        std::vector<std::array<mx::array, 3>> sources;
        sources.reserve(depth.size());
        for (size_t layer = 0; layer < depth.size(); ++layer) {
            std::vector<float> g(size_t(width) * hidden), u(g.size()), d(size_t(hidden) * width);
            for (int r = 0; r < width; ++r) {
                g[size_t(r) * hidden + r % hidden] = .125f + float(layer) * .0125f;
                u[size_t(r) * hidden + r % hidden] = -.25f + float(layer) * .01f;
                d[size_t(r % hidden) * width + r] = .03125f;
            }
            sources.push_back({mx::astype(mx::array(g.data(), {width, hidden}, mx::float32), mx::bfloat16),
                mx::astype(mx::array(u.data(), {width, hidden}, mx::float32), mx::bfloat16),
                mx::astype(mx::array(d.data(), {hidden, width}, mx::float32), mx::bfloat16)});
            for (auto &source : sources.back()) mx::eval(source);
        }
        auto input = mx::astype(mx::random::normal({rows, hidden}, mx::float32, mx::random::key(17)) * .25f, mx::bfloat16);
        mx::eval(input);
        for (int channels : {1024, 2048}) {
            Device device;
            GraphShape shape{Kind::SwiGLU, rows, hidden, channels, 128, 128, false};
            auto spec = w8_swiglu_program(shape, 20260930, 1.f);
            Program program(device, spec.mil, spec.constants, argv[1]);
            const int first = width - channels;
            std::vector<Bank> banks;
            std::vector<Surface> outputs;
            for (size_t layer = 0; layer < sources.size(); ++layer) {
                banks.emplace_back(device, hidden, channels);
                auto &bank = banks.back();
                auto g = device.stage_w8(weight(sources[layer][0]), {first, channels, 0, hidden, 128}, bank.g, bank.sg);
                auto u = device.stage_w8(weight(sources[layer][1]), {first, channels, 0, hidden, 128}, bank.u, bank.su);
                auto d = device.stage_w8(weight(sources[layer][2]), {0, hidden, first, channels, 512}, bank.d, bank.sd);
                const auto gr = g.finish(), ur = u.finish(), dr = d.finish();
                check(gr.ok && ur.ok && dr.ok, "calibration fixture W8 staging failed");
                outputs.emplace_back(device, spec.packed_rows, rows, Element::FP16);
            }
            Surface x(device, hidden, rows, Element::I8), tx(device, 1, rows, Element::FP16);
            auto activation = device.stage_w8(weight(input), {0, rows, 0, hidden, 128, 20260930, true}, x, tx);
            check(activation.finish().ok, "calibration fixture A8 staging failed");
            auto bindings = [&](int count) {
                std::vector<CalibrationBindings> result;
                for (int layer = 0; layer < count; ++layer) {
                    auto &bank = banks.at(layer);
                    result.push_back({{{"x", x}, {"tx", tx}, {"wg", bank.g}, {"sg", bank.sg},
                                       {"wu", bank.u}, {"su", bank.su}, {"wd", bank.d}}, {{"y", outputs.at(layer)}}});
                }
                return result;
            };
            uint64_t timeline = 0;
            // A prepared request is not a live evaluation. Releasing its wait
            // without submit must leave the poisoned output untouched.
            std::memset(outputs[0].data(), 0x7f, outputs[0].rows() * outputs[0].pitch());
            {
                auto row = bindings(1)[0];
                auto prepared = program.prepare(row.inputs, row.outputs, 1, 2);
                auto moved = std::move(prepared);
                device.release_prepared(1);
                check(static_cast<const uint8_t *>(outputs[0].data())[0] == 0x7f, "prepare issued driver work");
                bool rejected = false;
                try { prepared.submit(); } catch (const std::runtime_error &) { rejected = true; }
                check(rejected, "moved-from prepared request accepted");
                auto ticket = moved.submit();
                check(ticket.finish().ok && device.value() == 2, "prepared W8 request failed");
                rejected = false;
                try { moved.submit(); } catch (const std::runtime_error &) { rejected = true; }
                check(rejected, "prepared request submitted twice");
                timeline = 2;
            }
            bool nonzero = false;
            for (uint32_t r = 0; r < outputs[0].rows(); ++r) for (int c = 0; c < rows; ++c) {
                const auto bits = reinterpret_cast<const uint16_t *>(
                    static_cast<const uint8_t *>(outputs[0].data()) + r * outputs[0].pitch())[c];
                const float value = tc::gguf::fp16_to_float(bits);
                check(std::isfinite(value), "prepared W8 output left poison or became nonfinite");
                if (r < hidden) nonzero |= value != 0;
            }
            check(nonzero, "prepared W8 evaluation produced no nonzero partial");
            // Binding disposal must break the never-submitted callback cycle.
            for (int trial = 0; trial < 32; ++trial) {
                auto row = bindings(1)[0];
                auto discarded = program.prepare(row.inputs, row.outputs, 3, 4);
                auto replacement = program.prepare(row.inputs, row.outputs, 3, 4);
                replacement = std::move(discarded);
            }
            // A submitted ticket must outlive the originating Device,
            // Program, bindings and every input Surface wrapper. Retaining
            // only the output for the oracle must not keep the model alive.
            std::optional<Ticket> retained;
            auto lifetime_output = outputs[0];
            {
                Device temporary_device;
                Program temporary_program(temporary_device, spec.mil, spec.constants, argv[1]);
                auto original = bindings(1)[0];
                std::vector<std::pair<std::string, Surface>> local_inputs;
                for (const auto &source : original.inputs) {
                    Surface copy(temporary_device, source.second.rows(), source.second.columns(), source.second.element());
                    check(copy.pitch() == source.second.pitch(), "lifetime fixture surface pitch changed");
                    std::memcpy(copy.data(), source.second.data(), source.second.rows() * source.second.pitch());
                    local_inputs.emplace_back(source.first, std::move(copy));
                }
                lifetime_output = Surface(temporary_device, spec.packed_rows, rows, Element::FP16);
                std::vector<std::pair<std::string, Surface>> local_outputs{{"y", lifetime_output}};
                auto prepared = temporary_program.prepare(local_inputs, local_outputs, 1, 2);
                temporary_device.release_prepared(1);
                retained = prepared.submit();
            }
            check(retained->finish().ok, "prepared ticket lost input/model/event owners");
            for (uint32_t row = 0; row < lifetime_output.rows(); ++row)
                check(!std::memcmp(static_cast<const uint8_t *>(lifetime_output.data()) + row * lifetime_output.pitch(),
                                   static_cast<const uint8_t *>(outputs[0].data()) + row * outputs[0].pitch(), rows * 2),
                      "prepared ticket output changed after caller destruction");
            for (int count : {1, 4}) {
                CalibrationBatch warm(device, program, bindings(count), timeline);
                auto result = warm.measure(true);
                check(result.ok && result.ane_calls == uint64_t(count), "prepared W8 warmup did not execute every layer");
            }
            std::vector<uint8_t> reference(outputs[0].rows() * outputs[0].pitch());
            std::memcpy(reference.data(), outputs[0].data(), reference.size());
            std::array<std::array<double, 2>, 3> times{};
            for (int part = 0; part < 3; ++part) for (int count : {1, 4}) {
                std::vector<double> samples;
                for (int trial = 0; trial < 3; ++trial) {
                    std::vector<mx::array> gpu_results;
                    CalibrationBatch batch(device, program, bindings(count), timeline);
                    auto submit_gpu = [&] {
                        for (int layer = 0; layer < count; ++layer) {
                            auto g = mx::slice(sources[layer][0], {0, 0}, {first, hidden});
                            auto u = mx::slice(sources[layer][1], {0, 0}, {first, hidden});
                            auto d = mx::slice(sources[layer][2], {0, 0}, {hidden, first});
                            auto gate = mx::matmul(input, mx::transpose(g));
                            auto up = mx::matmul(input, mx::transpose(u));
                            auto z = (gate / (mx::array(1.f, gate.dtype()) + mx::exp(-gate))) * up;
                            gpu_results.push_back(mx::matmul(z, mx::transpose(d)));
                        }
                        mx::async_eval(gpu_results);
                    };
                    const auto result = part == 1 ? batch.measure(true) :
                        batch.measure(part == 2, submit_gpu, [&] { mx::eval(gpu_results); });
                    check(result.ok && std::isfinite(result.seconds) && result.seconds > 0 &&
                          result.ane_calls == uint64_t(part == 0 ? 0 : count), "invalid alone/concurrent batch receipt");
                    check(!std::memcmp(reference.data(), outputs[0].data(), reference.size()), "prepared repeated W8 output changed");
                    samples.push_back(result.seconds);
                    bool rejected = false;
                    try { batch.measure(true); } catch (const std::invalid_argument &) { rejected = true; }
                    check(rejected, "calibration batch reused");
                }
                std::sort(samples.begin(), samples.end());
                times[part][count == 1 ? 0 : 1] = samples[1];
            }
            // A GPU callback throwing after submission must still drain both
            // arms. The next independent request must remain usable.
            for (bool finish_throws : {false, true}) {
                CalibrationBatch batch(device, program, bindings(4), timeline);
                std::optional<mx::array> gpu;
                int drains = 0;
                bool caught = false;
                try {
                    batch.measure(true, [&] {
                        gpu = mx::matmul(input, mx::transpose(sources[0][0]));
                        mx::async_eval(*gpu);
                        if (!finish_throws) throw std::runtime_error("calibration GPU submit failed");
                    }, [&] {
                        ++drains; mx::eval(*gpu);
                        if (finish_throws) throw std::runtime_error("calibration GPU finish failed");
                    });
                } catch (const std::runtime_error &error) {
                    caught = std::string(error.what()) == (finish_throws ? "calibration GPU finish failed" : "calibration GPU submit failed");
                }
                check(caught && drains == 1 && Program::healthy(), "calibration error lost original exception or failed to drain");
                CalibrationBatch recovered(device, program, bindings(1), timeline);
                check(recovered.measure(true).ok, "calibration error broke next evaluation");
            }
            auto reject = [&](std::vector<CalibrationBindings> rows) {
                bool rejected = false;
                try { CalibrationBatch invalid(device, program, std::move(rows), timeline); }
                catch (const std::invalid_argument &) { rejected = true; }
                check(rejected, "invalid calibration bindings accepted");
            };
            reject({}); reject(bindings(5));
            auto aliases = bindings(4); aliases[1].outputs = aliases[0].outputs; reject(std::move(aliases));
            aliases = bindings(1); aliases[0].outputs[0].second = x; reject(std::move(aliases));
            auto malformed = bindings(4); malformed.back().inputs[0].first = "missing_binding";
            const auto before = device.value();
            bool malformed_rejected = false;
            try { CalibrationBatch invalid(device, program, std::move(malformed), timeline); }
            catch (const CapabilityError &) { malformed_rejected = true; }
            check(malformed_rejected && device.value() == before && Program::healthy(),
                  "partial binding failure issued work or disabled a healthy driver");
            {
                CalibrationBatch validated(device, program, bindings(1), timeline);
                for (bool incomplete_gpu : {false, true}) {
                    bool rejected = false;
                    try {
                        if (incomplete_gpu) validated.measure(true, [] {});
                        else validated.measure(false);
                    } catch (const std::invalid_argument &) { rejected = true; }
                    check(rejected, "incomplete/no-part calibration accepted");
                }
                check(validated.measure(true).ok, "option validation consumed an unsubmitted batch");
            }
            bool rejected = false;
            try { device.release_prepared(timeline - 1); } catch (const std::runtime_error &) { rejected = true; }
            check(rejected, "prepared event allowed a reused timeline");
            std::cout << "PASS prepared W8 calibration channels=" << channels << " share=" << double(channels) / width
                      << " depth=" << depth[0] << ',' << depth[1] << ',' << depth[2] << ',' << depth[3] << ',' << depth[4]
                      << " gpu_1=" << times[0][0] << " gpu_4=" << times[0][1]
                      << " ane_1=" << times[1][0] << " ane_4=" << times[1][1]
                      << " both_1=" << times[2][0] << " both_4=" << times[2][1]
                      << "; fixture GPU head only (no staging/restore); not model/E2E calibration or physical overlap proof\n";
        }
        std::cout << "PASS prepared calibration ownership, one-shot, unsubmitted disposal, alias/timeline rejection and GPU-error cleanup\n";
      } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
