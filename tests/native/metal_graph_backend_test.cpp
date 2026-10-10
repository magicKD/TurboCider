#include "../../native/backends/metal_graph/backend.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace tc::metal_graph;
using namespace std::chrono_literals;

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class Fn> void rejects(Fn&& fn, const char* message) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}
constexpr const char* source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Params { uint count; float factor; float bias; };
kernel void affine(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant Params& p [[buffer(2)]],
                   uint index [[thread_position_in_grid]]) {
    if (index < p.count) output[index] = input[index] * p.factor + p.bias;
}
kernel void copy_values(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < 32) output[index] = input[index];
}
)metal";
struct Params { uint32_t count; float factor, bias; };
struct AccountingToken {
    std::shared_ptr<std::atomic<size_t>> releases;
    size_t upper;
    ~AccountingToken() { ++*releases; }
};
struct ExternalStorage {
    Buffer owner;
    explicit ExternalStorage(Buffer buffer) : owner(std::move(buffer)) {}
};

void test_graph_and_ownership() {
    auto releases = std::make_shared<std::atomic<size_t>>(0);
    size_t reservations = 0, commits = 0;
    BackendOptions options;
    options.allocation.reserve = [&](size_t upper, std::string_view) {
        ++reservations;
        auto token = std::make_shared<AccountingToken>();
        token->releases = releases; token->upper = upper;
        return std::static_pointer_cast<void>(token);
    };
    options.allocation.allocated = [&](const std::shared_ptr<void>& token, void* native, size_t bytes) {
        check(native && bytes <= std::static_pointer_cast<AccountingToken>(token)->upper,
              "allocation hook capacity/handle mismatch");
        ++commits;
    };
    Backend backend(options);
    auto library = backend.compile_library(source, "small native Metal contract");
    auto pipeline = backend.prepare_pipeline(library, "affine");
    auto same_pipeline = backend.prepare_pipeline(library, "affine");
    check(same_pipeline.thread_execution_width() == pipeline.thread_execution_width(), "prepared pipeline cache mismatch");
    auto input = backend.allocate(32 * sizeof(float), "input");
    auto output = backend.allocate(32 * sizeof(float), "output");
    auto* data = static_cast<float*>(input.contents());
    for (size_t i = 0; i < 32; ++i) data[i] = float(i);
    rejects([&] { (void)input.view(1, 4, ScalarType::FP32); }, "unaligned view accepted");
    rejects([&] { (void)input.view(0, 129); }, "out of bounds view accepted");
    rejects([&] { (void)input.view(SIZE_MAX, 4); }, "overflow view accepted");
    auto event = backend.make_event();
    CommandGraph graph;
    Params params{32, 3, 5};
    graph.dispatch(pipeline, {input.view(0,128,ScalarType::FP32), output.view(0,128,ScalarType::FP32)},
                   params, {1,1,1}, {32,1,1});
    params.factor = params.bias = 0; // graph owns parameter bytes
    graph.signal(event, 1);
    graph.wait(event, 1);
    auto ticket = backend.submit(std::move(graph));
    auto completion = ticket.wait(3s);
    check(completion.ok && completion.physically_complete && completion.submitted_buffers == 2 &&
          completion.completed_buffers == 2, "signal split completion contract failed");
    for (size_t i = 0; i < 32; ++i)
        check(static_cast<float*>(output.contents())[i] == float(i)*3+5, "parameter copy or Metal computation failed");

    // A discarded ticket must not drop external allocator ownership while a
    // command waits. Release the external gate promptly: this is not a hang test.
    auto owner = std::make_shared<ExternalStorage>(input);
    std::weak_ptr<ExternalStorage> weak = owner;
    auto imported = backend.import_buffer(input.native_handle(), owner);
    check(reservations == 2 && commits == 2, "import double-counted allocation");
    auto gate = backend.make_event(EventSource::External);
    auto ended = backend.make_event();
    CommandGraph held;
    held.wait(gate, 1);
    held.dispatch(pipeline, {imported.view(0,128,ScalarType::FP32), output.view(0,128,ScalarType::FP32)},
                  Params{32,2,1}, {1,1,1}, {32,1,1});
    held.signal(ended, 1);
    { auto abandoned = backend.submit(std::move(held)); }
    imported = {}; owner.reset(); input = {};
    check(!weak.expired(), "ticket destruction released in-flight external owner");
    gate.signal_external(1);
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while ((!weak.expired() || ended.value() != 1 || releases->load() < 1) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    check(weak.expired() && ended.value() == 1, "physical completion did not retire graph owners");
    output = {};
    check(releases->load() == 2, "accounting token lifetime differs from backing lifetime");
    std::cout << "device=" << backend.device_name() << " graph_ownership=passed\n";
}

void test_preflight_and_timeout() {
    BackendOptions options; options.maximum_command_buffers = 1;
    Backend backend(options);
    auto event = backend.make_event();
    CommandGraph bad_order; bad_order.wait(event,1); bad_order.signal(event,1);
    rejects([&] { (void)backend.submit(std::move(bad_order)); }, "future local producer accepted");
    CommandGraph too_many; too_many.signal(event,1); too_many.wait(event,1);
    rejects([&] { (void)backend.submit(std::move(too_many)); }, "CB capacity overflow accepted");
    check(backend.healthy() && event.value() == 0, "preflight failure submitted work");
    CommandGraph valid; valid.signal(event,1);
    check(backend.submit(std::move(valid)).wait(3s).ok, "preflight mutated event reservations");

    auto gate = backend.make_event(EventSource::External);
    CommandGraph wait_graph; wait_graph.wait(gate,1);
    auto pending = backend.submit(std::move(wait_graph));
    auto timeout = pending.wait(0ms);
    check(timeout.timed_out && !timeout.ok && !timeout.physically_complete && !backend.healthy(),
          "timeout falsely certified completion");
    gate.signal_external(1);
    auto completed = pending.wait(3s);
    check(completed.physically_complete && !completed.ok && completed.timed_out,
          "late completion resurrected timed-out result");
    std::cout << "preflight_timeout=passed\n";
}
}

int main() {
    try {
        test_graph_and_ownership();
        test_preflight_and_timeout();
        std::cout << "native Metal graph backend contract passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
