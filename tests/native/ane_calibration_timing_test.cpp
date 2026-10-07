#include "../../native/backends/ane_calibration_timing.hpp"
#include <iostream>

using namespace tc::ane;
namespace {
void check(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
struct FakeClock {
    using time_point = std::chrono::time_point<FakeClock, std::chrono::duration<double>>;
    static inline double seconds = 0;
    static time_point now() noexcept { return time_point{std::chrono::duration<double>{seconds}}; }
};
void test_samples(int warmups, int repeats) {
    FakeClock::seconds = 0;
    int resets = 0, submissions = 0, finishes = 0, pending = 0;
    std::vector<int> order;
    const auto samples = detail::measure_full_gpu_calibration<FakeClock>([&] {
        check(!pending && submissions == finishes, "reset before prior GPU arm drained");
        ++resets;
        FakeClock::seconds += 10; // preparation never enters a timed arm
    }, [&](int count) {
        check(!pending, "overlapping GPU calibration submissions");
        ++submissions;
        pending = count;
        order.push_back(count);
    }, [&] {
        check(pending != 0, "finish without a GPU submission");
        ++finishes;
        // One setup cost plus per-layer cost. Deliberately contaminate one
        // hot span per arm; medians must retain/reject neither raw sample.
        FakeClock::seconds += .009 + .008 * pending;
        if (finishes == 2 * warmups + 1 || finishes == 2 * warmups + 2)
            FakeClock::seconds += 1;
        pending = 0;
    }, warmups, repeats);
    check(resets == 2 * (warmups + repeats) && resets == submissions && submissions == finishes,
          "calibration warmup/repeat/drain counts differ");
    check(samples.seconds[0].size() == size_t(repeats) && samples.seconds[1].size() == size_t(repeats),
          "warmups leaked into samples or hot samples were discarded");
    check(std::abs(samples.layer_seconds - .008) < 1e-10, "fixed/reset cost cancellation or median failed");
    for (size_t index = 0; index < order.size(); ++index)
        check(order[index] == ((index / 2 + index % 2) % 2 ? 4 : 1), "one/four cyclic order changed");
    for (const auto &arm : samples.seconds)
        check(std::count_if(arm.begin(), arm.end(), [](double s) { return s > 1; }) == 1,
              "raw outlier was omitted from evidence");
}
void test_failures() {
    int calls = 0;
    const std::function<void()> reset = [&] { ++calls; };
    const std::function<void(int)> submit = [&](int) { ++calls; };
    const std::function<void()> finish = [&] { ++calls; };
    for (int warmups : {-1, 0, 1, 2, 8, 9}) for (int repeats : {-1, 0, 2, 3, 4, 7, 31, 32, 33}) {
        if ((warmups == 1 || warmups == 2 || warmups == 8) &&
            (repeats == 3 || repeats == 7 || repeats == 31)) continue;
        bool rejected = false;
        try { (void)measure_full_gpu_calibration(reset, submit, finish, warmups, repeats); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "invalid sampling bounds accepted");
    }
    for (int missing = 0; missing < 3; ++missing) {
        bool rejected = false;
        try {
            (void)measure_full_gpu_calibration(missing == 0 ? std::function<void()>{} : reset,
                missing == 1 ? std::function<void(int)>{} : submit,
                missing == 2 ? std::function<void()>{} : finish);
        } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "incomplete callbacks accepted");
    }
    check(calls == 0, "invalid arguments invoked a GPU callback");
    for (int failure = 0; failure < 4; ++failure) {
        int resets = 0, submissions = 0, finishes = 0;
        const std::string expected = failure == 0 ? "reset" : failure == 2 ? "finish" : "submit";
        bool caught = false;
        try {
            (void)measure_full_gpu_calibration([&] {
                ++resets;
                if (failure == 0) throw std::runtime_error("reset");
            }, [&](int) {
                ++submissions;
                if (failure == 1 || failure == 3) throw std::runtime_error("submit");
            }, [&] {
                ++finishes;
                if (failure == 2 || failure == 3) throw std::runtime_error("finish");
            });
        } catch (const std::runtime_error &error) { caught = error.what() == expected; }
        check(caught && resets == 1 && submissions == (failure ? 1 : 0) && finishes == submissions,
              "partial submission was not drained once, or first error was lost");
    }
    for (double span : {0., -.01, double(NAN), std::numeric_limits<double>::infinity()}) {
        FakeClock::seconds = 0;
        bool rejected = false;
        int finishes = 0;
        try {
            (void)detail::measure_full_gpu_calibration<FakeClock>([] {}, [](int) {}, [&] {
                ++finishes;
                FakeClock::seconds += span;
            }, 2, 7);
        } catch (const CapabilityError &) { rejected = true; }
        check(rejected && finishes == 1, "invalid clock span accepted or producer not drained");
    }
    FakeClock::seconds = 0;
    bool rejected = false;
    try {
        (void)detail::measure_full_gpu_calibration<FakeClock>([] {}, [](int) {}, [] {
            FakeClock::seconds += .015625; // exact binary fraction: identical one/four spans
        }, 2, 7);
    } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "nonpositive one/four difference was clamped into a result");
}
} // namespace
int main() {
    try {
        test_samples(2, 7);
        test_samples(1, 3);
        test_samples(8, 31);
        test_failures();
        std::cout << "PASS shared GPU calibration timing: cyclic serial order, exact hot samples, warmup exclusion, "
                     "median/fixed-cost cancellation, argument/span rejection and first-error drain\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
