#include "../../native/backends/ane_qkv_scheduler.hpp"
#include <cassert>
#include <limits>

int main() {
    using tc::ane::QkvScheduler;
    using Mode = QkvScheduler::Mode;
    QkvScheduler slow;
    for (int visit = 1; visit <= 4; ++visit) {
        auto p = slow.plan(0, 4226);
        assert(p.mode == (visit <= 2 ? Mode::Hybrid : Mode::GpuProbe));
        slow.observe(0, 4226, p, visit == 1 || visit == 3 ? 100. :
                     p.hybrid() ? .99 : 1.);
    }
    assert(slow.plan(0, 4226).mode == Mode::Gpu);
    assert(slow.plan(1, 4226).mode == Mode::Hybrid);
    assert(slow.plan(0, 4096).mode == Mode::Hybrid);
    for (int visit = 6; visit <= 34; ++visit)
        assert(slow.plan(0, 4226).mode == Mode::Gpu);
    auto probe = slow.plan(0, 4226);
    assert(probe.mode == Mode::GpuProbe);
    slow.observe(0, 4226, probe, 1.);
    auto retry = slow.plan(0, 4226);
    assert(retry.mode == Mode::Hybrid);
    slow.observe(0, 4226, retry, .7); // EMA .99 -> .9175, reopens at 8%
    assert(slow.plan(0, 4226).mode == Mode::HybridUntimed);

    QkvScheduler marginal;
    for (int visit = 1; visit <= 4; ++visit) {
        auto p = marginal.plan(0, 4226);
        marginal.observe(0, 4226, p, visit % 2 ? 100. :
                         p.hybrid() ? .949 : 1.);
    }
    assert(marginal.plan(0, 4226).mode == Mode::HybridUntimed);
    QkvScheduler invalid;
    for (int visit = 1; visit <= 4; ++visit) {
        auto p = invalid.plan(0, 4226);
        invalid.observe(0, 4226, p, visit == 1 || visit == 3 ?
            std::numeric_limits<double>::quiet_NaN() : p.hybrid() ? 2. : 1.);
    }
    // Invalid samples do not count as route warmup or create a decision.
    assert(invalid.plan(0, 4226).mode == Mode::Hybrid);
}
