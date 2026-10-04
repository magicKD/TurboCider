#include "../../native/backends/ane_scheduler.hpp"
#include <cassert>

int main() {
    using tc::ane::RowScheduler;
    using Mode = RowScheduler::Mode;
    using tc::ane::PartitionAxis;
    // Channels cover ALL rows, including a short/tail bucket. The common
    // whole-block on/off controller must not apply the row chunk balancer.
    RowScheduler channels(32,-1,PartitionAxis::IntermediateChannels);
    for(int visit=1;visit<=4;++visit) {
        auto p=channels.plan(0,17);
        assert(p.chunks==(visit<=2?1:0));
        channels.observe(0,17,p.chunks,p.chunks?.5:1.,100.,.0001);
    }
    assert(channels.plan(0,17).chunks==1);
    assert(channels.plan(1,17).chunks==1);
    assert(channels.plan(0,65).chunks==1);
    assert(RowScheduler(32,1,PartitionAxis::IntermediateChannels).plan(0,1).chunks==1);
    bool rejected=false;
    try { RowScheduler invalid(32,2,PartitionAxis::IntermediateChannels); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    // Decisions distinguish measured GPU probes from genuinely unsplit GPU
    // work. Shape and layer isolation still apply, and disabled layers retry.
    RowScheduler plans(32);
    const auto preview=plans.peek_plan(0,97);
    assert(preview.chunks==1&&plans.peek_plan(0,97).chunks==1);
    assert(plans.plan(0,97).chunks==1); // only this call advances the visit
    assert(plans.plan(0,97).chunks==1);
    assert(plans.plan(0,97).mode==Mode::GpuProbe);
    plans=RowScheduler(32);
    assert(plans.plan(0, 32).mode == Mode::Gpu);
    for (int visit = 1; visit <= 4; ++visit) {
        const auto plan = plans.plan(0, 97);
        assert(plan.split() == (visit <= 2));
        assert(plan.measured());
        assert(plan.mode == (visit <= 2 ? Mode::Hybrid : Mode::GpuProbe));
        plans.observe(0, 97, plan.chunks, visit <= 2 ? .010 : .005, .004, .009);
    }
    for (int visit = 5; visit <= 34; ++visit) {
        const auto plan = plans.plan(0, 97);
        assert(plan.mode == Mode::Gpu && !plan.split() && plan.chunks == 0);
    }
    assert(plans.plan(0, 97).mode == Mode::GpuProbe);
    assert(plans.plan(0, 97).mode == Mode::Hybrid);
    assert(plans.plan(1, 97).mode == Mode::Hybrid);
    assert(plans.plan(0, 193).mode == Mode::Hybrid);
    const auto ablation = RowScheduler(32, 0).plan(0, 97);
    assert(ablation.mode == Mode::SplitProbe && ablation.split() && ablation.measured());
    RowScheduler fixed(32, 20);
    assert(fixed.select(0, 32) == 0);
    assert(fixed.select(0, 33) == 1);
    assert(fixed.select(0, 97) == 3);
    RowScheduler disabled(32, 0);
    assert(disabled.select(0, 97) == 0);
    RowScheduler adaptive(32);
    for (int visit = 1; visit <= 4; ++visit) {
        int chunks = adaptive.select(0, 97);
        assert(chunks == (visit <= 2 ? 1 : 0));
        adaptive.observe(0, 97, chunks, chunks ? .010 : .005, .004, .009);
    }
    // Unprofitable hybrid disables; a new layer/shape does not inherit this.
    assert(adaptive.select(0, 97) == 0);
    assert(adaptive.select(1, 97) == 1);
    assert(adaptive.select(0, 193) == 1);
    for (int visit = 6; visit <= 35; ++visit) assert(adaptive.select(0, 97) == 0);
    assert(adaptive.select(0, 97) > 0); // periodic reprobe, not permanent disable
    RowScheduler balance(32);
    assert(balance.select(0, 1024) == 1);
    balance.observe(0, 1024, 1, 100., 100., .0001); // compile outlier
    assert(balance.select(0, 1024) == 1); // must not tune from warmup
    balance.observe(0, 1024, 1, .020, .020, .0005);
    assert(balance.select(0, 1024) == 0);
    balance.observe(0, 1024, 0, 100.); // first GPU call is also excluded
    assert(balance.select(0, 1024) == 0);
    balance.observe(0, 1024, 0, .030);
    const int chosen = balance.select(0, 1024);
    assert(chosen == 18); // global minimum, then one hysteresis decision
    // A profitable layer does not probe again in every short request.
    for (int visit = 6; visit <= 34; ++visit) assert(balance.select(0, 1024) > 0);
    assert(balance.select(0, 1024) == 0); // periodic GPU baseline refresh
    // The initial small share loses to GPU, but the predicted balanced
    // share wins. GPU probes must not disable that unmeasured candidate.
    RowScheduler retune(32);
    for (int visit = 1; visit <= 2; ++visit) {
        assert(retune.select(0, 1024) == 1);
        retune.observe(0, 1024, 1, .020, .018, .001);
    }
    for (int visit = 3; visit <= 4; ++visit) {
        assert(retune.select(0, 1024) == 0);
        retune.observe(0, 1024, 0, .015);
    }
    const int tuned = retune.select(0, 1024);
    assert(tuned == 12);
    retune.observe(0, 1024, tuned, 10., 10., .012); // new shape compilation
    assert(retune.select(0, 1024) == tuned);
    retune.observe(0, 1024, tuned, .013, .0116, .012);
    const auto steady = retune.plan(0, 1024);
    assert(steady.chunks == tuned && steady.mode == Mode::HybridUntimed);
    assert(steady.split() && !steady.measured());
    for (int visit = 8; visit <= 33; ++visit)
        assert(retune.plan(0, 1024).mode == Mode::HybridUntimed);
    assert(retune.plan(0, 1024).mode == Mode::Hybrid);
    assert(retune.plan(0, 1024).mode == Mode::GpuProbe);
    assert(retune.plan(0, 1024).mode == Mode::Hybrid);
    // A later slower whole-block sample disables hybrid; never compare a
    // short in-FFN window from steady untimed blocks with full GPU probes.
    retune.observe(0, 1024, tuned, .1);
    assert(retune.plan(0, 1024).mode == Mode::Gpu);

    // On/off margins apply to full-block timing, not isolated FFNs. Preserve
    // the measured 2%/5% policy, independently of partition-change hysteresis.
    auto calibrate = [](RowScheduler &scheduler, double hybrid) {
        for (int visit = 1; visit <= 4; ++visit) {
            const auto plan = scheduler.plan(0, 97);
            scheduler.observe(0, 97, plan.chunks, plan.chunks ? hybrid : 1.0);
        }
    };
    for (double ratio : {.99, .985, .98}) {
        RowScheduler marginal(32);
        calibrate(marginal, ratio);
        assert(marginal.plan(0, 97).mode == Mode::Gpu);
        assert(marginal.plan(1, 97).mode == Mode::Hybrid);
        assert(marginal.plan(0, 193).mode == Mode::Hybrid);
    }
    RowScheduler profitable(32);
    calibrate(profitable, .979);
    assert(profitable.plan(0, 97).mode == Mode::Hybrid);

    RowScheduler hysteresis(32);
    calibrate(hysteresis, .99);
    auto reprobe = [&](int first_visit, double hybrid) {
        for (int visit = first_visit; visit <= first_visit + 31; ++visit) {
            const auto plan = hysteresis.plan(0, 97);
            if (visit == first_visit + 30) {
                assert(plan.mode == Mode::GpuProbe);
                hysteresis.observe(0, 97, 0, 1.0);
            } else if (visit == first_visit + 31) {
                assert(plan.mode == Mode::Hybrid);
                hysteresis.observe(0, 97, plan.chunks, hybrid);
            } else {
                assert(plan.mode == Mode::Gpu);
            }
        }
    };
    // EMA .99 -> .96: >2% is not enough to reopen a disabled layer.
    reprobe(5, .87);
    // EMA .96 -> .94: now clears the stronger 5% reenable threshold.
    reprobe(37, .88);
    assert(hysteresis.plan(0, 97).mode == Mode::HybridUntimed);

    RowScheduler forced(32, 1);
    for (int visit = 0; visit < 40; ++visit) {
        const auto plan = forced.plan(0, 97);
        assert(plan.mode == Mode::Hybrid && plan.chunks == 1);
        forced.observe(0, 97, plan.chunks, 10.0, 1.0, 9.0);
    }

    // Keep the measured one-chunk seed: a quarter-share start did not improve
    // edit request latency. Long shapes must still leave a nonempty GPU head,
    // with sub-chunk inputs and explicit fixed partitions unchanged.
    for (int rows : {1, 287, 288, 289, 1024, 1152, 2303, 2304, 4096, 4226, 16384}) {
        RowScheduler seed(288);
        const auto plan = seed.plan(0, rows);
        const int expected = rows <= 288 ? 0 : 1;
        assert(plan.chunks == expected);
        assert(plan.chunks * 288 < rows);
        assert(plan.mode == (expected ? Mode::Hybrid : Mode::Gpu));
        assert(RowScheduler(288, 1).plan(0, rows).chunks == (rows > 288 ? 1 : 0));
    }
    RowScheduler long_seed(288);
    assert(long_seed.plan(0, 4226).chunks == 1);
    long_seed.observe(0, 4226, 1, .050, .040, .100); // excluded warmup
    assert(long_seed.plan(0, 4226).chunks == 1);
    long_seed.observe(0, 4226, 1, .050, .040, .012); // measured rates choose 3
    assert(long_seed.plan(0, 4226).mode == Mode::GpuProbe);
    long_seed.observe(0, 4226, 0, .020);
    assert(long_seed.plan(0, 4226).mode == Mode::GpuProbe);
    long_seed.observe(0, 4226, 0, .020);
    assert(long_seed.plan(0, 4226).chunks == 3);
    long_seed.observe(0, 4226, 3, .500, .005, .400); // new partition warmup
    assert(long_seed.plan(0, 4226).chunks == 3);
    long_seed.observe(0, 4226, 3, .500, .005, .400); // slow ANE: shrink to 1
    assert(long_seed.plan(0, 4226).chunks == 1);
    assert(long_seed.plan(1, 4226).chunks == 1); // no cross-layer timing leak
    assert(long_seed.plan(0, 1024).chunks == 1); // no prefill/decode leakage

    // Exclude the first execution of EACH route; GPU
    // compile outliers must not turn an unprofitable adapter into a win.
    for (double hybrid : {.5, 1.1}) {
        RowScheduler order(32);
        for (int visit = 1; visit <= 4; ++visit) {
            const auto plan = order.plan(0, 64);
            const bool probe = visit >= 3;
            assert(plan.mode == (probe ? Mode::GpuProbe : Mode::Hybrid));
            const bool warmup = visit == 1 || visit == 3;
            order.observe(0, 64, plan.chunks, warmup ? 100. : probe ? 1. : hybrid);
        }
        for (int visit = 5; visit <= 34; ++visit) {
            const auto plan = order.plan(0, 64);
            assert(plan.mode == (hybrid > 1 ? Mode::Gpu :
                visit > 6 && visit != 34 ? Mode::HybridUntimed : Mode::Hybrid));
        }
        auto plan = order.plan(0, 64);
        assert(plan.mode == Mode::GpuProbe);
        order.observe(0, 64, 0, 1.);
        plan = order.plan(0, 64);
        assert(plan.mode == Mode::Hybrid);
        order.observe(0, 64, plan.chunks, .1);
        assert(order.plan(0, 64).mode == Mode::HybridUntimed);
        assert(order.plan(1, 64).mode == Mode::Hybrid);
        assert(order.plan(0, 97).mode == Mode::Hybrid);
        assert(order.plan(0, 32).mode == Mode::Gpu);
        assert(RowScheduler(32, 0).plan(0, 64).mode == Mode::SplitProbe);
        assert(RowScheduler(32, 2).plan(0, 97).chunks == 2);
    }
}
