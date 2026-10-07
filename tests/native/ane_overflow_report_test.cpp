#include "../../native/core/ane_overflow_report.hpp"
#include <cassert>
#include <type_traits>

int main() {
    tc::ane::OverflowReport report;
    static_assert(noexcept(report.record({})));
    static_assert(std::is_trivially_copyable_v<tc::ane::OverflowReport>);
    report.record({0,1056,0,1,0,1,1,true});
    assert(!report.size && !report.dropped);
    report.record({2,1056,7,3,2,1,16,true});
    assert(report.size==1 && report.events[0].layer==2 && report.events[0].rows==1056 &&
        report.events[0].runtime_call_begin==7 && report.events[0].runtime_call_count==3 &&
        report.events[0].retries==2 && report.events[0].headroom_before==1 && report.events[0].headroom_after==16);
    report.record({3,4224,10,1,1,16,64,false});
    assert(!report.events[1].completed && report.events[1].headroom_before==16);
    for(size_t i=2;i<report.capacity+5;++i)report.record({int(i),1056,i,2,1,1,4,true});
    assert(report.size==report.capacity && report.dropped==5 && report.events[0].layer==2);
    const auto copy=report;report.record({1,1056,99,2,1,1,4,true});
    assert(copy.dropped==5 && report.dropped==6 && copy.events[0].retries==2);
}
