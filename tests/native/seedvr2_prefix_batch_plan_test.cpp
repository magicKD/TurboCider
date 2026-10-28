#include "../../native/models/seedvr2/prefix_batch_plan.hpp"
#include "../../native/models/seedvr2/prefix_buffer_lease.hpp"
#include "../../native/models/seedvr2/fixed_feature_shape.hpp"
#include <cassert>
#include <iostream>
#include <limits>

using namespace tc::seedvr2;
template <class Action> void rejected(Action action) {
    bool caught = false;
    try { action(); } catch (const std::exception &) { caught = true; }
    assert(caught);
}
int main() {
    for (int total : {1,512,1088,4096,7168,28672,31744,126976,std::numeric_limits<int>::max()}) {
        for (int rows : {128,256,512,1024}) {
            const int maximum = rows == 1024 ? 51 : 64;
            const auto batches = prefix_batch_ranges(total,rows,maximum);
            int64_t next = 0;
            for (auto [first,last] : batches) {
                assert(first == next && last > first && int64_t(last)-first <= int64_t(rows)*maximum);
                assert((int64_t(last)-first+rows-1)/rows <= maximum);
                next = last;
            }
            assert(next == total);
            if (total == 28672 && rows == 512) assert(batches.size() == 1);
            if (total == 126976 && rows == 512) assert(batches.size() == 4);
        }
    }
    rejected([] { prefix_batch_ranges(0,512,64); });
    rejected([] { prefix_batch_ranges(1,0,64); });
    rejected([] { prefix_batch_ranges(1,127,64); });
    rejected([] { prefix_batch_ranges(1,2048,64); });
    rejected([] { prefix_batch_ranges(1,512,0); });
    rejected([] { prefix_batch_ranges(1,512,65); });
    PrefixBufferLease lease;
    rejected([&] { lease.mark_written(0); });
    lease.begin(2);
    rejected([&] { lease.begin(1); });
    rejected([&] { lease.release(); });
    lease.mark_written(0);
    rejected([&] { lease.mark_written(0); });
    lease.mark_written(1);
    lease.release();
    lease.begin(1); lease.mark_written(0); lease.release();
    const std::vector<size_t> shape{1,2560,1,512};
    const std::vector<std::pair<size_t,size_t>> ranges{{1,1},{2560,1},{1,1},{512,1}};
    assert(exact_fixed_feature_shape(2,shape,{shape},ranges));
    assert(exact_fixed_feature_shape(3,shape,{},ranges));
    assert(!exact_fixed_feature_shape(1,shape,{shape},ranges));
    assert(!exact_fixed_feature_shape(2,shape,{shape,{1,2560,1,1024}},ranges));
    auto broad = ranges; broad.back().second = 2;
    assert(!exact_fixed_feature_shape(3,shape,{},broad));
    std::cout << "PASS exact coverage/bounds/tails/int-limit/lease/static-shape guards\n";
}
