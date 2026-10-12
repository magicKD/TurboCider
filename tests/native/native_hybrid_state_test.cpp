#include "../../native/backends/metal_graph/native_hybrid_state.hpp"
#include <cassert>
#include <iostream>
using namespace tc::metal_graph;
int main() {
    HybridPhysicalEvidence e;
    e.handoff.ok = e.handoff.all_work_physically_complete = true;
    assert(hybrid_all_physically_complete(e));
    assert(!hybrid_publishable(e, false, true));
    e.gpu_attempted = true; // Unknown/throwing submission must retain owners.
    assert(!hybrid_all_physically_complete(e));
    e.gpu = Completion{true, true, false, 1, 1, 0., {}};
    e.consumer_attempted = true;
    assert(!hybrid_all_physically_complete(e));
    e.consumer = Completion{false, false, true, 1, 0, 0., "deadline"};
    assert(!hybrid_publishable(e, false, true));
    e.consumer->physically_complete = true; // Late callback frees, never publishes.
    assert(hybrid_all_physically_complete(e));
    assert(!hybrid_publishable(e, true, true));
    e.consumer = Completion{true, true, false, 1, 1, 0., {}};
    assert(hybrid_publishable(e, false, true));
    assert(!hybrid_publishable(e, false, false)); // Arithmetic flags are independent.
    e.handoff.retired = true;
    assert(!hybrid_publishable(e, false, true));
    e.handoff.retired = false;
    e.handoff.all_work_physically_complete = false; // ANE submit without Ticket.
    assert(!hybrid_all_physically_complete(e));
    assert(!hybrid_publishable(e, false, true));
    std::cout << "native hybrid aggregate physical/publication contract passed\n";
}
