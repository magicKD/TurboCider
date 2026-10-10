#include "../../native/backends/private/ane_ticket_state.hpp"
#include <iostream>
#include <stdexcept>

using tc::ane::private_api::detail::TicketCompletionState;

namespace {
void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
bool may_publish(const TicketCompletionState &state) {
    const auto &s = state.status();
    return s.ok && s.physical_complete && !s.retired;
}
void check_retired(const TicketCompletionState &state, bool physical, bool timed_out,
                   const char *diagnostic) {
    const auto &s = state.status();
    check(state.terminal() && s.retired && !s.ok && !may_publish(state), "retired result became publishable");
    check(s.physical_complete == physical, "physical completion evidence changed incorrectly");
    check(s.timed_out == timed_out, "first timeout state changed");
    check(s.error == diagnostic, "first failure diagnostic changed");
}
} // namespace

int main() {
    try {
        TicketCompletionState pending;
        for (int i = 0; i != 4; ++i) {
            check(!pending.terminal() && !may_publish(pending), "status retired or completed a pending request");
            check(!pending.status().timed_out && !pending.status().retired &&
                  !pending.status().physical_complete && pending.status().error.empty(), "status has side effects");
        }

        TicketCompletionState success;
        check(!success.completed(true), "success requested failure release");
        check(success.terminal() && may_publish(success), "successful callback not publishable");
        check(!success.completed(false, "duplicate callback"), "duplicate callback changed first result");
        check(may_publish(success), "duplicate callback replaced first callback");

        for (bool callback_ok : {false, true}) {
            TicketCompletionState timeout;
            check(timeout.retire("deadline", true), "first timeout did not retire");
            check_retired(timeout, false, true, "deadline");
            check(!timeout.retire("second timeout", true), "second timeout repeated failure release");
            check(!timeout.completed(callback_ok, "late failure"), "late callback repeated failure release");
            check_retired(timeout, true, true, "deadline");

            TicketCompletionState rejected;
            check(rejected.retire("enqueue rejected"), "rejection did not retire");
            check_retired(rejected, false, false, "enqueue rejected");
            check(!rejected.retire("later deadline", true), "rejection became a timeout");
            check(!rejected.completed(callback_ok, "late failure"), "rejected callback repeated failure release");
            check_retired(rejected, true, false, "enqueue rejected");
        }

        // Some drivers invoke the callback before evaluateWithModel returns.
        // A subsequent NO/exception still invalidates the logical result.
        TicketCompletionState callback_before_rejection;
        callback_before_rejection.completed(true);
        check(callback_before_rejection.retire("enqueue returned NO"), "fast callback hid submit rejection");
        check_retired(callback_before_rejection, true, false, "enqueue returned NO");

        TicketCompletionState failed;
        check(failed.completed(false, "driver failed"), "failed callback did not request failure release");
        check_retired(failed, true, false, "driver failed");
        check(!failed.retire("later timeout", true), "failed callback failure overwritten");
        check(!failed.completed(true), "second callback resurrected failed result");
        check_retired(failed, true, false, "driver failed");

        TicketCompletionState unsubmitted;
        unsubmitted.discard_unsubmitted();
        check_retired(unsubmitted, true, false, "");

        // Existing three-field aggregate callers remain source-compatible;
        // their un-migrated physical state stays conservatively unknown.
        tc::ane::gpu::Completion legacy{true, false, ""};
        check(legacy.ok && !legacy.physical_complete && !legacy.retired, "legacy aggregate contract changed");
        std::cout << "PASS private ANE ticket CPU contract: status, sticky timeout/rejection, late callbacks, fast callback before reject, unsubmitted discard\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
