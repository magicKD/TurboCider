// CPU-only test of the actual private Buffer::Impl, not a mirrored surrogate.
// Do not separately compile backend.mm into this target. NSObject stands in for
// a retained opaque buffer; no MTLDevice, command queue or GPU work is created.
#include "../../native/backends/metal_graph/backend.mm"
#include <array>
#include <iostream>

struct LifetimeLog {
    std::array<unsigned, 8> events{};
    size_t count = 0;
    void record(unsigned event) noexcept { if (count < events.size()) events[count++] = event; }
};

@interface TCNativeBufferLifetimeProbe : NSObject {
    LifetimeLog *_log;
}
- (instancetype)initWithLog:(LifetimeLog *)log;
@end
@implementation TCNativeBufferLifetimeProbe
- (instancetype)initWithLog:(LifetimeLog *)log {
    self = [super init];
    if (self) _log = log;
    return self;
}
- (void)dealloc { _log->record(1); }
@end

namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
std::shared_ptr<void> claim(LifetimeLog &log) {
    return {&log, [&log](void *) noexcept { log.record(3); }};
}
struct ExternalAllocation {
    id<MTLBuffer> native;
    LifetimeLog *log = nullptr;
    ~ExternalAllocation() noexcept { native = nil; log->record(2); }
};
}

int main() {
    @autoreleasepool {
        try {
            {
                LifetimeLog log;
                auto buffer = std::make_shared<tc::metal_graph::Buffer::Impl>();
                buffer->native = (id<MTLBuffer>)[[TCNativeBufferLifetimeProbe alloc] initWithLog:&log];
                buffer->accounting_lease = claim(log);
                auto alias = buffer;
                buffer.reset();
                check(log.count == 0, "Buffer alias released backing or budget early");
                alias.reset();
                check(log.count == 2 && log.events[0] == 1 && log.events[1] == 3,
                      "native backing must release before accounting claim");
            }
            {
                LifetimeLog log;
                auto external = std::make_shared<ExternalAllocation>();
                external->native = (id<MTLBuffer>)[[TCNativeBufferLifetimeProbe alloc] initWithLog:&log];
                external->log = &log;
                auto buffer = std::make_shared<tc::metal_graph::Buffer::Impl>();
                buffer->native = external->native;
                buffer->external_owner = external;
                buffer->accounting_lease = claim(log);
                external.reset();
                buffer.reset();
                check(log.count == 3 && log.events[0] == 1 && log.events[1] == 2 && log.events[2] == 3,
                      "imported allocator owner must release before accounting claim");
            }
            {
                LifetimeLog log;
                auto shared_claim = claim(log);
                auto buffer = std::make_shared<tc::metal_graph::Buffer::Impl>();
                buffer->native = (id<MTLBuffer>)[[TCNativeBufferLifetimeProbe alloc] initWithLog:&log];
                buffer->accounting_lease = shared_claim;
                buffer.reset();
                check(log.count == 1 && log.events[0] == 1, "shared lease released another owner's accounting");
                shared_claim.reset();
                check(log.count == 2 && log.events[1] == 3, "shared claim lifetime changed");
            }
            std::cout << "PASS native Buffer CPU lifetime: native/external backing before claim, aliases and shared claims retained\n";
        } catch (const std::exception &error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
