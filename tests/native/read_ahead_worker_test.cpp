#include "../../native/runtime/streaming/read_ahead_worker.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace tc;
using namespace tc::streaming;

template <class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    assert(rejected);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path path = argv[1];
    const size_t capacity = 16384;
    std::vector<std::byte> source(capacity * 129 + 17);
    for (size_t i = 0; i < source.size(); ++i) source[i] = std::byte((i * 13 + i / 17) % 256);
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(source.data()), std::streamsize(source.size()));
    }
    OwnedSourceFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC)); assert(fd);
    assert(::lseek(fd.get(), 123, SEEK_SET) == 123);
    std::vector<std::byte> buffers[2]{std::vector<std::byte>(capacity), std::vector<std::byte>(capacity)};
    std::vector<std::byte> observed(source.size());
    {
        ReadAheadWorker reader(fd.get(), source.size(), capacity);
        assert(reader.idle()); rejects([&] { reader.take(); });
        rejects([&] { reader.submit({}, 0); });
        rejects([&] { reader.submit({observed.data(), capacity + 1}, 0); });
        rejects([&] { reader.submit({buffers[0].data(), capacity}, source.size()); });
        reader.submit(buffers[0], 0);
        rejects([&] { reader.submit(buffers[1], 0); });
        unsigned slot = 0;
        uint64_t bytes = 0, jobs = 0;
        for (size_t offset = 0; offset < source.size();) {
            const size_t count = std::min(capacity, source.size() - offset);
            const auto completed = reader.take();
            assert(completed.bytes == count && completed.seconds >= 0); bytes += completed.bytes; ++jobs;
            const size_t next = offset + count;
            if (next < source.size()) reader.submit({buffers[slot ^ 1].data(), std::min(capacity, source.size() - next)}, next);
            // Consume current while the next job can write ONLY the other slot.
            std::memcpy(observed.data() + offset, buffers[slot].data(), count);
            offset = next; slot ^= 1;
        }
        assert(bytes == source.size() && jobs == 130 && observed == source && reader.idle());
        std::thread intruder([&] { rejects([&] { reader.idle(); }); }); intruder.join();
        reader.stop_and_drain(); reader.stop_and_drain(); rejects([&] { reader.submit(buffers[0], 0); });
    }
    assert(::lseek(fd.get(), 0, SEEK_CUR) == 123); // pread never changes caller offset.
    {
        // Disjoint final-storage ranges model BF16 same-width direct read.
        std::fill(observed.begin(), observed.end(), std::byte{});
        ReadAheadWorker reader(fd.get(), source.size(), capacity);
        reader.submit({observed.data(), capacity}, 0);
        for (size_t offset = 0; offset < source.size();) {
            const auto count = std::min(capacity, source.size() - offset);
            assert(reader.take().bytes == count);
            const auto next = offset + count;
            if (next < source.size()) reader.submit({observed.data() + next, std::min(capacity, source.size() - next)}, next);
            assert(std::memcmp(observed.data() + offset, source.data() + offset, count) == 0);
            offset = next;
        }
        assert(reader.idle());
    }
    std::atomic<bool> cancel{true};
    rejects([&] { ReadAheadWorker reader(fd.get(), source.size(), capacity, &cancel); });
    cancel.store(false);
    {
        ReadAheadWorker reader(fd.get(), source.size(), capacity, &cancel);
        reader.submit(buffers[0], 0); cancel.store(true);
        rejects([&] { reader.take(); }); reader.stop_and_drain();
    }
    cancel.store(false);
    // Destruction with a queued/inflight write drains before its target dies.
    for (int i = 0; i < 128; ++i) {
        ReadAheadWorker reader(fd.get(), source.size(), capacity);
        reader.submit(buffers[0], 0);
    }
    {
        OwnedSourceFd writer(::open(path.c_str(), O_WRONLY | O_CLOEXEC)); assert(writer);
        ReadAheadWorker reader(fd.get(), source.size(), capacity);
        assert(::ftruncate(writer.get(), 0) == 0);
        reader.submit(buffers[0], 0); rejects([&] { reader.take(); });
        rejects([&] { reader.submit(buffers[0], 0); });
    }
    rejects([&] { ReadAheadWorker reader(-1, source.size(), capacity); });
    rejects([&] { ReadAheadWorker reader(fd.get(), 0, capacity); });
    std::cout << "PASS bounded pread worker: two-slot/full-output, disjoint aliases, offset/range/owner, error poison, cancel and queued-destruction drain\n";
}
