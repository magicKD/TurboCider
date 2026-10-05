#include "../../native/runtime/async_lora_preflight.hpp"
#include "../../native/platform/apple/sha256_fd.hpp"
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <thread>

namespace fs = std::filesystem;
using namespace std::chrono_literals;
static constexpr const char *abc_sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
static std::atomic<bool> no_cancel{false};
static int cases = 0;

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> static void rejects(F action, const std::string &message) {
    try { action(); }
    catch (const std::exception &error) {
        check(std::string(error.what()).find(message) != std::string::npos, "unexpected rejection");
        return;
    }
    throw std::runtime_error("expected rejection");
}
static void write(const fs::path &path, const std::string &bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << bytes;
    check(bool(file), "fixture write failed");
}
static auto hash = [](int fd, const std::atomic<bool> &stop) { return tc::detail::sha256_fd(fd, stop); };
static void wait(const std::atomic<bool> &ready) {
    const auto deadline = tc::Clock::now() + 2s;
    while (!ready.load()) {
        check(tc::Clock::now() < deadline, "fixture synchronization timed out");
        std::this_thread::sleep_for(1ms);
    }
}
static void preserve_mtime(const fs::path &path, const struct stat &original) {
    timespec times[2] = {original.st_atimespec, original.st_mtimespec};
    check(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0, "cannot restore fixture mtime");
}

static void run_cases(const fs::path &folder, const std::string &chunk_digest) {
    const auto file = folder / "source.safetensors";
    write(file, "abc");
    {
        tc::LoraFileFingerprint source(file);
        check(lseek(source.descriptor(), 2, SEEK_SET) == 2, "cannot set offset");
        check(hash(source.descriptor(), no_cancel) == abc_sha, "known digest mismatch");
        check(lseek(source.descriptor(), 0, SEEK_CUR) == 2, "pread changed the caller offset");
        source.revalidate(); ++cases;
    }
    {
        tc::LoraFileFingerprint source(folder / "chunks.bin");
        check(hash(source.descriptor(), no_cancel) == chunk_digest, "multi-chunk digest mismatch");
        ++cases;
    }
    {
        const auto empty = folder / "empty"; write(empty, "");
        tc::LoraFileFingerprint source(empty);
        check(hash(source.descriptor(), no_cancel) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty digest mismatch");
        ++cases;
    }
    {
        std::atomic<bool> cancelled{true};
        tc::LoraFileFingerprint source(file);
        rejects([&] { hash(source.descriptor(), cancelled); }, "generation cancelled");
        rejects([&] { hash(-1, no_cancel); }, "invalid SHA-256");
        ++cases;
    }
    {
        const auto fifo = folder / "fifo";
        check(mkfifo(fifo.c_str(), 0600) == 0, "cannot create fifo");
        rejects([&] { tc::LoraFileFingerprint source(fifo); }, "not a regular file");
        rejects([&] { tc::LoraFileFingerprint source(folder / "missing"); }, "cannot resolve");
        int descriptors[2]; check(pipe(descriptors) == 0, "cannot create pipe");
        rejects([&] { hash(descriptors[0], no_cancel); }, "not a regular file");
        close(descriptors[0]); close(descriptors[1]); ++cases;
    }
    {
        std::atomic<bool> started{false}, release{false};
        std::thread::id background;
        tc::AsyncLoraPreflight task(file, [&](int fd, const std::atomic<bool> &stop) {
            background = std::this_thread::get_id(); started = true;
            while (!release.load() && !stop.load()) std::this_thread::sleep_for(1ms);
            return hash(fd, stop);
        });
        wait(started); check(background != std::this_thread::get_id(), "hash did not run on worker");
        release = true;
        auto verified = task.take(no_cancel);
        check(verified.value->digest == abc_sha && verified.value->bytes() == 3, "async verification failed");
        check(verified.value->canonical() == fs::canonical(file), "canonical binding missing");
        check(verified.seconds >= 0 && verified.wait_seconds >= 0 && verified.before_join_seconds >= 0 &&
              verified.before_join_seconds <= verified.seconds, "invalid task timings");
        rejects([&] { task.take(no_cancel); }, "already consumed"); ++cases;
    }
    {
        const auto alias = folder / "alias";
        const auto other = folder / "other"; write(other, "def"); fs::create_symlink(file, alias);
        tc::AsyncLoraPreflight task(alias, hash); auto verified = task.take(no_cancel);
        check(verified.value->canonical() == fs::canonical(file), "alias did not bind physical source");
        fs::remove(alias); fs::create_symlink(other, alias);
        rejects([&] { verified.value->revalidate(); }, "canonical path changed"); ++cases;
    }
    {
        std::atomic<bool> hashed{false}, release{false};
        tc::AsyncLoraPreflight task(file, [&](int fd, const std::atomic<bool> &stop) {
            auto digest = hash(fd, stop); hashed = true;
            while (!release.load() && !stop.load()) std::this_thread::sleep_for(1ms);
            return digest;
        });
        wait(hashed);
        struct stat original{}; check(stat(file.c_str(), &original) == 0, "cannot stat source");
        const auto replacement = folder / "replacement"; write(replacement, "def");
        preserve_mtime(replacement, original); fs::rename(replacement, file);
        struct stat changed{}; check(stat(file.c_str(), &changed) == 0, "cannot stat replacement");
        check(original.st_size == changed.st_size && original.st_mtimespec.tv_sec == changed.st_mtimespec.tv_sec &&
              original.st_mtimespec.tv_nsec == changed.st_mtimespec.tv_nsec && original.st_ino != changed.st_ino,
              "same-size/mtime replacement fixture invalid");
        release = true;
        rejects([&] { task.take(no_cancel); }, "file generation changed"); ++cases;
    }
    {
        write(file, "abc"); tc::AsyncLoraPreflight task(file, hash); auto verified = task.take(no_cancel);
        struct stat original{}; check(stat(file.c_str(), &original) == 0, "cannot stat source");
        std::this_thread::sleep_for(2ms); write(file, "xyz"); preserve_mtime(file, original);
        struct stat changed{}; check(stat(file.c_str(), &changed) == 0, "cannot stat changed source");
        check(original.st_ino == changed.st_ino && original.st_size == changed.st_size &&
              original.st_mtimespec.tv_sec == changed.st_mtimespec.tv_sec &&
              original.st_mtimespec.tv_nsec == changed.st_mtimespec.tv_nsec &&
              (original.st_ctimespec.tv_sec != changed.st_ctimespec.tv_sec ||
               original.st_ctimespec.tv_nsec != changed.st_ctimespec.tv_nsec), "ctime fixture invalid");
        rejects([&] { verified.value->revalidate(); }, "file generation changed");
        // A retained descriptor still receives a complete second digest;
        // preserving size/mtime cannot make altered bytes match the first one.
        check(hash(verified.value->descriptor(), no_cancel) != verified.value->digest, "second digest was reused");
        ++cases;
    }
    {
        write(file, "abc");
        std::atomic<bool> hashed{false}, release{false};
        tc::AsyncLoraPreflight task(file, [&](int fd, const std::atomic<bool> &stop) {
            auto digest = hash(fd, stop); hashed = true;
            while (!release.load() && !stop.load()) std::this_thread::sleep_for(1ms);
            return digest;
        });
        wait(hashed); { std::ofstream append(file, std::ios::binary | std::ios::app); append << "more"; }
        release = true; rejects([&] { task.take(no_cancel); }, "file generation changed"); ++cases;
    }
    {
        tc::AsyncLoraPreflight task(file, [](int, const std::atomic<bool> &) -> std::string {
            throw std::runtime_error("injected read failure");
        });
        rejects([&] { task.take(no_cancel); }, "injected read failure"); ++cases;
    }
    {
        std::atomic<bool> started{false}, finished{false}, cancelled{false};
        tc::AsyncLoraPreflight task(file, [&](int, const std::atomic<bool> &stop) -> std::string {
            started = true;
            while (!stop.load()) std::this_thread::sleep_for(1ms);
            finished = true; throw tc::Cancelled();
        });
        wait(started); cancelled = true;
        rejects([&] { task.take(cancelled); }, "generation cancelled");
        check(finished.load(), "cancelled task did not join"); ++cases;
    }
    {
        std::atomic<bool> started{false}, finished{false};
        rejects([&] {
            tc::AsyncLoraPreflight task(file, [&](int, const std::atomic<bool> &stop) -> std::string {
                started = true;
                while (!stop.load()) std::this_thread::sleep_for(1ms);
                finished = true; throw tc::Cancelled();
            });
            wait(started); throw std::runtime_error("owner primary exception");
        }, "owner primary exception");
        check(finished.load(), "owner unwind did not stop and join"); ++cases;
    }
    {
        write(file, "abc"); tc::LoraFileFingerprint source(file);
        struct Worker { int fd; std::string digest, error; } worker{source.descriptor(), {}, {}};
        pthread_attr_t attr; check(pthread_attr_init(&attr) == 0, "pthread attr failed");
        check(pthread_attr_setstacksize(&attr, 256 * 1024) == 0, "pthread stack failed");
        pthread_t thread;
        const int created = pthread_create(&thread, &attr, [](void *opaque) -> void * {
            auto &test = *static_cast<Worker *>(opaque);
            try { test.digest = hash(test.fd, no_cancel); }
            catch (const std::exception &error) { test.error = error.what(); }
            return nullptr;
        }, &worker);
        pthread_attr_destroy(&attr); check(created == 0, "pthread create failed");
        check(pthread_join(thread, nullptr) == 0 && worker.error.empty() && worker.digest == abc_sha,
              "FD hash did not fit 256 KiB stack"); ++cases;
    }
    std::cout << "PASS async LoRA preflight: " << cases << " CPU cases\n";
}

// Explicitly opt-in CPU/file-I/O observation; one full SHA, no model parser.
static int observe(const fs::path &source, const std::string &expected) {
    std::atomic<bool> cancelled{false};
    std::mutex mutex; std::condition_variable cv; bool done = false;
    std::thread deadline([&] {
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, 10s, [&] { return done; })) cancelled = true;
    });
    auto finish = [&] { { std::lock_guard lock(mutex); done = true; } cv.notify_one(); deadline.join(); };
    try {
        tc::AsyncLoraPreflight task(source, hash); auto result = task.take(cancelled);
        finish();
        std::cout << "{\"bytes\":" << result.value->bytes() << ",\"sha256\":\"" << result.value->digest
                  << "\",\"seconds\":" << result.seconds << ",\"pinned_match\":"
                  << (result.value->digest == expected ? "true" : "false")
                  << ",\"cache_state\":\"unknown\",\"scope\":\"one CPU-only initial SHA; no image speedup measured\"}\n";
        return result.value->digest == expected ? 0 : 1;
    } catch (...) { finish(); throw; }
}
int main(int argc, char **argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--observe") return observe(argv[2], argv[3]);
        check(argc == 3, "expected fixture directory and multi-chunk digest");
        run_cases(argv[1], argv[2]); return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
