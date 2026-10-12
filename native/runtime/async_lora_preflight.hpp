#pragma once

#include "async_preparation.hpp"
#include "../core/common.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tc {
class LoraFileFingerprint final {
    int fd_ = -1;
    std::filesystem::path named_, canonical_;
    struct stat generation_{};
    static bool same(const struct stat &a, const struct stat &b) {
        return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
            a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
            a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec &&
            S_ISREG(b.st_mode);
    }
  public:
    std::string digest;
    explicit LoraFileFingerprint(std::filesystem::path path) : named_(std::move(path)) {
        std::error_code error;
        canonical_ = std::filesystem::canonical(named_, error);
        require(!error, "cannot resolve LoRA source: " + named_.string());
        fd_ = open(canonical_.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        require(fd_ >= 0, "cannot open LoRA source: " + canonical_.string());
        // If construction throws, the destructor cannot close the descriptor.
        if (fstat(fd_, &generation_) || !S_ISREG(generation_.st_mode) || generation_.st_size < 0) {
            close(fd_); fd_ = -1;
            throw std::invalid_argument("LoRA source is not a regular file");
        }
        try { revalidate(); } catch (...) { close(fd_); fd_ = -1; throw; }
    }
    LoraFileFingerprint(const LoraFileFingerprint &) = delete;
    LoraFileFingerprint &operator=(const LoraFileFingerprint &) = delete;
    ~LoraFileFingerprint() { if (fd_ >= 0) close(fd_); }
    int descriptor() const { return fd_; }
    const std::filesystem::path &canonical() const { return canonical_; }
    uint64_t bytes() const { return static_cast<uint64_t>(generation_.st_size); }
    void revalidate() const {
        std::error_code error;
        const auto current = std::filesystem::canonical(named_, error);
        require(!error && current == canonical_, "LoRA canonical path changed during verification");
        struct stat held{}, named{}, target{};
        require(fstat(fd_, &held) == 0 && stat(named_.c_str(), &named) == 0 &&
                    stat(canonical_.c_str(), &target) == 0 && same(generation_, held) &&
                    same(generation_, named) && same(generation_, target),
                "LoRA file generation changed during verification");
    }
};

// A request owns this task and its stop flag. The factory owns a path and a
// CPU-only hasher, never a Request, MLX tensor or event callback. Destruction
// signals cancellation before joining, including owner exception unwinding.
class AsyncLoraPreflight final {
    using Clock = std::chrono::steady_clock;
    struct Completion {
        PreparationResult<LoraFileFingerprint> result;
        Clock::time_point start, finish;
    };
    std::shared_ptr<std::atomic<bool>> stop_ = std::make_shared<std::atomic<bool>>(false);
    std::future<Completion> future_;
  public:
    using Hasher = std::function<std::string(int, const std::atomic<bool> &)>;
    explicit AsyncLoraPreflight(std::filesystem::path path, Hasher hasher) {
        path = std::filesystem::absolute(path).lexically_normal();
        future_ = std::async(std::launch::async, [path = std::move(path), hasher = std::move(hasher), stop = stop_] {
            Completion done; done.start = Clock::now();
            try {
                if (stop->load(std::memory_order_relaxed)) throw Cancelled();
                auto file = std::make_unique<LoraFileFingerprint>(path);
                file->digest = hasher(file->descriptor(), *stop);
                file->revalidate();
                done.result.value = std::move(file);
            } catch (...) { done.result.error = std::current_exception(); }
            done.finish = Clock::now(); return done;
        });
    }
    AsyncLoraPreflight(const AsyncLoraPreflight &) = delete;
    AsyncLoraPreflight &operator=(const AsyncLoraPreflight &) = delete;
    ~AsyncLoraPreflight() {
        stop_->store(true, std::memory_order_relaxed);
        if (future_.valid()) { try { (void)future_.get(); } catch (...) {} }
    }
    PreparationResult<LoraFileFingerprint> take(const std::atomic<bool> &cancelled) {
        if (!future_.valid()) throw std::logic_error("LoRA verification was already consumed");
        const auto join = Clock::now();
        while (future_.wait_for(std::chrono::milliseconds(5)) != std::future_status::ready)
            if (cancelled.load(std::memory_order_relaxed)) stop_->store(true, std::memory_order_relaxed);
        auto done = future_.get();
        const auto wait = Clock::now();
        if (cancelled.load(std::memory_order_relaxed)) throw Cancelled();
        if (done.result.error) std::rethrow_exception(done.result.error);
        require(bool(done.result.value), "LoRA verification returned no source");
        done.result.value->revalidate();
        done.result.seconds = std::chrono::duration<double>(done.finish - done.start).count();
        done.result.wait_seconds = std::chrono::duration<double>(wait - join).count();
        done.result.before_join_seconds = std::max(0., std::chrono::duration<double>(
            std::min(join, done.finish) - done.start).count());
        return std::move(done.result);
    }
};
} // namespace tc
