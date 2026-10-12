#pragma once
#import <Foundation/Foundation.h>
#include "../../native/core/json_keys.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace tc_service {
inline constexpr size_t library_inventory_stdout_max = 8 * 1024 * 1024;
inline constexpr size_t library_inventory_stderr_max = 64 * 1024;

// Accessed only by the socket-serving thread. A child stuck in kernel I/O
// after SIGKILL must not turn a bounded metadata query into an infinite wait.
// The service loop retries WNOHANG for these owned PIDs; it never reaps a
// generation worker or sends a signal after the child has been reaped.
inline std::vector<pid_t> &inventory_deferred_children() {
    static std::vector<pid_t> children;
    return children;
}
inline void reap_inventory_children() noexcept {
    auto &children = inventory_deferred_children();
    children.erase(std::remove_if(children.begin(), children.end(), [](pid_t child) {
        int status = 0;
        const pid_t result = ::waitpid(child, &status, WNOHANG);
        return result == child || (result < 0 && errno == ECHILD);
    }), children.end());
}

struct InventoryFD {
    int value = -1;
    InventoryFD() = default;
    InventoryFD(const InventoryFD &) = delete;
    InventoryFD &operator=(const InventoryFD &) = delete;
    ~InventoryFD() { reset(); }
    void reset(int descriptor = -1) noexcept {
        if (value >= 0) ::close(value);
        value = descriptor;
    }
};
inline void inventory_pipe(InventoryFD &read, InventoryFD &write) {
    int pipeFDs[2];
    if (::pipe(pipeFDs) != 0) throw std::runtime_error("installations: cannot create helper pipe");
    read.reset(pipeFDs[0]); write.reset(pipeFDs[1]);
    // Keep pipe descriptors away from stdin/stdout/stderr, even when a CLI
    // was launched with one of its standard descriptors closed.
    for (InventoryFD *fd : {&read, &write}) {
        const int duplicate = ::fcntl(fd->value, F_DUPFD_CLOEXEC, 3);
        if (duplicate < 0) throw std::runtime_error("installations: cannot configure helper pipe");
        fd->reset(duplicate);
    }
    const int flags = ::fcntl(read.value, F_GETFL);
    if (flags < 0 || ::fcntl(read.value, F_SETFL, flags | O_NONBLOCK) != 0)
        throw std::runtime_error("installations: cannot configure nonblocking helper output");
}

struct InventoryChild {
    pid_t pid = -1;
    int status = 0;
    std::chrono::steady_clock::time_point deadline;
    explicit InventoryChild(std::chrono::steady_clock::time_point limit) : deadline(limit) {}
    InventoryChild(const InventoryChild &) = delete;
    InventoryChild &operator=(const InventoryChild &) = delete;
    bool exited() {
        if (pid < 0) return true;
        const pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == pid) { pid = -1; return true; }
        if (result < 0 && errno != EINTR) {
            if (errno == ECHILD) pid = -1; // Ownership has gone; never signal a recycled PID.
            throw std::runtime_error("installations: cannot read helper exit status");
        }
        return false;
    }
    ~InventoryChild() {
        if (pid < 0) return;
        // This PID belongs to this invocation, separately from active_child_.
        ::kill(pid, SIGKILL);
        const auto cleanupDeadline = std::min(deadline,
            std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
        do {
            const pid_t result = ::waitpid(pid, &status, WNOHANG);
            if (result == pid || (result < 0 && errno == ECHILD)) return;
            if (std::chrono::steady_clock::now() >= cleanupDeadline) break;
            (void)::poll(nullptr, 0, 5);
        } while (true);
        inventory_deferred_children().push_back(pid);
    }
};

struct InventorySpawnActions {
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    InventorySpawnActions() {
        if (::posix_spawn_file_actions_init(&actions) != 0)
            throw std::runtime_error("installations: cannot initialize helper launch");
        if (::posix_spawnattr_init(&attributes) != 0) {
            ::posix_spawn_file_actions_destroy(&actions);
            throw std::runtime_error("installations: cannot initialize helper launch attributes");
        }
    }
    ~InventorySpawnActions() {
        ::posix_spawnattr_destroy(&attributes);
        ::posix_spawn_file_actions_destroy(&actions);
    }
};
inline void inventory_spawn_check(int status) {
    if (status != 0) throw std::runtime_error("installations: cannot configure helper launch: " + std::string(std::strerror(status)));
}
inline void inventory_read(InventoryFD &fd, std::string &output, size_t maximum, const char *stream) {
    // Bound each drain so a writer cannot starve stop/deadline checks.
    char buffer[4096];
    for (int chunk = 0; chunk < 16 && fd.value >= 0; ++chunk) {
        const ssize_t count = ::read(fd.value, buffer, sizeof(buffer));
        if (count > 0) {
            if (size_t(count) > maximum - output.size())
                throw std::runtime_error(std::string("installations: helper ") + stream + " exceeds its size limit");
            output.append(buffer, size_t(count));
        } else if (count == 0) fd.reset();
        else if (errno == EINTR) continue;
        else if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        else throw std::runtime_error(std::string("installations: cannot read helper ") + stream);
    }
}
inline std::string inventory_error_detail(const std::string &text) {
    // A failed helper can include a small diagnostic; never expose its entire
    // 64 KiB stderr buffer in the RPC envelope.
    std::string detail = text.substr(0, 2048);
    if (detail.find('\0') != std::string::npos ||
        ![[NSString alloc] initWithBytes:detail.data() length:detail.size() encoding:NSUTF8StringEncoding])
        return "invalid helper diagnostic";
    return detail;
}
inline void inventory_require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("installations: ") + message);
}
inline bool inventory_version(id value) {
    return [value isKindOfClass:NSNumber.class] &&
        CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() && [value doubleValue] == 1;
}
inline bool inventory_boolean(id value, bool expected) {
    return [value isKindOfClass:NSNumber.class] &&
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() && [value boolValue] == expected;
}
inline NSDictionary *inventory_result(const std::string &wire) {
    std::string text = wire;
    if (!text.empty() && text.back() == '\n') text.pop_back();
    if (!text.empty() && text.back() == '\r') text.pop_back();
    inventory_require(!text.empty() && text.find_first_of("\r\n") == std::string::npos,
                      "helper must return one JSON line; update turbocider-library if its protocol is older");
    try { tc::reject_duplicate_json_keys(text); }
    catch (const std::exception &error) {
        throw std::runtime_error("installations: invalid helper JSON: " + std::string(error.what()));
    }
    id decoded = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:text.data() length:text.size()] options:0 error:nil];
    inventory_require([decoded isKindOfClass:NSDictionary.class], "helper returned invalid JSON");
    NSDictionary *envelope = decoded;
    if (inventory_boolean(envelope[@"ok"], false)) {
        NSString *message = [envelope[@"error"] isKindOfClass:NSString.class] ? envelope[@"error"] : @"helper rejected inventory";
        throw std::runtime_error("installations: " + inventory_error_detail(message.UTF8String ?: "helper rejected inventory"));
    }
    inventory_require(inventory_boolean(envelope[@"ok"], true) && envelope.count == 2 &&
        [envelope[@"result"] isKindOfClass:NSDictionary.class], "helper success envelope is incompatible; update turbocider-library");
    NSDictionary *result = envelope[@"result"];
    inventory_require(result.count == 5 && inventory_version(result[@"schema_version"]) &&
        [result[@"root"] isKindOfClass:NSString.class] && [result[@"root"] isAbsolutePath] &&
        [result[@"scope"] isEqual:@"registered_metadata"] && inventory_boolean(result[@"files_verified"], false) &&
        [result[@"index"] isKindOfClass:NSDictionary.class], "helper inventory schema is incompatible; update turbocider-library");
    NSString *root = result[@"root"];
    inventory_require(root.UTF8String && std::strlen(root.UTF8String) == [root lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
                      "helper root contains an embedded NUL");
    NSDictionary *index = result[@"index"];
    inventory_require(inventory_version(index[@"schemaVersion"]) && [index[@"installations"] isKindOfClass:NSArray.class],
                      "helper returned an invalid model-library index");
    for (NSString *key in index)
        inventory_require([@[@"schemaVersion", @"installations", @"loras", @"anePartitions"] containsObject:key],
                          "helper model-library index has unsupported fields");
    for (NSString *key in @[@"installations", @"loras", @"anePartitions"]) {
        id entries = index[key]; if (!entries) continue;
        inventory_require([entries isKindOfClass:NSArray.class], "helper model-library entries must be arrays");
        for (id entry in entries)
            inventory_require([entry isKindOfClass:NSDictionary.class], "helper model-library entry must be an object");
    }
    return result;
}

inline NSDictionary *library_inventory(const std::string &executable, const std::function<bool()> &shouldStop) {
    reap_inventory_children();
    inventory_require(!shouldStop(), "service is stopping");
    std::error_code error;
    const auto path = std::filesystem::absolute(executable, error).lexically_normal();
    inventory_require(!error && path.is_absolute(), "cannot locate the service executable");
    const auto helper = path.parent_path() / "turbocider-library";
    struct stat info {};
    inventory_require(::stat(helper.c_str(), &info) == 0 && S_ISREG(info.st_mode) && ::access(helper.c_str(), X_OK) == 0,
                      "turbocider-library is missing or not executable beside the service; install matching App/CLI helpers");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    InventoryFD outputRead, outputWrite, errorRead, errorWrite;
    inventory_pipe(outputRead, outputWrite); inventory_pipe(errorRead, errorWrite);
    InventorySpawnActions launch;
    inventory_spawn_check(::posix_spawnattr_setflags(&launch.attributes, POSIX_SPAWN_CLOEXEC_DEFAULT));
    inventory_spawn_check(::posix_spawn_file_actions_addopen(&launch.actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0));
    inventory_spawn_check(::posix_spawn_file_actions_adddup2(&launch.actions, outputWrite.value, STDOUT_FILENO));
    inventory_spawn_check(::posix_spawn_file_actions_adddup2(&launch.actions, errorWrite.value, STDERR_FILENO));
    for (int descriptor : {outputRead.value, outputWrite.value, errorRead.value, errorWrite.value})
        inventory_spawn_check(::posix_spawn_file_actions_addclose(&launch.actions, descriptor));
    std::vector<std::string> arguments = {helper.string(), "inventory"};
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    std::vector<std::string> environment;
    for (char **entry = environ; entry && *entry; ++entry) {
        std::string value(*entry);
        if (!value.starts_with("TURBOCIDER_LIBRARY_PARENT_PID=")) environment.push_back(std::move(value));
    }
    environment.emplace_back("TURBOCIDER_LIBRARY_PARENT_PID=" + std::to_string(::getpid()));
    std::vector<char *> envp;
    for (auto &entry : environment) envp.push_back(entry.data());
    envp.push_back(nullptr);
    InventoryChild child(deadline);
    const int spawnStatus = ::posix_spawn(&child.pid, helper.c_str(), &launch.actions, &launch.attributes, argv.data(), envp.data());
    if (spawnStatus != 0) {
        child.pid = -1;
        throw std::runtime_error("installations: cannot launch turbocider-library: " + std::string(std::strerror(spawnStatus)));
    }
    outputWrite.reset(); errorWrite.reset();
    std::string output, diagnostic;
    bool exited = false;
    while (true) {
        inventory_require(!shouldStop(), "service stopped during inventory");
        inventory_require(std::chrono::steady_clock::now() < deadline, "library inventory exceeded its 10 second deadline");
        inventory_read(outputRead, output, library_inventory_stdout_max, "stdout");
        inventory_read(errorRead, diagnostic, library_inventory_stderr_max, "stderr");
        if (!exited) exited = child.exited();
        if (exited && outputRead.value < 0 && errorRead.value < 0) break;
        pollfd descriptors[] = {{outputRead.value, POLLIN, 0}, {errorRead.value, POLLIN, 0}};
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        const int timeout = int(std::clamp<int64_t>(remaining, 0, 50));
        const int polled = ::poll(descriptors, 2, timeout);
        if (polled < 0 && errno != EINTR) throw std::runtime_error("installations: cannot poll helper output");
    }
    if (!WIFEXITED(child.status) || WEXITSTATUS(child.status) != 0) {
        // Decode a structured helper error when present, even on exit 1.
        if (!output.empty()) {
            try { (void)inventory_result(output); }
            catch (const std::exception &failure) { throw std::runtime_error(failure.what()); }
        }
        const std::string detail = inventory_error_detail(diagnostic);
        throw std::runtime_error("installations: turbocider-library exited unsuccessfully" +
                                 (detail.empty() ? std::string() : ": " + detail));
    }
    return inventory_result(output);
}
} // namespace tc_service
