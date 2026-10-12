// Filesystem/locking contracts only: no Core ML, MLX, model or GPU dependency.
#include "../../native/backends/ane_artifact_lease.hpp"
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sys/wait.h>

using Lease = tc::ane::detail::ArtifactLease;
namespace fs = std::filesystem;

static void write(const fs::path &path, const std::string &text = "compiled fixture") {
    std::ofstream stream(path);
    stream << text;
    stream.close();
    assert(stream.good());
}
static void graph(const fs::path &root) {
    fs::create_directories(root / "graph.mlmodelc/subdirectory");
    fs::permissions(root / "graph.mlmodelc", fs::perms::owner_all);
    fs::permissions(root / "graph.mlmodelc/subdirectory", fs::perms::owner_all);
    write(root / "graph.mlmodelc/model.mil");
    write(root / "graph.mlmodelc/subdirectory/metadata.bin");
    fs::permissions(root / "graph.mlmodelc/model.mil", fs::perms::owner_read | fs::perms::owner_write);
    fs::permissions(root / "graph.mlmodelc/subdirectory/metadata.bin", fs::perms::owner_read | fs::perms::owner_write);
}
static std::string read(const fs::path &path) {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
static void write_pipe(int fd, const std::string &text) {
    size_t offset = 0;
    while (offset < text.size()) {
        const auto amount = ::write(fd, text.data() + offset, text.size() - offset);
        if (amount < 0 && errno == EINTR) continue;
        assert(amount > 0);
        offset += size_t(amount);
    }
}
static std::pair<pid_t, fs::path> live_child(const fs::path &temporary,
                                          const fs::path &external = {}, bool hardlink = false) {
    int pipe_fds[2];
    assert(::pipe(pipe_fds) == 0);
    const auto pid = ::fork();
    assert(pid >= 0);
    if (!pid) {
        ::close(pipe_fds[0]);
        try {
            Lease lease(temporary);
            graph(lease.root);
            if (!external.empty()) {
                if (hardlink) fs::create_hard_link(external, lease.root / "graph.mlmodelc/unsafe.bin");
                else fs::create_directory_symlink(external, lease.root / "graph.mlmodelc/unsafe.bin");
            }
            write_pipe(pipe_fds[1], lease.root.string() + "\n");
            ::close(pipe_fds[1]);
            for (;;) ::pause();
        } catch (...) { ::_exit(2); }
    }
    ::close(pipe_fds[1]);
    std::string path;
    char ch = 0;
    for (;;) {
        const auto amount = ::read(pipe_fds[0], &ch, 1);
        if (amount < 0 && errno == EINTR) continue;
        assert(amount == 1);
        if (ch == '\n') break;
        path.push_back(ch);
    }
    ::close(pipe_fds[0]);
    return {pid, fs::path(path)};
}
static void crash(pid_t child) {
    assert(::kill(child, SIGKILL) == 0);
    int status = 0;
    assert(::waitpid(child, &status, 0) == child);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
}

int main() {
    char pattern[] = "/tmp/tc-ane-lease-host-XXXXXX";
    const char *created = ::mkdtemp(pattern);
    assert(created);
    const fs::path temporary = created;
    std::vector<std::string> messages;
    const auto reporter = [&](const std::string &message) { messages.push_back(message); };
    const auto marker = ".runtime-ane-lease-v1";

    // A separate open/lock in the same process must not reap a live instance.
    fs::path first;
    {
        Lease lease(temporary, reporter);
        first = lease.root;
        graph(first);
        const auto result = Lease::reap(temporary, reporter);
        assert(result.live == 1 && result.removed == 0 && result.failed == 0);
        assert(read(first / "graph.mlmodelc/model.mil") == "compiled fixture");
        {
            Lease other(temporary, reporter);
            graph(other.root);
            assert(fs::exists(first));
            assert(Lease::reap(temporary, reporter).live == 2);
        }
        assert(fs::exists(first));
    }
    assert(!fs::exists(first) && fs::is_empty(temporary));
    assert(messages.empty());
    for (mode_t mask : {mode_t(0000), mode_t(0002), mode_t(0077)}) {
        const auto previous = ::umask(mask);
        fs::path directory;
        {
            Lease lease(temporary, reporter);
            directory = lease.root;
            graph(directory);
        }
        ::umask(previous);
        assert(!fs::exists(directory) && fs::is_empty(temporary));
    }

    // A live OTHER process remains untouched; SIGKILL releases its lock and
    // the next recovery removes its complete graph without relying on a PID.
    const auto [child, abandoned] = live_child(temporary);
    assert(Lease::reap(temporary, reporter).live == 1);
    assert(fs::exists(abandoned / "graph.mlmodelc/model.mil"));
    crash(child);
    const auto recovered = Lease::reap(temporary, reporter);
    assert(recovered.removed == 1 && recovered.failed == 0 && !fs::exists(abandoned));

    // Old names and unmarked directories are never claimed as owned leases.
    const auto old = temporary / "turbocider-runtime-ane-ABC123";
    const auto unknown = temporary / "turbocider-runtime-ane-0000000000000001";
    graph(old);
    graph(unknown);
    assert(Lease::reap(temporary, reporter).removed == 0);
    assert(fs::exists(old / "graph.mlmodelc/model.mil"));
    assert(fs::exists(unknown / "graph.mlmodelc/model.mil"));
    fs::remove_all(old); fs::remove_all(unknown);

    // Copying a legitimate marker into a DIFFERENT directory cannot transfer
    // ownership: its recorded device/inode is bound to the original root.
    {
        Lease live(temporary, reporter);
        const auto replay = temporary / "turbocider-runtime-ane-0000000000000002";
        fs::create_directory(replay);
        assert(::chmod(replay.c_str(), 0700) == 0);
        graph(replay);
        write(replay / marker, read(live.root / marker));
        assert(::chmod((replay / marker).c_str(), 0600) == 0);
        assert(Lease::reap(temporary, reporter).removed == 0);
        assert(fs::exists(replay / "graph.mlmodelc/model.mil"));
        fs::remove_all(replay);
    }

    // Root symlinks and graph symlinks cannot touch their external target.
    const auto external = temporary / "external-data";
    fs::create_directory(external);
    write(external / "sentinel", "keep");
    const auto root_link = temporary / "turbocider-runtime-ane-0000000000000003";
    fs::create_directory_symlink(external, root_link);
    const auto before_link = messages.size();
    assert(Lease::reap(temporary, reporter).removed == 0);
    assert(messages.size() > before_link && read(external / "sentinel") == "keep");
    fs::remove(root_link);
    const auto [symlink_child, symlink_root] = live_child(temporary, external);
    crash(symlink_child);
    const auto before_graph_link = messages.size();
    assert(Lease::reap(temporary, reporter).failed == 1);
    assert(messages.size() > before_graph_link && read(external / "sentinel") == "keep");
    assert(fs::is_symlink(symlink_root / "graph.mlmodelc/unsafe.bin"));
    fs::remove(symlink_root / "graph.mlmodelc/unsafe.bin");
    assert(Lease::reap(temporary, reporter).removed == 1);

    // Hardlinks and widened root permissions also fail closed. Reaping never
    // truncates or modifies the external file even when a graph is malformed.
    const auto [hardlink_child, hardlink_root] = live_child(temporary, external / "sentinel", true);
    crash(hardlink_child);
    assert(Lease::reap(temporary, reporter).failed == 1);
    assert(read(external / "sentinel") == "keep");
    fs::remove(hardlink_root / "graph.mlmodelc/unsafe.bin");
    assert(::chmod(hardlink_root.c_str(), 0755) == 0);
    assert(Lease::reap(temporary, reporter).removed == 0 && fs::exists(hardlink_root));
    assert(::chmod(hardlink_root.c_str(), 0700) == 0);
    assert(Lease::reap(temporary, reporter).removed == 1);

    // Explicit cleanup failure is observable; it does not erase foreign root
    // entries. Restoring the recognized layout permits an ordinary retry.
    {
        Lease lease(temporary, reporter);
        graph(lease.root);
        write(lease.root / "unexpected-user-file", "keep");
        const auto before_failure = messages.size();
        assert(!lease.cleanup() && messages.size() > before_failure);
        assert(read(lease.root / "unexpected-user-file") == "keep");
        assert(fs::exists(lease.root / marker));
        fs::remove(lease.root / "unexpected-user-file");
        assert(lease.cleanup() && lease.root.empty());
    }

    // Removal of graph/marker can succeed while final rmdir is denied by a
    // changed parent permission. Restore a NEW locked marker, then retry.
    // Root bypasses Unix permission checks, so this fixture is user-only.
    if (::geteuid() != 0) {
        Lease lease(temporary, reporter);
        graph(lease.root);
        const auto old_marker = read(lease.root / marker);
        assert(::chmod(temporary.c_str(), 0500) == 0);
        const auto before_failure = messages.size();
        assert(!lease.cleanup());
        assert(messages.size() > before_failure);
        assert(messages.back().find("locked marker restored") != std::string::npos);
        assert(!fs::exists(lease.root / "graph.mlmodelc"));
        assert(fs::exists(lease.root / marker) && read(lease.root / marker) != old_marker);
        assert(::chmod(temporary.c_str(), 0700) == 0);
        assert(Lease::reap(temporary, reporter).live == 1);
        assert(lease.cleanup() && lease.root.empty());
    }

    // A substituted marker inode must not act as the still-live owner's lock.
    {
        Lease lease(temporary, reporter);
        const auto original = temporary / "saved-marker";
        fs::rename(lease.root / marker, original);
        write(lease.root / marker, read(original));
        assert(::chmod((lease.root / marker).c_str(), 0600) == 0);
        assert(!lease.cleanup());
        // Even a copied marker body in an otherwise recognized root cannot
        // transfer the old lock to a replacement inode that looks unlocked.
        assert(Lease::reap(temporary, reporter).removed == 0);
        assert(fs::exists(original));
        fs::remove(lease.root / marker);
        fs::rename(original, lease.root / marker);
        assert(lease.cleanup());
    }
    assert(read(external / "sentinel") == "keep");
    fs::remove_all(external);
    assert(fs::is_empty(temporary));
    fs::remove(temporary);
    std::cout << "PASS private artifact locks, crash recovery, ownership and observable refusal\n";
}
