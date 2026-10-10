#include "../../native/backends/gguf_read_snapshot.hpp"
#include <cassert>
#include <fstream>
#include <sys/mman.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    const auto root = std::filesystem::path(argv[1]);
    auto source = root / "read only model.gguf";
    const std::string contents(2 * 65536 + 17, 'q');
    { std::ofstream output(source, std::ios::binary); output << contents; }
    assert(::chmod(source.c_str(), 0444) == 0);
    struct stat initial{}; assert(::stat(source.c_str(), &initial) == 0);
    auto unchanged = [&] {
        struct stat now{}; assert(::stat(source.c_str(), &now) == 0);
        assert(now.st_ino == initial.st_ino && now.st_size == initial.st_size && now.st_mode == initial.st_mode);
        assert(now.st_mtimespec.tv_sec == initial.st_mtimespec.tv_sec && now.st_mtimespec.tv_nsec == initial.st_mtimespec.tv_nsec);
        assert(now.st_ctimespec.tv_sec == initial.st_ctimespec.tv_sec && now.st_ctimespec.tv_nsec == initial.st_ctimespec.tv_nsec);
        std::ifstream input(source, std::ios::binary);
        assert(std::string(std::istreambuf_iterator<char>(input), {}) == contents);
    };
    for (bool force_copy : {false, true}) {
        std::filesystem::path private_file, private_directory;
        {
            tc::GgufReadSnapshot snapshot(source, force_copy);
            if (force_copy) assert(!snapshot.cloned());
            private_file = snapshot.path(); private_directory = private_file.parent_path();
            struct stat status{}; assert(::stat(private_file.c_str(), &status) == 0);
            assert((status.st_mode & 0777) == 0600 && status.st_ino != initial.st_ino);
            assert(::stat(private_directory.c_str(), &status) == 0 && (status.st_mode & 0777) == 0700);
            int fd = ::open(private_file.c_str(), O_RDWR | O_APPEND); assert(fd >= 0);
            auto *mapping = static_cast<unsigned char *>(::mmap(nullptr, contents.size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
            assert(mapping != MAP_FAILED);
            // Deliberately dirty the compatibility copy, never the model.
            mapping[0] = 'x';
            assert(::munmap(mapping, contents.size()) == 0); ::close(fd);
            snapshot.check_unchanged(); unchanged();
        }
        assert(!std::filesystem::exists(private_file) && !std::filesystem::exists(private_directory));
        unchanged();
    }
    { tc::GgufReadSnapshot snapshot(source, true);
      assert(::chmod(source.c_str(), 0644) == 0);
      bool rejected = false;
      try { snapshot.check_unchanged(); } catch (const std::exception &) { rejected = true; }
      assert(rejected); }
    bool rejected = false;
    try { tc::GgufReadSnapshot snapshot(root / "missing.gguf"); } catch (const std::exception &) { rejected = true; }
    assert(rejected);
}
