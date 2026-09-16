// Compile the real parser with a test-only strdup failure point. Nothing is
// injected into the production dylib or normal inference configuration.
#import <Foundation/Foundation.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fail_at = -1, copy_count = 0;
static char *fault_strdup(const char *value) {
    if (copy_count++ == fail_at) return NULL;
    return strdup(value);
}
#define strdup fault_strdup
#include "../../native/models/ltx_runtime/ltx_safetensors.m"
#undef strdup

static size_t descriptors(void) {
    DIR *directory = opendir("/dev/fd"); assert(directory);
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory))) if (entry->d_name[0] != '.') ++count;
    closedir(directory);
    return count;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    const int fd = open(argv[1], O_RDONLY | O_CLOEXEC); assert(fd >= 0);
    char error[1024] = {0}; ltx_st_header header = {0};
    assert(ltx_st_read_header_fd(fd, argv[1], &header, error, sizeof(error)));
    assert(header.tensor_count == 1 && header.metadata_config && header.metadata_gemma_config);
    const int copies = copy_count;
    assert(copies == 4); // tensor name, path, two metadata strings
    ltx_st_free_header(&header);
    const size_t before = descriptors();
    for (int use_fd = 0; use_fd < 2; ++use_fd) {
        for (int i = 0; i < copies; ++i) {
            fail_at = i; copy_count = 0; error[0] = 0;
            const int ok = use_fd ? ltx_st_read_header_fd(fd, argv[1], &header, error, sizeof(error))
                                  : ltx_st_read_header(argv[1], &header, error, sizeof(error));
            assert(!ok && error[0]);
            assert(!header.path && !header.tensors && !header.tensor_count &&
                   !header.metadata_config && !header.metadata_gemma_config && !header.snapshot.valid);
            assert(fcntl(fd, F_GETFD) >= 0 && descriptors() == before);
            ltx_st_free_header(&header); // cleared failed output is safe to clean again
        }
    }
    fail_at = -1; copy_count = 0;
    assert(ltx_st_read_header_fd(fd, argv[1], &header, error, sizeof(error)));
    assert(ltx_st_validate_snapshot_fd(&header, fd, argv[1], error, sizeof(error)));
    ltx_st_free_header(&header); close(fd);
    puts("PASS safetensors header failure injection: name/path/metadata, legacy+fd, no double free/fd leak");
    return 0;
}
