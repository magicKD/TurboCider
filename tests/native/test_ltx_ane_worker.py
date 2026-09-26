"""Exercise the native scheduling code with a CPU prediction stand-in.

This tests pthread lifecycle/publication, not Core ML correctness or speed.
The scheduler is extracted verbatim so changes to it are exercised directly.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/models/ltx_runtime/ltx_ane_mlp.m"


class WorkerLifecycleTests(unittest.TestCase):
    def test_native_scheduler(self):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = SOURCE.read_text()
        scheduler = source.split("static void *ane_persistent_thread", 1)[1]
        scheduler = "static void *ane_persistent_thread" + scheduler.split(
            "ltx_ane_mlp *ltx_ane_mlp_create", 1
        )[0]
        cleanup = source.split("void ltx_ane_mlp_free(ltx_ane_mlp *mlp) {", 1)[1]
        cleanup = cleanup.split("@autoreleasepool", 1)[0]
        harness = r'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    pthread_t thread;
    pthread_mutex_t worker_mutex;
    pthread_cond_t worker_condition;
    int worker_started, worker_pending, worker_stop;
    int inflight, async_ok, fail, predictions;
    double async_ms;
    char async_error[1024];
} ltx_ane_mlp;
static void ane_fail(char *error, size_t size, const char *format, ...) {
    if (!error || !size) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, size, format, args);
    va_end(args);
}
static void *ane_prediction_thread(void *opaque) {
    ltx_ane_mlp *mlp = opaque;
    mlp->predictions++;
    mlp->async_ok = !mlp->fail;
    snprintf(mlp->async_error, sizeof(mlp->async_error), "mock failure");
    return NULL;
}
'''
        harness += scheduler
        harness += "static void destroy(ltx_ane_mlp *mlp) {" + cleanup + "}\n"
        harness += r'''
int main(void) {
    char error[1024];
    assert(!ane_start(NULL, error, sizeof(error)));
    for (int enabled = 0; enabled <= 1; enabled++) {
        setenv("TURBOCIDER_LTX_ANE_PERSISTENT_WORKER", enabled ? "1" : "0", 1);
        for (int repetition = 0; repetition < 4; repetition++) {
            ltx_ane_mlp mlp = {0};
            assert(!ane_wait(&mlp, error, sizeof(error)));
            for (int index = 0; index < 250; index++) {
                mlp.fail = (index % 7 == 0);
                assert(ane_start(&mlp, error, sizeof(error)));
                assert(!ane_start(&mlp, error, sizeof(error)));
                assert(ane_wait(&mlp, error, sizeof(error)) == !mlp.fail);
                assert(mlp.predictions == index + 1);
                assert(mlp.worker_started == enabled);
                assert(!mlp.inflight);
            }
            /* Destruction also drains an outstanding prediction. */
            mlp.fail = 0;
            assert(ane_start(&mlp, error, sizeof(error)));
            destroy(&mlp);
            assert(mlp.predictions == 251);
        }
    }
    ltx_ane_mlp unused = {0};
    destroy(&unused);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="ltx-ane-worker-") as directory:
            directory = Path(directory)
            c_path = directory / "worker.c"
            executable = directory / "worker"
            c_path.write_text(harness)
            subprocess.run(
                [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-pthread", str(c_path), "-o", str(executable)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
