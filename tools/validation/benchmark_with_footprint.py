"""Wrap benchmark_native.py with self-process physical-footprint sampling.

Usage: python benchmark_with_footprint.py --library ... --output ... (same args)
Build helper with: xcrun clang -dynamiclib -O2 tools/validation/task_footprint.c
  -o build/native/task-footprint.dylib
Phys footprint includes this process's accounted device allocations, not all
Core ML service processes or ANE-exclusive residency. Do not add it to MLX/RSS.
"""
import ctypes
import json
from pathlib import Path
import runpy
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[sys.argv.index('--output')+1])
if out.exists():
    raise ValueError('Use a new output directory')
out.mkdir(parents=True)
lib = ctypes.CDLL(str(ROOT/'build/native/task-footprint.dylib'))
lib.tc_task_footprint.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
rows=[]
stop=threading.Event()
start=time.perf_counter()


def sample():
    while not stop.is_set():
        v=(ctypes.c_uint64*3)()
        if lib.tc_task_footprint(v):
            raise RuntimeError('task_info failed')
        rows.append(dict(seconds=time.perf_counter()-start,phys_footprint_bytes=v[0],
                         resident_bytes=v[1],ledger_peak_bytes=v[2]))
        stop.wait(.025)


worker=threading.Thread(target=sample,daemon=True)
worker.start()
try:
    runpy.run_path(str(ROOT/'tools/native/benchmark_native.py'),run_name='__main__')
finally:
    stop.set()
    worker.join()
    report=dict(interval_seconds=.025,samples=rows,
                sampled_peak_phys_footprint_bytes=max(r['phys_footprint_bytes'] for r in rows),
                ledger_peak_phys_footprint_bytes=max(r['ledger_peak_bytes'] for r in rows),
                scope='Self-process task_info; excludes other Core ML service processes; not ANE-exclusive RAM')
    (out/'footprint.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2),flush=True)
