#!/usr/bin/env python3
"""Native strict envelope checks plus Foundation/Swift request digest golden."""
from pathlib import Path
import subprocess
import os
import tempfile
root=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='tc-worker-protocol-') as tmp:
    work=Path(tmp);native=work/'native';swift=work/'swift';wire=work/'wire.json'
    subprocess.run(['clang++','-std=c++20','-fobjc-arc','-Wall','-Wextra','-Werror',str(root/'tests/native/worker_protocol_test.mm'),'-framework','Foundation','-o',str(native)],check=True)
    subprocess.run([str(native)],check=True)
    subprocess.run(['swiftc',str(root/'apps/macos/WorkerRequestEnvelope.swift'),str(root/'tests/integration/WorkerRequestEnvelopeTests.swift'),'-o',str(swift)],check=True)
    subprocess.run([str(swift),str(root/'tests/fixtures/streaming/worker-request-v1.json'),str(wire)],check=True)
    subprocess.run([str(native),str(wire)],check=True)
    link=work/'symlink.json';link.symlink_to(wire)
    fifo=work/'fifo';os.mkfifo(fifo)
    for path in [link,work/'empty.json',work/'oversize.json',fifo]:
        if path==work/'empty.json':path.write_bytes(b'')
        if path==work/'oversize.json':path.write_bytes(b' '*(1+(1<<20)))
        result=subprocess.run([str(native),str(path)],capture_output=True,timeout=5)
        expected=b'worker_input_open_failed' if path==link else b'worker_input_size_invalid'
        assert result.returncode==2 and expected in result.stderr,'wrong input file rejection'
    print('input symlink/empty/oversize/FIFO rejection PASS')
