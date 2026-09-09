"""No weights: the App-owned service exits when its parent disappears."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

CLI = Path(__file__).resolve().parents[2] / 'build/native/turbocider'

class LifecycleTests(unittest.TestCase):
    def test_parent_exit_releases_socket_and_state(self):
        with tempfile.TemporaryDirectory(prefix='tc-api-owner-') as tmp:
            sock = '/private/tmp/tc-owner-' + uuid.uuid4().hex[:8] + '.sock'
            log = Path(tmp) / 'child.log'
            owner_code = '''
import os,subprocess,sys,time
with open(sys.argv[3],'w') as log:
 p=subprocess.Popen([sys.argv[1],'serve',sys.argv[2],sys.argv[4]],env={**os.environ,'TURBOCIDER_SERVICE_PARENT_PID':str(os.getpid())},stdout=log,stderr=log)
 print(p.pid,flush=True)
 sys.stdin.readline()
'''
            owner = subprocess.Popen([sys.executable, '-u', '-c', owner_code, str(CLI), sock, str(log), tmp], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            child_pid = int(owner.stdout.readline())
            replacement = None
            try:
                deadline = time.monotonic() + 10
                while not Path(sock).exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(Path(sock).exists(), log.read_text())
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(5); client.connect(sock)
                    client.sendall(b'{"action":"service_status"}\n')
                    result = json.loads(client.recv(65536))
                    self.assertEqual(result['result']['pid'], child_pid)
                owner.kill(); owner.wait(timeout=5)
                deadline = time.monotonic() + 10
                while Path(sock).exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertFalse(Path(sock).exists(), 'Orphan service retained socket')
                replacement = subprocess.Popen([str(CLI), 'serve', sock, tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertEqual(json.loads(replacement.stdout.readline()), {'ready': True})
                replacement.terminate(); self.assertEqual(replacement.wait(timeout=10), 0)
            finally:
                if owner.poll() is None: owner.kill(); owner.wait(timeout=5)
                owner.stdin.close(); owner.stdout.close()
                if replacement:
                    if replacement.poll() is None: replacement.kill(); replacement.wait(timeout=5)
                    replacement.stdout.close(); replacement.stderr.close()
                # Only kill the known child if it is still our service at this socket.
                if Path(sock).exists():
                    try: os.kill(child_pid, 15)
                    except ProcessLookupError: pass
                Path(sock + '.lock').unlink(missing_ok=True)

if __name__ == '__main__': unittest.main()
