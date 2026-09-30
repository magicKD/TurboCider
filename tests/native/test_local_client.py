"""Small CPU/socket regressions; no daemon, model, Metal or downloads."""
import importlib.util
import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / "bindings/python/turbocider_local.py"
spec = importlib.util.spec_from_file_location("turbocider_local", MODULE)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)


class ClientTests(unittest.TestCase):
    def exchange(self, chunks, action="models", *, max_bytes=1024, **fields):
        with tempfile.TemporaryDirectory(prefix="tc-client-", dir="/tmp") as tmp:
            path = str(Path(tmp) / "api.sock")
            captured, failures = [], []
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(path)
                listener.listen(1)
                listener.settimeout(3)

                def serve():
                    try:
                        connection, _ = listener.accept()
                        with connection:
                            connection.settimeout(3)
                            with connection.makefile("rb") as stream:
                                captured.append(json.loads(stream.readline()))
                            for part in chunks:
                                connection.sendall(part)
                    except Exception as error:
                        failures.append(error)

                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                try:
                    result = api.Client(path, timeout=2, max_response_bytes=max_bytes).rpc(action, **fields)
                finally:
                    thread.join(timeout=4)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(failures, failures)
                    self.assertEqual(captured, [{"action": action, **fields}])
                return result

    def test_fragmented_utf8_response(self):
        payload = json.dumps({"ok": True, "result": {"name": "图片"}}, ensure_ascii=False).encode() + b"\n"
        self.assertEqual(self.exchange([payload[:7], payload[7:35], payload[35:]]), {"name": "图片"})

    def test_service_error(self):
        with self.assertRaisesRegex(api.APIError, "invalid history"):
            self.exchange([b'{"ok":false,"error":"invalid history"}\n'])

    def test_lost_submit_reply_is_ambiguous(self):
        with self.assertRaises(api.TransportError) as error:
            self.exchange([], "submit", model_path="/model", request={})
        self.assertTrue(error.exception.submission_may_have_succeeded)

    def test_read_failure_is_not_ambiguous_submission(self):
        with self.assertRaises(api.TransportError) as error:
            self.exchange([])
        self.assertFalse(error.exception.submission_may_have_succeeded)

    def test_duplicate_or_unframed_response_rejected(self):
        for response in [b'{"ok":true,"ok":false,"result":{}}\n',
                         b'{"ok":true,"result":{}}', b'{"ok":1,"result":{}}\n',
                         b'{"ok":true,"result":NaN}\n']:
            with self.subTest(response=response), self.assertRaises(api.TransportError):
                self.exchange([response])

    def test_response_limit(self):
        with self.assertRaisesRegex(api.TransportError, "exceeds"):
            self.exchange([b'{"ok":true,"result":"' + b'x' * 100 + b'"}\n'], max_bytes=32)

    def test_request_size_and_nonfinite_rejected_before_connection(self):
        client = api.Client("/does/not/exist")
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            client.rpc("plan", request={"prompt": "x" * 1048576})
        with self.assertRaises(ValueError):
            client.rpc("jobs", offset=float("nan"))

    def test_wait_uses_terminal_result_and_preserves_failure(self):
        client = api.Client("/unused")
        ready = {"id": "job", "state": "succeeded", "result": {"output": "/image.png"}}
        with patch.object(client, "_exchange", return_value=ready) as rpc:
            self.assertEqual(client.wait("job"), ready)
            self.assertEqual(rpc.call_args.args[:2], ("status", {"id": "job"}))
        for state in ["failed", "cancelled", "interrupted"]:
            value = {"id": "job", "state": state}
            with patch.object(client, "_exchange", return_value=value), self.assertRaises(api.JobFailed) as error:
                client.wait("job")
            self.assertEqual(error.exception.job, value)

    def test_timeout_keeps_id_and_never_cancels(self):
        client = api.Client("/unused")
        value = {"id": "job", "state": "running"}
        with patch.object(client, "_exchange", return_value=value) as rpc:
            with self.assertRaises(api.JobTimeout) as error:
                client.wait("job", timeout=.01, poll_interval=.1)
            self.assertEqual(error.exception.job_id, "job")
            self.assertEqual(error.exception.last_job, value)
            self.assertTrue(all(call.args[0] == "status" for call in rpc.call_args_list))

    def test_installations_validates_metadata_without_compatibility_claims(self):
        client = api.Client("/unused")
        valid = {"schema_version": 1, "root": "/registry", "scope": "registered_metadata",
                 "files_verified": False, "index": {"schemaVersion": 1, "installations": []}}
        with patch.object(client, "rpc", return_value=valid) as rpc:
            self.assertEqual(client.installations(), valid)
            rpc.assert_called_once_with("installations")
        invalid = [None, {**valid, "schema_version": True}, {**valid, "scope": "scanned"},
                   {**valid, "files_verified": True}, {**valid, "root": "relative"},
                   {**valid, "root": "/registry\0ignored"}]
        for key, value in (("schemaVersion", 2), ("installations", None),
                           ("loras", {}), ("anePartitions", ["not an object"])):
            item = copy.deepcopy(valid); item["index"][key] = value; invalid.append(item)
        for value in invalid:
            with self.subTest(value=value), patch.object(client, "rpc", return_value=value):
                with self.assertRaises(api.TransportError) as error:
                    client.installations()
                self.assertFalse(error.exception.submission_may_have_succeeded)

    def test_installations_on_old_service_has_no_fallback_or_retry(self):
        client = api.Client("/unused")
        with patch.object(client, "rpc", side_effect=api.APIError("unknown action")) as rpc:
            with self.assertRaisesRegex(api.APIError, "unknown action"):
                client.installations()
            rpc.assert_called_once_with("installations")


if __name__ == "__main__":
    unittest.main()
