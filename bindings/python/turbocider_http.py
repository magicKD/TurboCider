"""Dependency-free client for the optional TurboCider HTTP gateway. Never retries."""
from __future__ import annotations
import json
import math
import urllib.error
import urllib.parse
import urllib.request


class HTTPError(RuntimeError):
    pass


class HTTPClient:
    def __init__(self, url="http://127.0.0.1:8765", token="", timeout=30):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Expected an http(s) API URL without credentials, query or fragment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def exchange(self, method, path, body=None, content_type="application/json", binary=False):
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.url + path, data=body, headers=headers, method=method)
        # Do not forward credentials to a redirected host or honor proxy env for local API.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=self.timeout) as response:
                data = response.read(256 * 1024 * 1024 + 1)
                if len(data) > 256 * 1024 * 1024:
                    raise HTTPError("Response exceeds 256 MiB")
        except urllib.error.HTTPError as error:
            detail = error.read(8192).decode("utf-8", "replace")
            raise HTTPError(f"TurboCider HTTP {error.code}: {detail}") from error
        except (OSError, urllib.error.URLError) as error:
            suffix = "; submission may have succeeded; inspect API jobs before resubmitting" if method == "POST" and path == "/v1/jobs" else ""
            raise HTTPError(f"TurboCider unavailable at {self.url}: {error}{suffix}") from error
        if binary:
            return data
        try:
            return json.loads(data)
        except ValueError as error:
            raise HTTPError("Invalid JSON response; inspect job history before retrying a submission") from error

    def catalog(self):
        return self.exchange("GET", "/v1/catalog")

    def upload(self, png):
        return self.exchange("POST", "/v1/assets", png, "image/png")["id"]

    def plan(self, installation_id, request):
        return self.exchange("POST", "/v1/plan", {"installation_id": installation_id, "request": request})

    def submit(self, installation_id, request):
        return self.exchange("POST", "/v1/jobs", {"installation_id": installation_id, "request": request})["id"]

    def status(self, job_id):
        return self.exchange("GET", "/v1/jobs/" + urllib.parse.quote(job_id, safe=""))

    def cancel(self, job_id):
        return self.exchange("POST", "/v1/jobs/" + urllib.parse.quote(job_id, safe="") + "/cancel", {})

    def artifact(self, job_id):
        return self.exchange("GET", "/v1/jobs/" + urllib.parse.quote(job_id, safe="") + "/artifact", binary=True)
