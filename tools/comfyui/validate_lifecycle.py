#!/usr/bin/env python3
"""Opt-in actual ComfyUI interruption, timeout and recovery. Existing FLUX weights only."""
import argparse
import json
from pathlib import Path
import sys
import time
from validate import exchange
from turbocider_http import HTTPClient


def until(check, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.2)
    raise TimeoutError("Lifecycle condition did not finish")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = HTTPClient(args.api_url)
    catalog = api.catalog()
    installation = next(i for i in catalog["installations"] if i["modelID"] == "flux2-klein-4b")
    queue = exchange(args.comfy_url, "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("Lifecycle tests require an idle ComfyUI queue")
    if api.exchange("GET", "/v1/health")["active_job"]:
        raise RuntimeError("Lifecycle tests require an idle TurboCider service")
    model = {"class_type": "TurboCiderModel", "inputs": {"installation": installation["id"], "api_url": args.api_url, "token_env": "TURBOCIDER_API_TOKEN"}}
    results = []
    for mode in ("interrupt", "timeout", "recovery"):
        graph = {"1": model}
        if mode == "recovery":
            graph["2"] = {"class_type": "TurboCiderRequest", "inputs": {"backend": ["1", 0], "timeout_seconds": 120,
                "request_json": json.dumps({"operation": "image.generate", "prompt": "A green ceramic teapot on a table", "width": 512, "height": 512, "steps": 4, "seed": 99})}}
        else:
            graph["2"] = {"class_type": "TurboCiderGenerate", "inputs": {"backend": ["1", 0], "prompt": "A blue ceramic teapot", "width": 1024, "height": 1024, "steps": 4, "seed": 71, "batch_size": 2,
                "advanced": "{}", "timeout_seconds": 1 if mode == "timeout" else 120}}
            graph["3"] = {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "TurboCider-lifecycle"}}
        old_ids = {j["id"] for j in api.exchange("GET", "/v1/jobs")["jobs"]}
        prompt_id = exchange(args.comfy_url, "/prompt", {"prompt": graph})["prompt_id"]
        def new_job():
            return next((j for j in api.exchange("GET", "/v1/jobs")["jobs"] if j["id"] not in old_ids), None)
        job = until(new_job)
        if mode == "interrupt":
            def running():
                state = api.status(job["id"])
                return state if state["state"] == "running" else None
            until(running)
            exchange(args.comfy_url, "/interrupt", {})
        history = until(lambda: exchange(args.comfy_url, "/history/" + prompt_id).get(prompt_id), 180)
        terminal = until(lambda: (j if (j := api.status(job["id"]))["state"] in ("succeeded", "failed", "cancelled", "interrupted") else None), 120)
        new_jobs = [j for j in api.exchange("GET", "/v1/jobs")["jobs"] if j["id"] not in old_ids]
        assert len(new_jobs) == 1, "Batch continued after interrupt/timeout"
        assert terminal["state"] == ("succeeded" if mode == "recovery" else "cancelled"), terminal
        assert history["status"]["status_str"] == ("success" if mode == "recovery" else "error"), history
        result = {"case": mode, "prompt_id": prompt_id, "native_job": terminal, "comfy_status": history["status"], "native_jobs_submitted": len(new_jobs)}
        results.append(result)
        (args.output_dir / (mode + ".json")).write_text(json.dumps(result, indent=2))
        print(mode, "PASS", job["id"], flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
