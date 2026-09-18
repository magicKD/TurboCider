"""Aggregate completed Z-Image stage profiles without treating serial timings as parallel timings."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(directory):
    report = json.loads((directory / "report.json").read_text())
    requests, blocks, phases, ended = {}, defaultdict(list), defaultdict(list), {}
    trace = directory / "blocks.jsonl"
    if trace.exists():
        # A subsequent request may already be appending while report.json still
        # describes the completed requests. Ignore only an unfinished last line.
        data = trace.read_text()
        lines = data.splitlines()
        if data and not data.endswith("\n"):
            lines = lines[:-1]
        for line in lines:
            row = json.loads(line)
            request = row["request"]
            if row["type"] == "request":
                requests[request] = row
            elif row["type"] == "block":
                blocks[request].append(row)
            elif row["type"] == "phase":
                phases[request].append(row)
            elif row["type"] == "request_end":
                ended[request] = row
    cases = defaultdict(list)
    records = []
    for run in report["runs"]:
        if run["status"]:
            continue
        timings = run["metrics"]["timings_seconds"]
        streaming = run["metrics"]["block_residency"]
        row = dict(name=run["name"], case=run["case"], warmup=run["warmup"],
                   wall_seconds=run["wall_seconds"], denoise_seconds=timings["denoise"],
                   decode_seconds=timings["vae_decode"],
                   read_bytes=streaming["request_bytes_loaded"],
                   read_wait_seconds=streaming["request_wait_seconds"],
                   swapin_bytes=run["system_vm_delta_bytes"]["Swapins"],
                   swapout_bytes=run["system_vm_delta_bytes"]["Swapouts"],
                   process_footprint_bytes=run["process_memory"]["phys_footprint_bytes"],
                   png_sha256=run["png_sha256"])
        if "hybrid_eager_segments" in run:
            row["hybrid_eager_segments"] = run["hybrid_eager_segments"]
        snapshots = run.get("phase_vm", {})
        row["phase_system_vm_delta_bytes"] = {}
        for name, begin, end in [("denoise", "denoise_begin", "denoise_end"),
                                 ("decode_and_postprocess", "denoise_end", "export_begin")]:
            if begin in snapshots and end in snapshots:
                row["phase_system_vm_delta_bytes"][name] = {
                    k: snapshots[end]["system_vm"][k] - snapshots[begin]["system_vm"][k]
                    for k in snapshots[begin]["system_vm"]}
        trace_id = run["trace_request"]
        if trace_id is not None:
            expected = 32 * run["request"]["steps"]
            assert requests[trace_id]["mode"] == run["profile_mode"]
            assert ended[trace_id]["completed"] and ended[trace_id]["blocks"] == expected
            assert len(blocks[trace_id]) == expected
            assert [b["index"] for b in blocks[trace_id]] == list(range(expected))
            main = [b for b in blocks[trace_id] if b["name"].startswith("layers.")]
            assert len(main) == 30 * run["request"]["steps"]
            seconds = [k for k in main[0] if k.endswith("_seconds")]
            row["main_block_mean_ms"] = {
                k.removesuffix("_seconds"): statistics.mean(b[k] for b in main) * 1000 for k in seconds}
            row["main_block_total_seconds"] = {k: sum(b[k] for b in main) for k in seconds}
            row["all_block_total_seconds"] = sum(b["total_seconds"] for b in blocks[trace_id])
            row["phase_memory"] = {p["phase"]: {k: p[k] for k in ["mlx_active_bytes", "mlx_peak_bytes"]}
                                   for p in phases[trace_id]}
            row["per_main_layer_mean_ms"] = {
                str(i): statistics.mean(b["total_seconds"] for b in main if b["name"] == f"layers.{i}") * 1000
                for i in range(30)}
        records.append(row)
        if not run["warmup"]:
            cases[run["case"]].append(row)
    summary = {}
    for case, runs in cases.items():
        row = {"sample_count": len(runs), "run_names": [r["name"] for r in runs]}
        for k in ["wall_seconds", "denoise_seconds", "decode_seconds", "read_bytes",
                  "read_wait_seconds", "process_footprint_bytes"]:
            row[k] = statistics.median(r[k] for r in runs)
        for k in ["swapin_bytes", "swapout_bytes"]:
            row[k] = sum(r[k] for r in runs)
        if "main_block_mean_ms" in runs[0]:
            row["main_block_mean_ms"] = {
                k: statistics.median(r["main_block_mean_ms"][k] for r in runs)
                for k in runs[0]["main_block_mean_ms"]}
        summary[case] = row
    return {"notes": report["notes"], "instrumented_output_parity": report.get("instrumented_output_parity"),
            "cases": summary, "runs": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for case, row in result["cases"].items():
        print(f"{case}: wall={row['wall_seconds']:.3f}s denoise={row['denoise_seconds']:.3f}s "
              f"wait={row['read_wait_seconds']:.6f}s swap-out={row['swapout_bytes'] / 2**20:.1f}MiB")
        if "main_block_mean_ms" in row:
            print("  " + json.dumps({k: round(v, 3) for k, v in row["main_block_mean_ms"].items()}))


if __name__ == "__main__":
    main()
