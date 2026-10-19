#!/usr/bin/env python3
"""CPU audit of r256 / r128 / r256 bracketed six-step GPU cohorts.

Rank changes the adapter/model update. Pixel differences are observations,
never an equivalence certificate or a base40 speed denominator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from flux9_bf16_framework_screen import write_json
from qwen21_dit_cache_report import image_metrics
from qwen_student_prefix_screen import ADAPTER_PINS, validate
from runtime_ane_common import sha256_file


def comparable_request(request):
    result = {key: value for key, value in request.items() if key != "output"}
    result["loras"] = [{key: value for key, value in adapter.items() if key != "path"}
                       for adapter in result["loras"]]
    return result


def audit(before, candidate, after):
    import numpy as np
    from PIL import Image
    roots = dict(before=before, candidate=candidate, after=after)
    summaries, reports = {}, {}
    for label, root in roots.items():
        summaries[label] = json.loads((root / "summary.json").read_text())
        reports[label] = json.loads((root / "joint/report.json").read_text())
        s = summaries[label]
        rank = "r128" if label == "candidate" else "r256"
        expected = next(p["sha256"] for p in ADAPTER_PINS.values() if p["rank"] == rank)
        if s["status"] != "complete" or s["adapter_sha256"] != expected:
            raise ValueError("complete pinned rank cohort required: " + label)
        validate(reports[label], 73, "joint")
    for key in ("library_sha256", "reference_sha256", "reference_size"):
        if len({s[key] for s in summaries.values()}) != 1:
            raise ValueError("cohort identity differs: " + key)

    def pixels(label, index):
        with Image.open(roots[label] / "joint" / f"{index}.png") as im:
            im.load()
            if im.size != (512, 512):
                raise ValueError("complete student512 output required")
            return np.asarray(im.convert("RGBA"))

    control_hashes = []
    for i in range(11):
        reference = comparable_request(json.loads((before / "joint" / f"request-{i}.json").read_text()))
        for label, root in roots.items():
            r = comparable_request(json.loads((root / "joint" / f"request-{i}.json").read_text()))
            if r != reference:
                raise ValueError("request changed beyond declared adapter path/output")
        a, b = pixels("before", i), pixels("after", i)
        if not np.array_equal(a, b):
            raise ValueError("r256 bracket output changed; no rank-only attribution")
        control_hashes.append(hashlib.sha256(a.tobytes()).hexdigest())

    def wall(label, i):
        return reports[label]["runs"][i]["request_wall_including_export"]

    def timing(indices):
        values = {label: [wall(label, i) for i in indices] for label in roots}
        medians = {label: statistics.median(rows) for label, rows in values.items()}
        return dict(samples=values, medians=medians,
            reduction_vs_before_percent=100 * (1 - medians["candidate"] / medians["before"]),
            reduction_vs_after_percent=100 * (1 - medians["candidate"] / medians["after"]))

    quality = []
    for i in (1, 2, 5, 7, 8, 9):
        a, b = pixels("before", i), pixels("candidate", i)
        r = reports["candidate"]["runs"][i]["metrics"]
        quality.append(dict(index=i, operation=r["operation"], seed=r["seed"],
            snapshot_hit="edit prefix KV snapshot hit" in r["acceleration_selection"],
            rgb=image_metrics(a[:, :, :3], b[:, :, :3]),
            alpha_max_abs=int(np.abs(a[:, :, 3].astype(int) - b[:, :, 3]).max())))
    return dict(schema="tc-qwen-student-rank-comparison-v1", qualification_passed=False,
        scope="pinned r256/r128 are different adapters; same joint arithmetic/six-step task; process brackets, not interleaved ABBA or teacher40 equivalence",
        library_sha256=summaries["before"]["library_sha256"],
        source_summary_sha256={label: sha256_file(root / "summary.json") for label, root in roots.items()},
        adapter_sha256={label: s["adapter_sha256"] for label, s in summaries.items()},
        all_eleven_r256_bracket_rgba_equal=True, r256_control_rgba_sha256=control_hashes,
        generation=timing((1, 2, 3)), edit_same_seed=timing((7,)), edit_new_seed=timing((8,)),
        quality=quality)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.before.resolve(), args.candidate.resolve(), args.after.resolve())
    if args.output.exists():
        p.error("choose fresh rank-audit output; raw evidence is not overwritten")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
