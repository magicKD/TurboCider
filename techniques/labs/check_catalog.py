#!/usr/bin/env python3
"""Audit per-technical-chapter implementation/case/control/revisit coverage."""
import json
from pathlib import Path

from common import emit
from run import CASES

LABS = Path(__file__).resolve().parent
WIKI = LABS.parent
CATEGORIES = ("hardware", "metal", "ane", "gpu-ane", "memory", "algorithms", "media", "profiling", "models", "designs")
DEVICE_CASES = {"metal:" + name for name in ("vector", "fusion", "gemm", "norm-rope", "attention", "graph", "overlap")}
DEVICE_CASES |= {"coreml:frozen-fp16", "coreml:runtime-fp16", "coreml:hybrid", "native:public-w8"}
DEVICE_CASES |= {"metal:counter-stage", "mlx:bf16-ffn-profile"}
DEVICE_CASES |= {"mlx:attention-replay"}
DEVICE_CASES |= {"private:pmu"}
DEVICE_CASES |= {"private:shared-surface"}
DEVICE_CASES |= {"mlx:external-adoption"}
DEVICE_CASES |= {"private:qwen-shared-ffn"}
REFERENCES = {"reference:" + name for name in ("split-ffn", "interval-peak", "model-header", "lab-summary")}


def audit(document=None, wiki=WIKI):
    if document is None:
        document = json.loads((wiki / "labs/catalog.json").read_text(encoding="utf-8"))
    expected = {str(page.relative_to(wiki)) for category in CATEGORIES for page in (wiki / category).glob("*.md")
                if page.name != "README.md"}
    errors, seen, used = [], set(), set()
    for entry in document["entries"]:
        chapter = entry["chapter"]
        if chapter in seen:
            errors.append("duplicate chapter: " + chapter)
        seen.add(chapter)
        if not (wiki / chapter).is_file():
            errors.append("missing chapter: " + chapter)
        elif "labs/" not in (wiki / chapter).read_text(encoding="utf-8"):
            errors.append("chapter does not link its implementation lab: " + chapter)
        for code in entry["code"]:
            path = (wiki / code).resolve()
            if not path.is_relative_to(wiki.resolve()) or not path.is_file():
                errors.append("missing/nonportable code: " + code)
        if not entry["cases"] or not entry["code"] or len(entry["controls"]) < 2 or not entry["revisit_when"].strip():
            errors.append("incomplete executable/control/revisit card: " + chapter)
        for case in entry["cases"]:
            if case not in set(CASES) | DEVICE_CASES | REFERENCES:
                errors.append("unknown case: " + case)
            used.add(case)
    for chapter in sorted(expected - seen):
        errors.append("technical chapter lacks implementation card: " + chapter)
    for case in sorted(set(CASES) - used):
        errors.append("CPU case not discoverable from any technical chapter: " + case)
    return {"technical_chapters": len(expected), "cards": len(seen), "cpu_cases": len(CASES),
            "device_case_selectors": len(DEVICE_CASES), "errors": errors,
            "scope": "local required code/chapters/selectors/controls/revisit; optional integration_code is not a dependency; not semantic or device qualification"}


if __name__ == "__main__":
    result = audit()
    emit(result)
    raise SystemExit(bool(result["errors"]))
