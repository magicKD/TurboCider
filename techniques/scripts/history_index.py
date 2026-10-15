#!/usr/bin/env python3
"""Index only relevant Codex session messages; never export tool payloads.

Default output contains metadata/keyword hit counts, not conversation text.
Use --excerpts only for a private local audit, never commit that report.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import re
import sys

KEYWORDS = {
    "handoff": r"IOSurface|outputBackings|shared.event|交接|zero.copy",
    "partition": r"channel.split|row.split|通道|分区|fork.join",
    "quantization": r"W8A8|ConvRot|Hadamard|量化|SmoothQuant",
    "scheduling": r"prefetch|lookahead|预取|并行|straggler",
    "memory": r"streaming|residency|驻留|峰值|swap|pread",
    "metal": r"SGMMA|SIMD|Metal.kernel|fusion|融合",
    "quality": r"relative.L2|parity|画质|finite|NaN",
}
PATTERNS = {key: re.compile(value, re.I) for key, value in KEYWORDS.items()}
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|(?:api[_-]?key|token|password)\s*[:=]\s*\S+)", re.I)


def messages(record):
    """Whitelist human/assistant prose; omit tools, system, and reasoning."""
    if not isinstance(record, dict):
        return []
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return []
    if record.get("type") != "response_item" or payload.get("type") != "message":
        return []
    role = payload.get("role")
    if role not in ("user", "assistant") or payload.get("channel") == "analysis":
        return []
    result = []
    for item in payload.get("content", []):
        if not isinstance(item, dict) or item.get("type") not in ("input_text", "output_text") or not isinstance(item.get("text"), str):
            continue
        content = item["text"]
        if any(marker in content for marker in ("<environment_context>", "<codex_internal_context", "Guardian review handoff", '"risk_level"', '"outcome":')):
            continue
        result.append((role, content))
    return result


def inspect(path, cwd_filters, excerpts=False, limit=3, match=None, tail_excerpts=False):
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        meta = json.loads(first)
        if meta.get("type") != "session_meta":
            return None
        info = meta.get("payload", {})
        cwd = info.get("cwd", "")
        # Match directory components, not a loose substring of unrelated paths.
        if not any(name in Path(cwd).parts for name in cwd_filters):
            return None
        result = {"session": path.name, "id": info.get("id"), "date": info.get("timestamp"),
                  "project": Path(cwd).name, "hits": {key: 0 for key in PATTERNS},
                  "matched_lines": [], "malformed_lines": 0}
        extracts = deque(maxlen=limit) if tail_excerpts and limit else []
        for line_number, line in enumerate(stream, 2):
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                result["malformed_lines"] += 1
                continue
            for role, content in messages(record):
                if match is not None and not match.search(content):
                    continue
                # Session context/goal dumps are not historical technical conclusions.
                if "<environment_context>" in content or "<codex_internal_context" in content:
                    continue
                hits = [key for key, pattern in PATTERNS.items() if pattern.search(content)]
                if not hits:
                    continue
                for key in hits:
                    result["hits"][key] += 1
                result["matched_lines"].append({"line": line_number, "role": role, "topics": hits})
                if excerpts and role == "assistant" and limit and (tail_excerpts or len(extracts) < limit):
                    # Best effort only: private excerpts can still contain sensitive prose.
                    extracts.append({"line": line_number, "text": SECRET.sub("[REDACTED]", content[:1800])})
        if excerpts:
            result["private_excerpts"] = list(extracts)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Explicit Codex sessions directory (read only)")
    parser.add_argument("--project", action="append", required=True, help="Exact cwd component; repeatable")
    parser.add_argument("--excerpts", action="store_true", help="PRIVATE LOCAL OUTPUT ONLY; prose may be sensitive")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--match", help="optional case-insensitive prose regex; no tool payloads searched")
    parser.add_argument("--tail-excerpts", action="store_true", help="keep last N matching assistant excerpts, still PRIVATE")
    args = parser.parse_args()
    if not args.root.is_dir() or args.limit < 0:
        parser.error("root must exist and limit must be nonnegative")
    try:
        pattern = re.compile(args.match, re.I) if args.match else None
    except re.error as error:
        parser.error("invalid match regex: " + str(error))
    results, failures = [], []
    for path in sorted(args.root.rglob("*.jsonl")):
        try:
            result = inspect(path, args.project, args.excerpts, args.limit, pattern, args.tail_excerpts)
            if result:
                results.append(result)
        except (OSError, ValueError, TypeError) as error:
            failures.append({"session": path.name, "error_type": type(error).__name__})
    print(json.dumps({"scope": "metadata and keyword index; not validated optimization evidence",
                      "sessions": results, "read_failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
