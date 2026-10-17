#!/usr/bin/env python3
"""Render a compact index directly from Markdown page metadata."""
import argparse
from pathlib import Path
import sys
from knowledge import ROOT, FACETS, load


def render(index):
    lines = ["# 知识索引：直接来自 Markdown", "",
             "[知识库](../README.md) · [使用与维护](../guides/knowledge-base.md) · [检索工具](../scripts/README.md)", "",
             "从页面标题、链接和短元数据生成，不使用 JSON catalog，不登记实验或验收状态。完整结果通过查询按需展开。", "",
             "## 内容类型", "", "| 类型 | 页数 | 查询 |", "| --- | --- | --- |"]
    for value in index["vocabulary"]["page_type"]:
        count = sum(value in node["page_type"] for node in index["nodes"])
        lines.append(f"| `{value}` | {count} | `--page-type {value}` |")
    for field in FACETS:
        lines += ["", "## " + field, "", "| 标签 | 页数 | 查询 |", "| --- | --- | --- |"]
        for value in index["vocabulary"][field]:
            count = sum(value in node[field] for node in index["nodes"])
            lines.append(f"| `{value}` | {count} | `--{field} {value}` |")
    return "\n".join([*lines, ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = render(load(args.root))
    if args.check:
        if (args.root / "queries/README.md").read_text(encoding="utf-8") != result:
            print("knowledge index differs from Markdown metadata", file=sys.stderr)
            return 1
        print("knowledge index is current")
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
