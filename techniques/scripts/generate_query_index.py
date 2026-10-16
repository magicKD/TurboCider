#!/usr/bin/env python3
"""Render topic facets from the canonical catalog/source registry; read-only."""
import argparse
from pathlib import Path
import sys

from knowledge import ROOT, load
from retrieval_contracts import FACETS


def render(index):
    lines = ["# 受控主题索引：从单一登记生成", "",
             "[总目录](../README.md) · [字段与别名合同](../data/README.md) · [查询入口](../scripts/README.md)", "",
             "由`scripts/generate_query_index.py`读取现有catalog/source登记生成，不维护第二份章节清单。标签只描述页面主题，不表达设备支持、dtype兼容或性能资格。空标签是未分类，不是通用适用。来源/证据不继承关联章节的标签。", "",
             "每行可用`python3 -B techniques/scripts/query.py --type technique --FIELD VALUE`查询；同维度重复值及跨维度均为AND。版本复查状态在实际查询/展开时计算，不把日期相关结果烘焙进静态索引。", ""]
    techniques = [n for n in index["nodes"] if n["kind"] == "technique"]
    for field in FACETS:
        lines += ["## " + field, "", "| 标签 | 已分类技术页 |", "| --- | --- |"]
        for value in index["vocabulary"][field]:
            links = [f'[{node["title"]}](../{node["path"]})' for node in techniques if value in node[field]]
            lines.append("| `" + value + "` | " + (" · ".join(links) or "尚无已分类技术页") + " |")
        unknown = [n for n in techniques if not n[field]]
        lines += ["", f"未分类技术页：{len(unknown)}；未分类项不会命中该维度的过滤。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render(load(args.root))
    if args.check:
        if (args.root / "queries/README.md").read_text(encoding="utf-8") != content:
            print("query index differs from canonical metadata", file=sys.stderr)
            return 1
        print("query index is current")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
