#!/usr/bin/env python3
"""Render a source index from canonical local-source metadata; read-only."""
import argparse
from pathlib import Path
import sys

from knowledge import ROOT, load


def render(index):
    lines = ["# 固定版本来源卡：源码事实与本地结果分开", "",
             "[总目录](../README.md) · [图与布局](../ane/coreml-graphs.md) · [研究工作台](../review/research-campaign-20261015.md)", "",
             "本页由`scripts/generate_source_index.py`从既有`references/sources.json`生成；章节通过catalog的`source_ids`复用同一登记。没有第二份手工章节清单。核心学习代码仍在`labs/`，来源卡不镜像整个上游仓库。", "",
             "| 来源 | 固定commit | 阅读/运行证据 |", "| --- | --- | --- |"]
    for node in index["nodes"]:
        if node["kind"] != "source" or "upstream" not in node:
            continue
        relative = Path(node["path"]).relative_to("sources").as_posix()
        upstream = node["upstream"]
        evidence = node.get("evidence", {})
        state = evidence.get("upstream_execution", evidence.get("runtime", "unknown"))
        label = upstream["repository"] + (" · " + upstream["topic"] if upstream.get("topic") else "")
        lines.append(f'| [{label}]({relative}) | `{upstream["commit"]}` | {state} |')
    official_nodes = [node for node in index["nodes"] if node["kind"] == "source" and "official" in node]
    if official_nodes:
        lines += ["", "## 官方API与SDK来源", "", "| 来源 | 固定读取身份 | 本地证据边界 |", "| --- | --- | --- |"]
        for node in official_nodes:
            relative = Path(node["path"]).relative_to("sources").as_posix()
            official = node["official"]
            lines.append(f'| [{official["publisher"]}]({relative}) | {official["snapshot_identity"]} | {node.get("evidence", {}).get("runtime", "unknown")} |')
    lines += ["", "源码存在、局部数值通过、消费者性能通过与真实请求通过是独立状态。原创MIL实验验证一个机制，不等于验收整个上游项目。各文件SHA、阅读范围和许可边界在来源卡与登记中保留。", "",
              "```sh", "python3 -B techniques/scripts/query.py 'K tiling' --type source --limit 3",
              "python3 -B techniques/scripts/get_page.py ane/coreml-graphs.md --follow-sources --max-bytes 32000",
              "python3 -B techniques/scripts/generate_source_index.py --check", "```", "",
              "来源卡是原创概括，不分发上游大段正文或代码。各项目及其第三方组件继续遵循原许可，不能用本目录许可覆盖。commit和文件SHA是实际阅读身份，活动网页仅供补充导航。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render(load(args.root))
    if args.check:
        if (args.root / "sources/README.md").read_text(encoding="utf-8") != content:
            print("source index differs from canonical metadata", file=sys.stderr)
            return 1
        print("source index is current")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
