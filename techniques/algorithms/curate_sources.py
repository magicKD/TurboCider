#!/usr/bin/env python3
"""Generate a compact, fixed-commit source reading ledger (no network).

Full source stays in ignored checkouts. Only identity, digest, links and an
explicit human-supplied review scope are distributed with the wiki.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def collect(plan, root, local):
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    entries = []
    for spec in plan["sources"]:
        folder = local.get(spec["checkout"], root / spec["checkout"])
        def git(*args):
            return subprocess.run(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C",
                                   str(folder), *args], env=env, capture_output=True, check=True, timeout=20).stdout
        commit = git("rev-parse", "HEAD").decode().strip()
        files = []
        for path in spec["files"]:
            if path.startswith("/") or ".." in Path(path).parts:
                raise ValueError("unsafe source path")
            value = git("show", commit + ":" + path)
            url = "../../" + path if spec.get("local_history") else f"https://github.com/{spec['repository']}/blob/{commit}/{path}"
            files.append({"path": path, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(), "url": url})
        entries.append({key: value for key, value in spec.items() if key not in ("checkout", "files")}
                       | {"commit": commit, "files": files})
    return {"schema": "algorithm-source-readings-v1", "reviewed_at": "2026-10-16",
            "network_used_by_curator": False, "project_code_executed_by_curator": False,
            "scope": "Selected code sections / documentation only, never full repository audit or model reproduction.",
            "sources": entries}


def validate(value):
    if value.get("schema") != "algorithm-source-readings-v1" or not value.get("sources"):
        raise ValueError("invalid source ledger")
    ids = set()
    for source in value["sources"]:
        if source["id"] in ids or not re.fullmatch(r"S[0-9]{2}[a-z]?", source["id"]):
            raise ValueError("invalid/duplicate source id")
        ids.add(source["id"])
        if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise ValueError("missing immutable commit")
        if not source["read_scope"] or not source["files"]:
            raise ValueError("missing explicit reading scope")
        paths = set()
        for item in source["files"]:
            if item["path"] in paths or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("invalid source digest/path")
            paths.add(item["path"])
            expected = ("../../" + item["path"] if source.get("local_history")
                        else f"https://github.com/{source['repository']}/blob/{source['commit']}/{item['path']}")
            if item["url"] != expected or type(item["bytes"]) is not int or item["bytes"] < 0:
                raise ValueError("source URL/content identity mismatch")
    return value


def render(value):
    lines = ["# 算法调研来源：固定版本、阅读范围与证据等级", "",
             "[算法入口](README.md) · [机器可读清单](sources.json) · [评测合同](evaluation.md) · [配套实验](../labs/algorithm-research.md)", "",
             "核查日期：2026-10-16。来源以作者开源实现、论文、作者技术笔记和 Apple/框架官方资料为主。S 表示指定源码接缝/章节已读，D 表示只核对文档；不是完整仓库审计。下载候选文件与实际阅读分开，下面只列实际采用的阅读范围。", "",
             "`sources.json` 保存 commit、文件 SHA-256 与长度，不镜像第三方正文。摘要为原创概括，原代码和模型继续遵循各自许可。活动项目页只供导航，固定文件链接用于追溯。没有把网页标题中的 lossless 改写为数值无损，也不把作者测量冒充本轮本机复现。", "",
             "本轮网页搜索工具未返回可用结果；GitHub API 返回403。已转用获准的 Git 只读源码下载和 raw 文件端点，覆盖下列公开项目；因此不宣称穷尽全网或核实远端所有最新分支。远端快照与本地既有 checkout 分别固定，不能混称同一更新时点。", ""]
    for source in value["sources"]:
        anchor = source["id"].lower()
        repository = (source["repository"] + "（本地历史，可选产品溯源）" if source.get("local_history")
                      else f'[{source["repository"]}](https://github.com/{source["repository"]})')
        lines += [f'<a id="{anchor}"></a>', f'## {source["id"]} — {source["name"]}', "",
                  f'仓库：{repository}；commit：`{source["commit"]}`。', "",
                  f'阅读：{source["level"]}；{source["read_scope"]}', "",
                  "运行边界：" + source["execution_scope"], ""]
        for item in source["files"]:
            lines.append(f'- [{item["path"]}]({item["url"]})')
        for url in source.get("additional_read_urls", []):
            lines.append(f'- 补充已读原文：[固定文件/作者笔记]({url})（不在上述文件哈希范围内）')
        lines.append("")
    lines += ["## 重放来源取证", "",
              "[source-plan.json](source-plan.json)是候选采集范围，不是已审计/支持列表；[review-plan.json](review-plan.json)是本页采用的具体阅读文件。", "",
              "```sh", "python3 techniques/algorithms/source_probe.py --help",
              "python3 techniques/algorithms/curate_sources.py --check techniques/algorithms/sources.json", "```", "",
              "源码下载需显式 `--download`，输出放到新的本地忽略目录；不安装依赖、不运行项目脚本、不更新既有 HEAD。partial clone 的缺失 blob 仅在显式下载模式获取。来源收集没有 GPU 工作。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--local", action="append", default=[], help="checkout-id=existing/path; inspect only")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", type=Path, help="offline validation of an existing source manifest")
    args = parser.parse_args()
    if args.check:
        value = validate(json.loads(args.check.read_text()))
        print(json.dumps({"sources": len(value["sources"]), "files": sum(len(s["files"]) for s in value["sources"]),
                          "errors": [], "scope": "metadata structure only; not online URL availability or source semantics"}))
        return
    if not args.plan or not args.root or not args.output_dir:
        parser.error("--plan, --root and --output-dir required")
    outputs = [args.output_dir / name for name in ("sources.json", "sources.md")]
    if any(path.exists() or path.is_symlink() for path in outputs):
        parser.error("generated source outputs must be new")
    local = dict(item.split("=", 1) for item in args.local)
    value = validate(collect(json.loads(args.plan.read_text()), args.root, {k: Path(v) for k, v in local.items()}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    outputs[1].write_text(render(value))
    print(json.dumps({"sources": len(value["sources"]), "files": sum(len(s["files"]) for s in value["sources"])}))


if __name__ == "__main__":
    main()
