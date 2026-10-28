"""Markdown-native, read-only retrieval. No experiment artifacts or catalogs."""
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FACETS = ("stage", "operator", "backend", "device", "symptom", "semantics", "representation", "boundary", "api")
FIELDS = ("section", "page_type", *FACETS)
PAGE_TYPES = ("method", "implementation", "case-study", "guide", "navigation", "source")
INTENTS = ("learn", "implement", "sources", "all", "evidence")
IDENTITY = re.compile(r"[a-z][a-z0-9-]{2,127}")
LINK = re.compile(r"!?\[([^\]]*)\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")
META = re.compile(r"<!-- knowledge\n(.*?)\n-->", re.S)
KEYWORDS = {
    "ffn": ("前馈网络", "feed forward network", "feed-forward network"),
    "attention": ("注意力", "scaled dot product attention", "sdpa"),
    "coreml": ("Core ML",), "ane": ("Neural Engine", "神经网络引擎"),
    "iosurface": ("IO Surface",), "lora": ("低秩适配",),
    "bf16": ("bfloat16",), "fp16": ("float16",), "fp32": ("float32",),
    "nax": ("GPU Neural Accelerators",),
}
ALIASES = {
    "stage": {"前馈网络": "ffn", "扩散变换器": "dit", "编码器": "encoder", "变分自编码器": "vae"},
    "operator": {"矩阵乘法": "gemm", "注意力": "attention", "sdpa": "attention", "swi-glu": "swiglu"},
    "backend": {"core ml": "coreml", "私有ane": "private-ane", "cpu参考": "cpu-reference"},
    "device": {"图形处理器": "gpu", "neural engine": "ane", "神经网络引擎": "ane"},
    "symptom": {"带宽受限": "memory-bound", "提交开销": "launch-bound", "布局错误": "wrong-layout", "过期缓存": "stale-cache"},
    "semantics": {"数值等价": "numerically-equivalent", "近似": "approximate", "仅合同": "contract-only"},
    "representation": {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"},
    "boundary": {"gpu→ane": "gpu-ane", "cpu→gpu": "cpu-gpu"},
    "api": {"公开": "public", "私有": "private"},
}


def local_file(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts or Path(relative).as_posix() != relative):
        raise ValueError("a relative knowledge asset path is required")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping knowledge asset: " + relative)
    return path


def strings(values, label):
    if (not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values)
            or len(values) != len(set(values))):
        raise ValueError("unique nonempty strings required: " + label)
    return values


def page_type(relative):
    parts = Path(relative).parts
    if parts[-1] == "README.md" or relative == "MAINTENANCE.md":
        return "navigation"
    if "case-studies" in parts:
        return "case-study"
    if parts[0] == "sources" or parts[-1] == "sources.md":
        return "source"
    if parts[0] in ("guides", "profiling", "models", "templates") or relative == "AGENTS.md":
        return "guide"
    return "method"


def metadata(content, relative):
    blocks = META.findall(content.partition("\n## ")[0][:4096])
    if len(blocks) > 1:
        raise ValueError("duplicate knowledge metadata: " + relative)
    data = {}
    if blocks:
        for line in blocks[0].splitlines():
            if not line.strip():
                continue
            key, separator, value = line.partition(":")
            if not separator or key not in ("id", "type", "aliases", *FACETS) or key in data:
                raise ValueError("unknown or duplicate Markdown metadata: " + relative)
            data[key] = value.strip()
    identity = data.get("id", "technique-" + re.sub(r"[^a-z0-9]+", "-", relative[:-3].casefold()).strip("-"))
    category = data.get("type", page_type(relative))
    if not IDENTITY.fullmatch(identity) or category not in PAGE_TYPES:
        raise ValueError("invalid Markdown knowledge identity/type: " + relative)
    result = {"id": identity, "page_type": [category]}
    for key in ("aliases", *FACETS):
        result[key] = strings([v.strip() for v in data.get(key, "").split(",") if v.strip()], key)
    return result


def linked_assets(root, relative, content):
    pages, code, external = [], [], []
    for label, raw in LINK.findall(content):
        target = urlsplit(raw.strip("<>"))
        if target.scheme in ("http", "https"):
            if target.geturl() not in [item["url"] for item in external]:
                external.append({"title": label, "url": target.geturl()})
            continue
        if target.scheme or target.netloc or not target.path:
            continue
        destination = ((root / relative).parent / unquote(target.path)).resolve()
        if not destination.is_relative_to(root):
            continue  # Optional product/other-repository links are not opened.
        path = destination.relative_to(root).as_posix()
        if destination.suffix == ".md" and destination.is_file():
            pages.append(path)
        elif path.startswith("examples/") and destination.suffix in (".py", ".metal") and destination.is_file():
            code.append(path)
    return list(dict.fromkeys(pages)), list(dict.fromkeys(code)), external


def summary(content):
    for line in META.sub("", content).splitlines()[1:]:
        if line.strip() and not line.startswith(("#", "[", "|", "```", "<!--", ">")):
            return line.strip()[:400]
    return ""


def load(root=ROOT):
    root = Path(root).resolve()
    nodes, links = [], {}
    for page in sorted(root.rglob("*.md")):
        relative = page.relative_to(root).as_posix()
        content = local_file(root, relative).read_text(encoding="utf-8")
        lines = content.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError("page title missing: " + relative)
        node = metadata(content, relative)
        node.update(path=relative, title=lines[0][2:].strip(),
                    kind="source" if node["page_type"] == ["source"] else "technique",
                    section=[Path(relative).parts[0]], summary=summary(content),
                    topic_metadata_declared=bool(META.search(content)))
        links[relative], node["code"], node["external_sources"] = linked_assets(root, relative, content)
        nodes.append(node)
    by_id, by_path, aliases = {}, {}, {}
    for node in nodes:
        if node["id"] in by_id or node["path"] in by_path:
            raise ValueError("duplicate knowledge identity/path: " + node["id"])
        by_id[node["id"]], by_path[node["path"]] = node, node["id"]
    for node in nodes:
        for alias in node["aliases"]:
            if not IDENTITY.fullmatch(alias) or alias in by_id or alias in aliases:
                raise ValueError("duplicate or invalid knowledge alias: " + alias)
            aliases[alias] = node["id"]
        node["sources"] = [by_path[path] for path in links[node["path"]]
                           if path in by_path and by_id[by_path[path]]["kind"] == "source" and path != node["path"]]
    vocabulary = {"kind": ["technique", "source"], "page_type": list(PAGE_TYPES),
                  "section": sorted({value for node in nodes for value in node["section"]}),
                  **{field: sorted({value for node in nodes for value in node[field]}) for field in FACETS}}
    return {"root": root, "nodes": nodes, "by_id": by_id, "by_path": by_path, "aliases": aliases,
            "vocabulary": vocabulary,
            "metadata_coverage": {"registered_pages": len(nodes),
                "unclassified_by_field": {field: sum(not n[field] for n in nodes) for field in FACETS}}}


def keyword_groups(text):
    text = text.casefold()
    terms = {key: key for key in KEYWORDS}
    terms.update({alias.casefold(): key for key, values in KEYWORDS.items() for alias in values})
    groups = {}
    def add(token):
        canonical = terms.get(token.casefold(), token.casefold())
        groups.setdefault(canonical, [canonical, *(a.casefold() for a in KEYWORDS.get(canonical, ()))])
    pattern = "|".join(re.escape(value) for value in sorted(terms, key=lambda v: (-len(v), v)))
    position = 0
    for match in re.finditer(r"(?<![a-z0-9_])(?:" + pattern + r")(?![a-z0-9_])", text.casefold()):
        for token in re.findall(r"[\w-]+", text[position:match.start()]):
            add(token)
        add(match.group())
        position = match.end()
    for token in re.findall(r"[\w-]+", text[position:]):
        add(token)
    return [{"canonical": key, "variants": list(dict.fromkeys(values))} for key, values in groups.items()]


def present(index, node):
    return {**node, "retrieval_grants_qualification": False}


def query(index, text="", filters=None, limit=10, intent="learn"):
    if type(limit) is not int or not 1 <= limit <= 100 or not isinstance(text, str):
        raise ValueError("text and limit 1..100 required")
    if intent not in INTENTS:
        raise ValueError("unknown reading intent: " + str(intent))
    warnings = []
    if intent == "evidence":
        intent = "sources"
        warnings.append("Experiment evidence is not distributed; returning source-reading references, not verification receipts.")
    filters = {} if filters is None else filters
    if not isinstance(filters, dict):
        raise ValueError("query filters must be an object")
    normalized = {}
    for field, values in filters.items():
        if field not in ("kind", *FIELDS):
            raise ValueError("unknown query dimension: " + str(field))
        choices = {value.casefold(): value for value in index["vocabulary"][field]}
        aliases = ALIASES.get(field, {})
        normalized[field] = []
        for value in strings(values, field):
            canonical = aliases.get(value.casefold(), value.casefold())
            if canonical not in choices:
                raise ValueError("unknown " + field + ": " + value)
            normalized[field].append(choices[canonical])
        strings(normalized[field], field)
    groups = keyword_groups(text)
    priority = {
        "learn": {"method": 24, "implementation": 16, "guide": 12, "navigation": 8, "case-study": 4},
        "implement": {"implementation": 24, "case-study": 20, "method": 8, "guide": 6, "source": 4},
        "sources": {"source": 24, "case-study": 4}, "all": {}}[intent]
    results = []
    for node in index["nodes"]:
        if any(not set(values) <= ({node["kind"]} if field == "kind" else set(node[field]))
               for field, values in normalized.items()):
            continue
        heading = node["title"].casefold()
        tags = " ".join([node["id"], *node["section"], *(v for field in FACETS for v in node[field])]).casefold()
        body = local_file(index["root"], node["path"]).read_text(encoding="utf-8").casefold() if groups else ""
        score = sum(max((10 if token in heading else 0) + (5 if token in tags else 0) + (1 if token in body else 0)
                        for token in group["variants"]) for group in groups)
        if groups and not score:
            continue
        strong = not groups or any(token in heading for group in groups for token in group["variants"])
        bonus = priority.get(node["page_type"][0], 0)
        results.append((score + (bonus if strong else min(bonus, 3)), score, node))
    results.sort(key=lambda row: (-row[0], -row[1], row[2]["id"]))
    return {"query": text, "intent": intent, "filters": normalized, "total_matches": len(results),
            "keyword_expansions": groups, "metadata_coverage": index["metadata_coverage"],
            "results": [present(index, node) for _, _, node in results[:limit]], "warnings": warnings,
            "retrieval_grants_qualification": False,
            "scope": "Knowledge notes and source references only; no packaged experiment data or verification status."}


def get_page(index, lookup, follow_sources=False, include_code=False, maximum_bytes=24000, offset_bytes=0):
    if type(maximum_bytes) is not int or not 256 <= maximum_bytes <= 256000:
        raise ValueError("maximum bytes must be 256..256000")
    if type(offset_bytes) is not int or offset_bytes < 0 or not isinstance(lookup, str):
        raise ValueError("a registered lookup and nonnegative byte offset are required")
    identity = lookup if lookup in index["by_id"] else index["aliases"].get(lookup, index["by_path"].get(lookup))
    if identity is None:
        raise ValueError("unknown knowledge page: " + lookup)
    node = index["by_id"][identity]
    wanted = [identity, *(node["sources"] if follow_sources else [])]
    remaining, pages, code, omitted_pages, omitted_code = maximum_bytes, [], [], [], []
    def expand(relative, offset=0):
        nonlocal remaining
        raw = local_file(index["root"], relative).read_bytes()
        if offset > len(raw):
            raise ValueError("byte offset exceeds the asset")
        raw[:offset].decode("utf-8")  # Refuse offsets inside a UTF-8 character.
        selected = raw[offset:offset + remaining]
        while selected:
            try:
                body = selected.decode("utf-8")
                break
            except UnicodeDecodeError as error:
                if error.start < len(selected) - 4:
                    raise
                selected = selected[:error.start]
        else:
            body = ""
        remaining -= len(selected)
        next_offset = offset + len(selected)
        return {"path": relative, "body": body, "truncated": next_offset < len(raw),
                "next_byte_offset": next_offset if next_offset < len(raw) else None}
    for page_id in dict.fromkeys(wanted):
        if remaining <= 0:
            omitted_pages.append(page_id)
            continue
        item = index["by_id"][page_id]
        pages.append({"id": page_id, "metadata": present(index, item),
                      **expand(item["path"], offset_bytes if page_id == identity else 0)})
    if include_code:
        for relative in node["code"]:
            if remaining <= 0:
                omitted_code.append(relative)
            else:
                code.append(expand(relative))
    return {"pages": pages, "code": code, "body_bytes_used": maximum_bytes - remaining,
            "omitted_pages": omitted_pages, "omitted_code": omitted_code,
            "external_sources": node["external_sources"], "retrieval_grants_qualification": False,
            "scope": "Read-only Markdown and small teaching examples; no experiment verification."}
