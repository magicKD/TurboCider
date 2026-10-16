"""Read-only retrieval from the existing catalog and local source manifest.

Keep labs/catalog.json as the chapter inventory. Only scope=local source entries
are registered; product paths and upstream checkouts are never opened. Section
labels describe documentation, not device placement or hardware qualification.
"""
import hashlib
import json
from pathlib import Path
import re

from retrieval_contracts import FACETS, claim_rows, context, facets, iso_day, keyword_groups, review, vocabulary

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("section", *FACETS)
IDENTITY = re.compile(r"[a-z][a-z0-9-]{2,127}")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in " + path.name + ": " + key)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("expected an object in " + path.name)
    return value


def string_list(value, label):
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(value) != len(set(value))):
        raise ValueError("expected unique nonempty strings: " + label)
    return value


def local_file(root, relative):
    if (not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative):
        raise ValueError("a registered relative asset path is required")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping knowledge asset: " + relative)
    return path


def title(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError("page title missing: " + path.name)


def records(document, key):
    values = document.get(key)
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError("expected object list: " + key)
    return values


def binding_changes(root, bindings, nodes, digests):
    changed = []
    for identity, expected in bindings.items():
        if identity not in digests:
            value = hashlib.sha256()
            with local_file(root, nodes[identity]["path"]).open("rb") as stream:
                for part in iter(lambda: stream.read(1 << 20), b""):
                    value.update(part)
            digests[identity] = value.hexdigest()
        if expected != digests[identity]:
            changed.append(identity)
    return sorted(changed)


def load(root=ROOT):
    root = Path(root).resolve()
    catalog = read_json(local_file(root, "labs/catalog.json"))
    cards = records(catalog, "entries")
    sources = records(read_json(local_file(root, "references/sources.json")), "sources")
    schema_version = catalog.get("retrieval_schema_version")
    if schema_version is not None and (type(schema_version) is not int or schema_version != 1):
        raise ValueError("unknown retrieval schema version")
    schema = vocabulary(read_json(local_file(root, "data/vocabulary.json"))) if schema_version else {
        "dimensions": {key: [] for key in FACETS}, "aliases": {key: {} for key in FACETS}, "keywords": {}}
    nodes = []
    for card in cards:
        relative = card.get("chapter")
        path = local_file(root, relative)
        if path.suffix != ".md":
            raise ValueError("catalog chapter must be Markdown")
        identity = "technique-" + re.sub(r"[^a-z0-9]+", "-", relative[:-3].casefold()).strip("-")
        node = {"id": identity, "kind": "technique", "path": relative, "title": title(path),
                "section": [Path(relative).parts[0]],
                "sources": ["source-" + value for value in string_list(card.get("source_ids", []), "source_ids")],
                "evidence_pages": ["evidence-" + value for value in string_list(card.get("evidence_ids", []), "evidence_ids")]}
        for field in ("code", "cases", "controls"):
            node[field] = string_list(card.get(field), field)
        if not isinstance(card.get("revisit_when"), str) or not card["revisit_when"].strip():
            raise ValueError("missing revisit condition: " + relative)
        node["revisit_when"] = card["revisit_when"]
        node.update(facets(card.get("facets"), schema, required=bool(schema_version)))
        node["topic_metadata_declared"] = "facets" in card
        for code in node["code"]:
            local_file(root, code)
        nodes.append(node)
    for source in sources:
        if source.get("scope") != "local":
            continue
        identity = source.get("id")
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            raise ValueError("invalid local source ID")
        path = local_file(root, source.get("path"))
        heading = source.get("role", path.name)
        if not isinstance(heading, str) or not heading.strip():
            raise ValueError("invalid local source role: " + identity)
        kind = source.get("kind", "source")
        if kind not in ("source", "evidence"):
            raise ValueError("unknown registered source/evidence kind: " + identity)
        nodes.append({"id": kind + "-" + identity, "kind": kind, "path": source["path"],
                      "title": heading, "section": [Path(source["path"]).parts[0]],
                      **facets(source.get("facets"), schema, required="facets" in source),
                      "topic_metadata_declared": "facets" in source,
                      "symbols": string_list(source.get("symbols", []), "symbols"),
                      **{key: source[key] for key in ("upstream", "official", "license", "evidence") if key in source},
                      "code": string_list(source.get("code", []), "source code"),
                      "sources": ["source-" + value for value in string_list(source.get("source_ids", []), "source_ids")],
                      "evidence_pages": []})
    by_id, by_path = {}, {}
    for node in nodes:
        identity = node["id"]
        if not IDENTITY.fullmatch(identity) or identity in by_id:
            raise ValueError("invalid/duplicate knowledge ID: " + identity)
        if node["path"] in by_path:
            raise ValueError("duplicate registered knowledge asset: " + node["path"])
        by_id[identity] = node
        by_path[node["path"]] = identity
    for node in nodes:
        for code in node["code"]:
            local_file(root, code)
        inferred = [by_path[code] for code in node["code"]
                    if code in by_path and by_id[by_path[code]]["kind"] == "source" and by_path[code] != node["id"]]
        node["sources"] = list(dict.fromkeys([*node["sources"], *inferred]))
        for source in node["sources"]:
            if source not in by_id or by_id[source]["kind"] != "source":
                raise ValueError("unresolved local source card: " + source)
        for evidence in node["evidence_pages"]:
            if evidence not in by_id or by_id[evidence]["kind"] != "evidence":
                raise ValueError("unresolved local evidence page: " + evidence)
    claims = claim_rows(read_json(local_file(root, "data/version-claims.json")), by_id) if schema_version else {}
    digests = {}
    for claim in claims.values():
        claim["changed_bindings"] = binding_changes(root, claim["bindings"], by_id, digests)
    for node in nodes:
        node["version_claim_ids"] = [key for key, claim in claims.items() if node["id"] in claim["applies_to"]]
    values = {"kind": ["technique", "source", "evidence"], **schema["dimensions"],
              "section": sorted({section for node in nodes for section in node["section"]})}
    return {"root": root, "nodes": sorted(nodes, key=lambda item: item["id"]),
            "by_id": by_id, "by_path": by_path, "vocabulary": values, "aliases": schema["aliases"],
            "keyword_aliases": schema["keywords"], "claims": claims,
            "metadata_coverage": {"registered_pages": len(nodes), "topic_classified_pages": sum(n["topic_metadata_declared"] for n in nodes),
                                  "unclassified_by_field": {field: sum(not node[field] for node in nodes) for field in FACETS}}}


def present(index, node, observed=None, as_of=None):
    reviews = []
    digests = {}
    for identity in node["version_claim_ids"]:
        claim = dict(index["claims"][identity])
        claim["changed_bindings"] = binding_changes(index["root"], claim["bindings"], index["by_id"], digests)
        reviews.append(review(claim, observed, as_of))
    return {**node, "version_review": reviews,
            "retrieval_grants_qualification": False}


def query(index, text="", filters=None, limit=10, observed=None, as_of=None):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    if not isinstance(text, str) or (filters is not None and not isinstance(filters, dict)):
        raise ValueError("query text and filter object required")
    context({} if observed is None else observed)
    if as_of is not None:
        iso_day(as_of)
    normalized = {}
    for field, values in (filters or {}).items():
        if not isinstance(field, str) or field not in FIELDS + ("kind",):
            raise ValueError("unknown query dimension: " + str(field))
        choices = {value.casefold(): value for value in index["vocabulary"][field]}
        choices.update(index["aliases"].get(field, {}))
        normalized[field] = []
        for value in string_list(values, field):
            if value.casefold() not in choices:
                raise ValueError("unknown " + field + ": " + value + "; choose " + ", ".join(choices))
            normalized[field].append(choices[value.casefold()])
        if len(normalized[field]) != len(set(normalized[field])):
            raise ValueError("duplicate canonical filter via aliases: " + field)
    groups = keyword_groups(text, index["keyword_aliases"])
    output = []
    for node in index["nodes"]:
        if any(not set(values) <= ({node["kind"]} if field == "kind" else set(node[field]))
               for field, values in normalized.items()):
            continue
        heading = node["title"].casefold()
        metadata = " ".join([node["id"], *node["section"], *node.get("cases", []),
                             *node.get("controls", []), *node.get("symbols", []),
                             *(value for field in FACETS for value in node[field])]).casefold()
        body = local_file(index["root"], node["path"]).read_text(encoding="utf-8").casefold() if groups else ""
        score = sum(max((10 if token in heading else 0) + (5 if token in metadata else 0)
                        + (1 if token in body else 0) for token in group["variants"]) for group in groups)
        if groups and not score:
            continue
        output.append((score, node))
    output.sort(key=lambda item: (-item[0], item[1]["id"]))
    return {"query": text, "filters": normalized, "total_matches": len(output),
            "keyword_expansions": groups, "metadata_coverage": index["metadata_coverage"],
            "results": [present(index, node, observed, as_of) for _, node in output[:limit]],
            "retrieval_grants_qualification": False,
            "scope": "curated topic filters/one-hop aliases and recorded version-review hints, NOT applicability, compatibility, device or performance qualification"}


def get_page(index, lookup, follow_sources=False, include_code=False, maximum_bytes=24000, follow_evidence=False, offset_bytes=0, observed=None, as_of=None):
    if type(maximum_bytes) is not int or not 256 <= maximum_bytes <= 256000:
        raise ValueError("maximum bytes must be 256..256000")
    if type(offset_bytes) is not int or offset_bytes < 0:
        raise ValueError("byte offset must be a nonnegative integer")
    if not isinstance(lookup, str):
        raise ValueError("a registered page ID or path is required")
    context({} if observed is None else observed)
    if as_of is not None:
        iso_day(as_of)
    identity = lookup if lookup in index["by_id"] else index["by_path"].get(lookup)
    if identity is None:
        raise ValueError("unknown registered page: " + lookup)
    node = index["by_id"][identity]
    wanted = [node]
    if follow_sources:
        wanted += [index["by_id"][source] for source in node["sources"]]
    if follow_evidence:
        wanted += [index["by_id"][evidence] for evidence in node["evidence_pages"]]
    wanted = list({item["id"]: item for item in wanted}.values())
    remaining, pages, code, omitted_pages, omitted_code = maximum_bytes, [], [], [], []

    def expand(relative, offset=0):
        nonlocal remaining
        path = local_file(index["root"], relative)
        size = path.stat().st_size
        if offset > size:
            raise ValueError("byte offset exceeds the registered asset")
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(remaining + 1)
        selected = raw[:remaining]
        try:
            body = selected.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason != "unexpected end of data" or error.end != len(selected):
                raise ValueError("byte offset must start on a UTF-8 boundary") from error
            selected = selected[:error.start]
            body = selected.decode("utf-8")
        remaining -= len(selected)
        truncated = offset + len(selected) < size
        return {"path": relative, "body": body, "truncated": truncated,
                "byte_offset": offset, "total_file_bytes": size,
                "next_byte_offset": offset + len(selected) if truncated else None}

    for item in wanted:
        if remaining <= 0:
            omitted_pages.append(item["id"])
        else:
            pages.append({**expand(item["path"], offset_bytes if item["id"] == identity else 0),
                          "id": item["id"], "metadata": present(index, item, observed, as_of)})
    expanded = {item["path"] for item in pages}
    if include_code:
        for relative in node["code"]:
            if relative in expanded:
                continue
            if remaining <= 0:
                omitted_code.append(relative)
            else:
                code.append(expand(relative))
    return {"id": identity, "pages": pages, "code": code,
            "omitted_pages": omitted_pages, "omitted_code": omitted_code,
            "budget_bytes": maximum_bytes, "body_bytes_used": maximum_bytes - remaining,
            "scope": "registered local text only; no upstream checkout access, execution or network"}
