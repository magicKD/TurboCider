"""Pure topic/alias/version-review contracts, with no device or network access.

No source code is imported, no compatible version range is guessed, and even
exact recorded context matches never qualify a runtime or performance profile.
"""
from datetime import date
import re

FACETS = ("stage", "operator", "backend", "device", "symptom", "semantics", "representation", "boundary", "api")
CONTEXT = ("soc", "ram_gib", "os_build", "sdk", "compiler", "mlx", "mlx_source_commit", "runtime_build_id")
TERM = re.compile(r"[a-z][a-z0-9-]{0,63}")
PROTECTED = {"nax", "ane", "fp8", "q8", "gguf", "m4", "m4-pro", "m4-max", "m5", "m5-pro", "m5-max"}


def require(value, message):
    if not value:
        raise ValueError(message)


def strings(value, label):
    require(isinstance(value, list) and all(isinstance(v, str) and v and v == v.strip() for v in value),
            "expected nonempty string list: " + label)
    require(len(value) == len({v.casefold() for v in value}), "duplicate case-insensitive values: " + label)
    return value


def alias_map(canonical, rows, label):
    require(isinstance(rows, dict) and set(rows) <= set(canonical), "unknown alias target: " + label)
    result = {v.casefold(): v for v in canonical}
    for target, variants in rows.items():
        for variant in strings(variants, label):
            folded = variant.casefold()
            protected = re.sub(r"[\s_]+", "-", folded)
            sku = re.fullmatch(r"(?:apple-)?m[1-9][0-9]*(?:-?(?:pro|max|ultra))?", protected)
            require((protected not in PROTECTED and not sku) or folded == target,
                    "distinct resource/format/SKU cannot be an alias: " + variant)
            require(folded not in result or result[folded] == target, "ambiguous/chained alias: " + variant)
            result[folded] = target
    return result


def vocabulary(document):
    require(isinstance(document, dict) and document.get("schema") == "tc-retrieval-vocabulary-v1", "unknown retrieval vocabulary schema")
    dimensions = document.get("dimensions")
    require(isinstance(dimensions, dict) and set(dimensions) == set(FACETS), "exact retrieval dimensions required")
    for field, values in dimensions.items():
        require(strings(values, field) and all(TERM.fullmatch(v) for v in values), "canonical lowercase vocabulary required")
    aliases = document.get("aliases")
    require(isinstance(aliases, dict) and set(aliases) <= set(FACETS), "unknown alias dimension")
    maps = {field: alias_map(values, aliases.get(field, {}), field) for field, values in dimensions.items()}
    keywords = document.get("keywords")
    require(isinstance(keywords, dict) and all(isinstance(k, str) and TERM.fullmatch(k) for k in keywords), "canonical keyword map required")
    keyword_map = alias_map(list(keywords), keywords, "keywords")
    return {"dimensions": dimensions, "aliases": maps, "keywords": keyword_map}


def facets(value, schema, required=False):
    if value is None and not required:
        return {key: [] for key in FACETS}
    require(isinstance(value, dict) and set(value) == set(FACETS), "all topic facets must be explicitly classified (empty means unknown)")
    for field, values in value.items():
        require(set(strings(values, field)) <= set(schema["dimensions"][field]), "unknown canonical facet: " + field)
    return {key: list(value[key]) for key in FACETS}


def keyword_groups(text, aliases):
    """One-hop, longest-phrase matching; each canonical group scores once."""
    require(isinstance(text, str), "query text must be a string")
    text = text.casefold()
    groups = {}

    def add(term):
        canonical = aliases.get(term.casefold(), term.casefold())
        variants = [canonical] + [a for a, target in aliases.items() if target == canonical and a != canonical]
        groups.setdefault(canonical, variants)

    pattern = "|".join(re.escape(term) for term in sorted(aliases, key=lambda s: (-len(s), s)))
    position = 0
    for match in re.finditer(r"(?<![a-z0-9_])(?:" + pattern + r")(?![a-z0-9_])", text) if pattern else ():
        for token in re.findall(r"[\w-]+", text[position:match.start()]):
            add(token)
        add(match.group())
        position = match.end()
    for token in re.findall(r"[\w-]+", text[position:]):
        add(token)
    return [{"canonical": key, "variants": value} for key, value in groups.items()]


def context(value):
    require(isinstance(value, dict) and set(value) <= set(CONTEXT), "unknown recorded-context dimension")
    require(all(isinstance(v, str) and v == v.strip() and 0 < len(v) <= 256 for v in value.values()),
            "context values are bounded exact strings, not ranges, aliases or inferred versions")
    require(all(v.casefold() not in {"latest", "current", "unknown", "any", "default"}
                and not any(c in v for c in "<>*^~|\n\r\0") for v in value.values()),
            "exact recorded context required, not latest/unknown or a version range")
    return value


def iso_day(value):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), "explicit ISO date required")
    return date.fromisoformat(value)


def claim_rows(document, nodes):
    require(isinstance(document, dict) and document.get("schema") == "tc-version-review-v1", "unknown version review schema")
    rows = document.get("claims")
    require(isinstance(rows, list), "version claims must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"id", "statement", "applies_to", "bindings", "observed_context", "reviewed_at", "review_after_days"},
                "explicit version claim fields required")
        identity = row["id"]
        require(isinstance(identity, str) and re.fullmatch(r"claim-[a-z0-9-]+", identity) and identity not in result, "unique claim ID required")
        require(isinstance(row["statement"], str) and row["statement"].strip(), "claim statement required")
        applies = strings(row["applies_to"], "applies_to")
        require(applies and set(applies) <= set(nodes), "claim must refer to registered local page IDs")
        require(context(row["observed_context"]), "nonempty exact observed context required")
        iso_day(row["reviewed_at"])
        require(type(row["review_after_days"]) is int and 1 <= row["review_after_days"] <= 365, "bounded review interval required")
        bindings = row["bindings"]
        require(isinstance(bindings, dict) and bindings and set(bindings) <= set(nodes), "registered content bindings required")
        for bound, digest in bindings.items():
            require(nodes[bound]["kind"] in ("source", "evidence") and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                    "source/evidence SHA256 content binding required")
        result[identity] = dict(row)
    return result


def review(claim, observed=None, as_of=None):
    observed = context({} if observed is None else observed)
    day = iso_day(as_of) if as_of is not None else date.today()
    age = (day - iso_day(claim["reviewed_at"])).days
    expected = claim["observed_context"]
    changed = sorted(k for k, v in observed.items() if k in expected and v != expected[k])
    missing = sorted(set(expected) - set(observed))
    uncovered = sorted(set(observed) - set(expected))
    reasons = []
    if claim.get("changed_bindings"):
        reasons.append("source-or-evidence-content-changed")
    if changed:
        reasons.append("recorded-context-changed")
    if age < 0:
        reasons.append("review-date-is-in-the-future")
    elif age >= claim["review_after_days"]:
        reasons.append("review-window-expired")
    state = "needs-review" if reasons else "context-incomplete" if missing or uncovered else "recorded-context-only"
    return {"id": claim["id"], "statement": claim["statement"], "state": state,
            "as_of": day.isoformat(), "reviewed_at": claim["reviewed_at"], "age_days": age,
            "reasons": reasons, "changed_context_fields": changed, "missing_context_fields": missing,
            "uncovered_context_fields": uncovered, "changed_bindings": claim.get("changed_bindings", []),
            "observed_context": expected, "retrieval_grants_qualification": False}
