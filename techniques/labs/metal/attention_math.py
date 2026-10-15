"""Protected CPU attention math and bounded campaign geometry; no frameworks."""
import math
import statistics


def row_oracle(query, keys, values, allowed, scale):
    if (not query or len(keys)!=len(values) or len(keys)!=len(allowed) or not keys or not values[0]
            or any(len(row)!=len(query) for row in keys) or any(len(row)!=len(values[0]) for row in values)
            or any(type(value) is not bool for value in allowed) or not math.isfinite(scale) or scale<=0):
        raise ValueError("finite positive scale and matched attention row geometry required")
    scores=[scale*math.fsum(float(a)*float(b) for a,b in zip(query,key)) if keep else None
            for key,keep in zip(keys,allowed)]
    selected=[value for value in scores if value is not None]
    if not selected:
        return [0.]*len(values[0])
    if not all(math.isfinite(value) for value in selected):
        raise ValueError("nonfinite CPU attention score")
    peak=max(selected)
    probabilities=[math.exp(value-peak) if value is not None else 0. for value in scores]
    denominator=math.fsum(probabilities)
    return [math.fsum(probability*float(row[column]) for probability,row in zip(probabilities,values))/denominator
            for column in range(len(values[0]))]


def validate_segments(segments, rows):
    cursor=0
    for segment in segments:
        if (type(segment.get("start")) is not int or type(segment.get("end")) is not int
                or type(segment.get("causal")) is not bool or segment["start"]!=cursor
                or not cursor<segment["end"]<=rows):
            raise ValueError("ordered complete segment partition required")
        cursor=segment["end"]
    if cursor!=rows:
        raise ValueError("segments must cover every query row")


def allowed_keys(query, queries, keys, mode, segments=()):
    if mode=="segments":
        validate_segments(segments,queries)
        segment=next(item for item in segments if item["start"]<=query<item["end"])
        return [column<segment["end"] and (not segment["causal"] or column<=query) for column in range(keys)]
    if mode=="causal":
        return [column<=query+keys-queries for column in range(keys)]
    if mode=="empty-first-row":
        return [query!=0]*keys
    if mode=="none":
        return [True]*keys
    raise ValueError("unknown mask contract")


def paired(samples):
    measured=[sample for sample in samples if not sample["warmup"]]
    if len(measured)!=8 or [row["trial"] for row in measured]!=list(range(8)):
        raise ValueError("eight complete ordered ABBA measured samples required")
    pairs={}
    for row in samples:
        if not row.get("component_numeric_pass"):
            return None
    for row in measured:
        if type(row["wall_ns"]) is not int or row["wall_ns"]<=0 or row["pair"]!=row["trial"]//2:
            raise ValueError("positive finite matching trial boundary required")
        arm=pairs.setdefault(row["pair"],{})
        if row["variant"] in arm:
            raise ValueError("duplicate paired arm")
        arm[row["variant"]]=row
    ratios=[]
    for arms in pairs.values():
        if set(arms)!={"baseline","candidate"} or arms["baseline"]["point"]!=arms["candidate"]["point"]:
            raise ValueError("missing or mismatched attention pair")
        ratios.append(arms["baseline"]["wall_ns"]/arms["candidate"]["wall_ns"])
    return {"source_kind":"derived","pairs":len(ratios),"paired_ratios":ratios,
            "median_paired_host_ratio":statistics.median(ratios),"production_qualified":False,
            "scope":"fixed-point prepared operator host boundary; small-n exploratory, not request speedup"}
