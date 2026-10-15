"""Capability/effect composition and bounded two-point calibration references."""
import math

try:
    from .common import report
except ImportError:
    from common import report


def compose(items, capabilities, horizon=12):
    if type(horizon) is not int or not 1 <= horizon <= 4096:
        raise ValueError("bounded explicit schedule horizon required")
    exclusive = {"attention_backend", "ffn_precision", "token_set", "step_output"}
    ids = set()
    active = []
    for item in items:
        if item["id"] in ids:
            raise ValueError("duplicate technique id")
        ids.add(item["id"])
        if not set(item["requires"]) <= set(capabilities):
            raise ValueError("missing model capability for " + item["id"])
        steps = set(item["steps"])
        if any(type(step) is not int or not 0 <= step < horizon for step in steps):
            raise ValueError("schedule step outside horizon")
        if steps:
            active.append(item)
    for index, a in enumerate(active):
        for b in active[index + 1:]:
            if not set(a["steps"]) & set(b["steps"]):
                continue
            aw, bw = set(a["writes"]), set(b["writes"])
            if aw & bw & exclusive:
                raise ValueError("exclusive seam conflict: " + a["id"] + "/" + b["id"])
            dependency = (aw & set(b["reads"])) | (bw & set(a["reads"]))
            if a["phase"] == b["phase"] and dependency:
                raise ValueError("same-phase ordering is ambiguous")
    return sorted(active, key=lambda item: (item["phase"], item["id"]))


def composition(seed=17):
    caps = ["blocks", "ffn", "steps", "attention"]
    ffn = {"id": "channel-ffn", "requires": ["ffn"], "phase": 30,
           "reads": ["hidden"], "writes": ["ffn_precision"], "steps": list(range(2, 10))}
    cache = {"id": "residual-cache", "requires": ["blocks", "steps"], "phase": 50,
             "reads": ["hidden"], "writes": ["step_output"], "steps": [4, 6, 8]}
    allowed = compose([cache, ffn], caps)
    rejected = []
    for name, items, model_caps in (
        ("two-ffn-precision-owners", [ffn, {**ffn, "id": "other-quantizer"}], caps),
        ("missing-ffn-seam", [ffn], ["blocks"]),
        ("same-phase-hidden-hazard", [ffn, {"id": "hidden-writer", "requires": [], "phase": 30,
         "reads": [], "writes": ["hidden"], "steps": [3]}], caps)):
        try:
            compose(items, model_caps)
        except ValueError as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise AssertionError("invalid composition accepted")
    off = {**ffn, "steps": []}
    assert compose([off], caps) == []
    # Mutually exclusive step ranges can share a seam, with explicit schedules.
    split_schedule = compose([{**ffn, "steps": [0]}, {**ffn, "id": "gpu-precision", "steps": [1]}], caps)
    return report("technique-composition", {"ordered": [item["id"] for item in allowed], "rejected": rejected,
                  "off_identity": True, "disjoint_schedules_accepted": len(split_schedule) == 2,
                  "limits": "Structural seam checks only. Numerics, lifetime and real-model combination still require experiments."})


def interpolate(point_a, point_b, share, key):
    a, b = point_a["share"], point_b["share"]
    if not a <= share <= b or a == b:
        raise ValueError("outside measured share interval")
    return point_a[key] + (point_b[key] - point_a[key]) * (share - a) / (b - a)


def select_share(points, full_gpu, widths, maximum_width, min_gain=.05):
    """Transparent conservative interpolation, not the production polygon fit."""
    if len(points) != 2 or not math.isfinite(full_gpu) or full_gpu <= 0 or not 0 <= min_gain < 1:
        raise ValueError("two measured points and a positive optimized GPU baseline required")
    points = sorted(points, key=lambda point: point["share"])
    if not 0 < points[0]["share"] < points[1]["share"] < 1:
        raise ValueError("ordered interior shares required")
    for point in points:
        if any(not math.isfinite(point[key]) or point[key] <= 0 for key in ("gpu", "ane", "both")):
            raise ValueError("positive independent spans required")
        if point["both"] < max(point["gpu"], point["ane"]) * .97:
            raise ValueError("both is not a complete independent concurrency boundary")
    choices = []
    for channels, full_width in widths:
        if channels > maximum_width:
            continue
        share = channels / full_width
        if not points[0]["share"] <= share <= points[1]["share"]:
            continue
        g, a, measured_both = [interpolate(*points, share, key) for key in ("gpu", "ane", "both")]
        prediction = max(g, a, measured_both)
        if prediction <= full_gpu * (1 - min_gain):
            choices.append({"channels": channels, "share": share, "seconds": prediction})
    if not choices:
        return {"selected": None, "reason": "no measured-range memory-admitted gain"}
    best = min(row["seconds"] for row in choices)
    selected = min((row for row in choices if row["seconds"] <= best * 1.01), key=lambda row: row["channels"])
    return {"selected": selected, "candidates": choices, "requires_independent_trial": True}


def calibration(seed=17):
    points = [{"share": .25, "gpu": .008, "ane": .003, "both": .009},
              {"share": .75, "gpu": .003, "ane": .008, "both": .0095}]
    choices = [(channels, 4096) for channels in range(512, 4096, 512)]
    selected = select_share(points, .012, choices, 2048)
    declined = select_share(points, .007, choices, 2048)
    assert selected["selected"] and declined["selected"] is None
    try:
        interpolate(*points, .9, "gpu")
    except ValueError:
        outside_rejected = True
    else:
        outside_rejected = False
    return report("share-calibration", {"measured_points": points, "proposal": selected,
                  "faster_gpu_declines": declined, "outside_measured_range_rejected": outside_rejected,
                  "fixed_cost_cancellation_example": {"one": .002 + .01, "four": .002 + .04, "per_layer": .01},
                  "limits": "Synthetic interpolation lesson. Production ChannelCostModel fits a conservative coupling envelope; no physical bandwidth attribution."},
                  scope="synthetic complete-span calibration policy")
