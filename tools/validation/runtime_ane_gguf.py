"""Explicit CPU-direct GGUF source/retention evidence, not speed qualification."""
import math

from runtime_ane_common import session_counter


def gguf_screen_environment(model_id, routes, import_kind="mlx", source="affine", retain=False, cache_bytes=0):
    if import_kind not in ("mlx", "cpu_direct") or source not in ("affine", "raw") or type(retain) is not bool:
        raise ValueError("invalid GGUF import/source/retention policy")
    if type(cache_bytes) is not int or not 0<=cache_bytes<=1<<30:
        raise ValueError("GGUF allocator cache hint requires 0...1GiB")
    explicit=import_kind=="cpu_direct" or source!="affine" or retain or cache_bytes
    if not explicit:
        return {}
    if (model_id!="z-image-turbo-gguf" or import_kind!="cpu_direct" or
            not routes or any(route not in ("gpu","runtime") for route in routes)):
        raise ValueError("GGUF source/retention screen requires CPU-direct Z GGUF GPU/runtime routes")
    return {"TURBOCIDER_Z_GGUF_IMPORT":"cpu_direct","TURBOCIDER_Z_GGUF_ANE_SOURCE":source,
            "TURBOCIDER_Z_GGUF_RETAIN_PACKED":"1" if retain else "0",
            "TURBOCIDER_Z_GGUF_ALLOCATOR_CACHE_BYTES":str(cache_bytes)}


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def validate_gguf_screen(rows, route, source, retain, cache_bytes=0):
    """Return canonical validation rows while preserving original receipts.

    Only this explicit experiment maps the CPU-direct native GPU label. Raw
    read counters are per bank; retention requires monotonic traffic and real
    zero warm import reads, never a changed source or a silently reloaded bank.
    """
    if (not rows or route not in ("gpu","runtime") or source not in ("affine","raw") or type(retain) is not bool or
            type(cache_bytes) is not int or not 0<=cache_bytes<=1<<30):
        raise ValueError("invalid direct GGUF screen request")
    identity=None;normalized=[];previous_raw=0
    for index,row in enumerate(rows):
        if row.get("model")!="z-image-turbo-gguf":raise ValueError("direct GGUF model identity missing")
        imported=row.get("gguf_import") or {}
        if (imported.get("recipe")!="gguf-mlx-compat-affine-packed-bank-v1" or
                imported.get("experimental") is not True or imported.get("whole_request_bounded_certified") is not False or
                imported.get("gpu_graph_recipe")!="native-compat-eager-v1" or
                session_counter(imported,"allocator_cache_limit_bytes")!=cache_bytes or
                imported.get("session_packed_retention") is not retain or
                imported.get("reused_packed_bank") is not (retain and index>0) or
                imported.get("released_before_vae") is not (not retain)):
            raise ValueError("direct GGUF import/retention/consumer policy was not reported")
        current=tuple(imported.get(key) for key in ("source_sha256","plan_digest","affine_packing_recipe","float_import_recipe"))
        if not all(_digest(value) for value in current[:2]) or not all(isinstance(x,str) and x for x in current[2:]):
            raise ValueError("direct GGUF source/import identity missing")
        if identity is not None and current!=identity:raise ValueError("GGUF source/import changed within resident trial")
        identity=current
        output,packed,read,total_read=(session_counter(imported,key) for key in
            ("output_bytes","packed_capacity_bytes","request_source_read_bytes","logical_source_bytes"))
        load=imported.get("request_load_seconds")
        if (not output or packed<output or not total_read or type(load) not in (int,float) or
                not math.isfinite(load) or load<0 or (retain and index>0 and (read or load)) or
                ((not retain or index==0) and (read!=total_read or load<=0))):
            raise ValueError("GGUF warm reuse lacks real import/read/allocation evidence")
        raw_budget,raw_peak,raw_live,raw_entries,raw_reads,raw_misses=(session_counter(imported,key) for key in
            ("raw_window_budget_bytes","raw_window_peak_bytes","raw_window_live_bytes","raw_window_entries",
             "raw_window_source_read_bytes_session_total","raw_window_misses_session_total"))
        if (raw_budget!=(256<<20 if source=="raw" else 0) or raw_peak>raw_budget or raw_live or raw_entries or
                imported.get("ane_weight_source")!=("bounded-raw-ggml-window-v1" if source=="raw" else "mlx-affine-import")):
            raise ValueError("GGUF ANE source/window/drain evidence does not match requested policy")
        if route=="gpu" or source=="affine":
            if raw_reads or raw_misses or raw_peak:raise ValueError("GPU/affine GGUF route unexpectedly consumed raw ANE weights")
        elif (not raw_misses or not raw_peak or raw_reads<=0 or (retain and raw_reads<=previous_raw)):
            raise ValueError("raw GGUF runtime never consumed fresh raw source weights")
        previous_raw=raw_reads
        if route=="gpu":
            if row.get("runtime_backend")!="mlx_cpp_metal_gguf_cpu_direct_packed" or row.get("runtime_precision")!="z-mlx-compat-affine-v1":
                raise ValueError("direct GGUF GPU baseline was not its native affine consumer")
            normalized.append(dict(row,runtime_backend="mlx_cpp_metal_gguf"))
        else:normalized.append(row)
    return normalized,dict(zip(("source_sha256","plan_digest","affine_packing_recipe","float_import_recipe"),identity))
