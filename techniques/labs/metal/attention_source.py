"""Read an explicit bounded BF16 replay capsule; never imports product code."""
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            value.update(chunk)
    return value.hexdigest()


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError("duplicate source JSON key")
        result[key]=value
    return result


def local(root,name):
    relative=Path(name)
    path=(root/relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("attention asset missing or escaping capture root")
    return path


def load(root, phase, np):
    receipt=local(root,"receipt.json")
    info=json.loads(receipt.read_text(),object_pairs_hook=unique_object)
    verified=info.get("build_inputs_verified")
    if (info.get("schema")!="tc-qwen-attention-capture-receipt-v1" or info.get("status")!="captured-replay-with-controls"
            or info.get("model_source_stats_unchanged") is not True or info.get("source_inputs_unchanged") is not True
            or not isinstance(verified,dict) or any(type(verified.get(key)) is not int or verified[key]<=0 for key in ("sources","mlx"))
            or info.get("build_inputs_verified")!=info.get("build_inputs_verified_after")):
        raise ValueError("completed source-stable original-input replay required")
    driver=info["driver_report"]
    if driver.get("loaded_runtime_id")!=info.get("runtime_build_id") or driver.get("schema")!="tc-qwen-attention-observer-replay-v1":
        raise ValueError("loaded runtime identity differs")
    records=[record for record in driver["records"] if record["phase"]==phase]
    if len(records)!=1 or driver.get("vae_decode") or driver.get("image_export"):
        raise ValueError("unique bounded attention phase required")
    record=records[0]
    if (record.get("actual_prefix_exact") is not True or record.get("source_inputs_unchanged") is not True
            or type(record.get("full_block_relative_l2")) not in (int,float) or not 0<=record["full_block_relative_l2"]<=.001):
        raise ValueError("full-block and actual prefix controls failed")
    artifact=next(item for item in info["artifacts"] if item["path"]==record["file"])
    path=local(root,record["file"])
    if path.stat().st_size>128*1024*1024 or digest(path)!=artifact["sha256"]:
        raise ValueError("capture tensor file identity/cap differs")
    with path.open("rb") as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:raise ValueError("short attention safetensors")
        size=struct.unpack("<Q",prefix)[0]
        if not 2<=size<=16*1024*1024 or size+8>path.stat().st_size:raise ValueError("unbounded attention header")
        header=json.loads(stream.read(size),object_pairs_hook=unique_object)
    if header.get("__metadata__",{}).get("recipe")!="qwen21-observer-attention-replay-v1" or header["__metadata__"].get("phase")!=phase:
        raise ValueError("attention replay serialization recipe differs")
    arrays={}
    for tensor in record["tensors"]:
        name=tensor["name"];item=header[name];shape=item["shape"]
        if (item["dtype"]!="BF16" or shape!=tensor["shape"] or len(shape)!=4
                or shape[0]!=1 or shape[1]!=32 or shape[3]!=128 or not 1<=shape[2]<=1536):
            raise ValueError("original replay BF16 head geometry differs")
        begin,end=item["data_offsets"];count=1
        for dim in shape:count*=dim
        if type(begin) is not int or type(end) is not int or begin<0 or end-begin!=2*count or 8+size+end>path.stat().st_size:
            raise ValueError("source tensor range differs")
        raw=np.memmap(path,mode="r",dtype="<u2",offset=8+size+begin,shape=shape)
        if hashlib.sha256(raw.tobytes()).hexdigest()!=tensor["sha256"]:
            raise ValueError("logical tensor digest differs")
        value=(np.asarray(raw,dtype=np.uint32)<<16).view(np.float32)
        if not np.isfinite(value).all():raise ValueError("nonfinite original attention input/reference")
        arrays[name]=value
    if set(arrays)!={"q","k","v","reference"}:
        raise ValueError("complete original attention capsule required")
    return arrays,record,{"receipt_sha256":digest(receipt),"artifact_sha256":artifact["sha256"],
                         "runtime_build_id":info["runtime_build_id"],"source_seed":info["seed"],
                         "capture_kind":info["capture_kind"]}
