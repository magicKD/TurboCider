#!/usr/bin/env python3
"""Real Z GGUF bounded-execution probe. Results are experimental, not certification."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from benchmark_z_image_streaming import process_memory, vm_counters
from benchmark_z_image_metal import loaded_runtime_libraries, system_state_metadata
from z_image_probe_identity import bind_components, revalidate_components


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline-bf16",action="store_true",help="same measurement harness, original pure-GPU BF16; requires --prefetch -1")
    parser.add_argument("--bind-components",action="store_true",help="hash actual local Comfy encoder/VAE/tokenizer outside timing; required for BF16 acceptance screen")
    parser.add_argument("--prefetch",type=int,nargs="+",default=[0,1])
    parser.add_argument("--source-residency",choices=["packed_resident","packed_streamed"],default="packed_resident")
    parser.add_argument("--native-import",choices=["mlx","cpu_direct"],help="explicit experimental import recipe for --prefetch -1")
    parser.add_argument("--compile-packed",action="store_true",help="explicit parameterized native QMM graph; authorized approximation, CPU-direct only")
    parser.add_argument("--packed-compute",choices=["native","f16_down64","f16_mpp_down64","qmm_f16_down64","qmm_f16_ref16_down64","qmm_f16_refmpp_dynamic"],help="experimental FP16 compute, FP32 glue; qmm mode never decodes dense W; refmpp uses dynamic per-row FP16 normalization")
    parser.add_argument("--native-allocator-cache-bytes",type=int,help="explicit FP16 experiment cache hint 0..1GiB; not a RAM cap")
    parser.add_argument("--raw-gpu-cache-bytes",type=int,help="explicit raw GPU profile cache hint 0..1GiB; NOT a RAM cap")
    parser.add_argument("--validate-source-blocks",action="store_true",help="diagnostic independent source trajectory, FP64 every-block N1; not timing/memory")
    parser.add_argument("--retain-packed",action="store_true",help="experimental packed-only session retention across VAE; CPU-direct only, NOT a RAM cap")
    parser.add_argument("--native-weight-limit-bytes",type=int,help="CPU-direct packed-bank managed ceiling; NOT whole-request RAM cap")
    parser.add_argument("--gpu-eval-blocks",action="store_true",help="explicit existing per-main-block eval boundary (separate measured candidate)")
    parser.add_argument("--runs",type=int,default=1)
    parser.add_argument("--warmup",type=int,default=0,help="retain but exclude these requests from warm timing statistics")
    parser.add_argument("--measurement",choices=["diagnostic","timing","memory"],default="diagnostic",
                        help="timing disables observer/event logging; memory samples separately at 10 ms")
    parser.add_argument("--prompt-cache",choices=["hit","miss"],default="hit",
                        help="miss uses a fresh engine per request without changing the prompt")
    parser.add_argument("--encoder-gguf",type=Path)
    parser.add_argument("--encoder-config",type=Path)
    parser.add_argument("--encoder-tokenizer",type=Path)
    parser.add_argument("--encoder-prefetch",type=int,choices=[0,1,2],default=2)
    parser.add_argument("--encoder-source-residency",choices=["packed_resident","packed_streamed"],default="packed_streamed")
    parser.add_argument("--encoder-weight-limit-bytes",type=int,default=1<<30,
                        help="private encoder managed-weight ceiling, NOT whole-request RAM cap")
    parser.add_argument("--encoder-metadata-cache",choices=("on","off"),help="verified CPU metadata/tokenizer only; no weights; off preserves reconstruct-per-request control")
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--precision",choices=["z-source-mixed-v1","z-source-mixed-f16-v1","z-source-exact-f32-v1","z-source-native-affine-v1","z-mlx-compat-affine-v1","z-mlx-compat-f16-v1","z-mlx-compat-f32-v1","z-dense-bf16-v1","z-raw-gpu-affine-f16-v1","z-raw-gpu-fixed-f16-v1","z-raw-gpu-fixed-refresident-f16-v1","z-raw-gpu-dependency-refresident-f16-v1"],default="z-source-mixed-v1")
    parser.add_argument("--prompt",default="A studio photograph of an adult ceramic artist, both hands visible while holding a small blue cup, neutral background, natural skin texture.")
    parser.add_argument("--alternate-prompt",help="diagnostic retained-bank or bound-encoder A/B/A invalidation; >=3 runs, no warmup/cancel")
    parser.add_argument("--dump",action="store_true",help="save diagnostic latents/pixels; not a performance run")
    parser.add_argument("--cancel-once-at-block",type=int,help="cancel the first request at a main block, then test retry")
    parser.add_argument("--cancel-once-at-encoder-layer",type=int,help="cancel first uncached Qwen3 encode, then test retry")
    parser.add_argument("--cancel-once-at-refiner",type=int,help="cancel first streamed refinement unit0..3, then test retry")
    parser.add_argument("--cancel-once-at-import-tensor",type=int,help="cancel first CPU-direct import at tensor0..452, then test retry")
    args=parser.parse_args()
    if args.gpu_eval_blocks: os.environ["TURBOCIDER_Z_EAGER_BLOCKS"]="1"
    gpu_eval_policy="each-main-block-v1" if "TURBOCIDER_Z_EAGER_BLOCKS" in os.environ else "default"
    if any(p not in (-1,0,1,2) for p in args.prefetch) or not 1<=args.runs<=24: parser.error("prefetch -1=native packed, 0/1/2=bounded; runs 1..24")
    if not 0<=args.warmup<=8: parser.error("warmup 0..8")
    cancellations=(args.cancel_once_at_block,args.cancel_once_at_encoder_layer,args.cancel_once_at_refiner,args.cancel_once_at_import_tensor)
    if sum(c is not None for c in cancellations)>1: parser.error("cancellation selectors are mutually exclusive")
    dense_bf16=args.precision=="z-dense-bf16-v1"
    raw_gpu=args.precision in ("z-raw-gpu-affine-f16-v1","z-raw-gpu-fixed-f16-v1","z-raw-gpu-fixed-refresident-f16-v1","z-raw-gpu-dependency-refresident-f16-v1")
    if dense_bf16 and (args.baseline_bf16 or min(args.prefetch)<0 or args.source_residency!="packed_streamed" or args.gpu_eval_blocks):
        parser.error("dense BF16 profile requires bounded packed_streamed and compiled execution")
    if raw_gpu and (args.baseline_bf16 or min(args.prefetch)<0 or args.source_residency!="packed_streamed"):
        parser.error("raw GPU profile requires bounded packed_streamed source")
    if args.raw_gpu_cache_bytes is not None:
        if not raw_gpu or not 0<=args.raw_gpu_cache_bytes<=1<<30: parser.error("raw GPU cache hint requires raw GPU profile and 0..1GiB")
        key="TURBOCIDER_Z_RAW_GPU_CACHE_BYTES";value=str(args.raw_gpu_cache_bytes)
        if key in os.environ and os.environ[key]!=value: parser.error("conflicting raw GPU cache environment")
        os.environ[key]=value
    if args.baseline_bf16 and (args.prefetch!=[-1] or args.native_import or args.native_weight_limit_bytes is not None or args.retain_packed or args.compile_packed or
            any(c is not None for c in cancellations) or args.encoder_gguf or args.precision!="z-source-mixed-v1" or
            any(k.startswith("TURBOCIDER_Z_GGUF_") or k.startswith("TURBOCIDER_Z_QWEN3_") or
                k.startswith("TURBOCIDER_QWEN3_GGUF_") for k in os.environ)):
        parser.error("BF16 baseline requires prefetch -1 without GGUF import/encoder/profile/cancellation controls")
    native_import=args.native_import or os.environ.get("TURBOCIDER_Z_GGUF_IMPORT","mlx")
    if native_import not in ("mlx","cpu_direct"): parser.error("unknown native import recipe")
    if native_import=="cpu_direct" and any(p!=-1 for p in args.prefetch): parser.error("CPU-direct import requires only native packed prefetch -1")
    if args.compile_packed:
        if native_import!="cpu_direct" or args.baseline_bf16 or args.gpu_eval_blocks: parser.error("packed compile requires CPU-direct without baseline/eager controls")
        key="TURBOCIDER_Z_GGUF_COMPILE_PACKED"
        if key in os.environ and os.environ[key]!="1": parser.error("conflicting packed compile environment")
        os.environ[key]="1"
    compile_packed=os.environ.get("TURBOCIDER_Z_GGUF_COMPILE_PACKED","0")=="1"
    compute=args.packed_compute or os.environ.get("TURBOCIDER_Z_GGUF_COMPUTE","native")
    if compute not in ("native","f16_down64","f16_mpp_down64","qmm_f16_down64","qmm_f16_ref16_down64","qmm_f16_refmpp_dynamic"): parser.error("unknown packed compute")
    if compute!="native" and (native_import!="cpu_direct" or not compile_packed or args.baseline_bf16 or args.gpu_eval_blocks):
        parser.error("GPU FP16 compute requires CPU-direct compiled candidate without baseline/eager controls")
    if args.packed_compute:
        key="TURBOCIDER_Z_GGUF_COMPUTE"
        if key in os.environ and os.environ[key]!=compute: parser.error("conflicting packed compute environment")
        os.environ[key]=compute
    if args.native_allocator_cache_bytes is not None:
        if compute=="native" or native_import!="cpu_direct" or not 0<=args.native_allocator_cache_bytes<=1<<30:
            parser.error("GGUF allocator cache requires explicit FP16 CPU-direct recipe and 0..1GiB")
        key="TURBOCIDER_Z_GGUF_ALLOCATOR_CACHE_BYTES";value=str(args.native_allocator_cache_bytes)
        if key in os.environ and os.environ[key]!=value: parser.error("conflicting GGUF allocator cache environment")
        os.environ[key]=value
    if args.validate_source_blocks:
        if (compute=="native" and not raw_gpu) or args.measurement!="diagnostic": parser.error("source block validation requires diagnostic FP16 experiment or raw GPU profile")
        key="TURBOCIDER_Z_GGUF_VALIDATE_BLOCKS"
        if key in os.environ and os.environ[key]!="1": parser.error("conflicting source validation environment")
        os.environ[key]="1"
    if args.measurement!="diagnostic" and os.environ.get("TURBOCIDER_Z_GGUF_VALIDATE_BLOCKS","0")!="0":
        parser.error("source validation cannot contaminate timing/memory")
    if args.retain_packed:
        if native_import!="cpu_direct" or args.baseline_bf16: parser.error("packed retention requires CPU-direct candidate")
        key="TURBOCIDER_Z_GGUF_RETAIN_PACKED"
        if key in os.environ and os.environ[key]!="1": parser.error("conflicting packed retention environment")
        os.environ[key]="1"
    encoder_for_alternation=all((args.encoder_gguf,args.encoder_config,args.encoder_tokenizer)) or bool(os.environ.get("TURBOCIDER_Z_QWEN3_GGUF"))
    if args.alternate_prompt is not None and (args.measurement!="diagnostic" or not (args.retain_packed or encoder_for_alternation) or
            args.runs<3 or args.warmup or any(c is not None for c in cancellations) or not args.alternate_prompt.strip()):
        parser.error("alternate prompt requires diagnostic retained-bank or bound-encoder >=3 runs without warmup/cancel")
    if args.native_import:
        if os.environ.get("TURBOCIDER_Z_GGUF_IMPORT",native_import)!=native_import: parser.error("conflicting native import environment")
        os.environ["TURBOCIDER_Z_GGUF_IMPORT"]=native_import
    if args.native_weight_limit_bytes is not None:
        if native_import!="cpu_direct" or args.native_weight_limit_bytes<=0: parser.error("positive native weight ceiling requires CPU-direct import")
        key="TURBOCIDER_Z_GGUF_PACKED_WEIGHT_LIMIT_BYTES";value=str(args.native_weight_limit_bytes)
        if key in os.environ and os.environ[key]!=value: parser.error("conflicting native weight ceiling environment")
        os.environ[key]=value
    if args.cancel_once_at_import_tensor is not None and (native_import!="cpu_direct" or not 0<=args.cancel_once_at_import_tensor<=452):
        parser.error("import cancellation requires CPU-direct import and tensor0..452")
    if args.measurement!="diagnostic" and (args.dump or any(c is not None for c in cancellations)):
        parser.error("dump/cancellation requires diagnostic measurement")
    encoder_paths=(args.encoder_gguf,args.encoder_config,args.encoder_tokenizer)
    if any(encoder_paths) and not all(encoder_paths): parser.error("encoder GGUF/config/tokenizer must be supplied together")
    if args.encoder_weight_limit_bytes<=0: parser.error("encoder managed-weight ceiling must be positive")
    encoder_environment={}
    if all(encoder_paths):
        # The native Z consumer binds its own tokenizer under the model root.
        # Verify this explicit reference instead of pretending an env override
        # can replace that tokenizer.
        model_root=args.model if args.model.is_dir() else args.model.parent
        native_tokenizer=model_root/"tokenizer/tokenizer.json"
        if not all(path.is_file() for path in (*encoder_paths,native_tokenizer)):
            parser.error("missing encoder GGUF/config/tokenizer or native model tokenizer")
        with args.encoder_tokenizer.open("rb") as given,native_tokenizer.open("rb") as native:
            if hashlib.file_digest(given,"sha256").digest()!=hashlib.file_digest(native,"sha256").digest():
                parser.error("encoder tokenizer differs from native model tokenizer")
        encoder_environment={"TURBOCIDER_Z_QWEN3_GGUF":str(args.encoder_gguf.resolve()),
            "TURBOCIDER_Z_QWEN3_GGUF_CONFIG":str(args.encoder_config.resolve()),
            "TURBOCIDER_QWEN3_GGUF_PREFETCH":str(args.encoder_prefetch),
            "TURBOCIDER_QWEN3_GGUF_SOURCE_RESIDENCY":args.encoder_source_residency,
            "TURBOCIDER_QWEN3_GGUF_WEIGHT_LIMIT_BYTES":str(args.encoder_weight_limit_bytes)}
        for key,value in encoder_environment.items():
            if key in os.environ and os.environ[key]!=value: parser.error("conflicting encoder environment: "+key)
        os.environ.update(encoder_environment)
    encoder_environment={key:value for key,value in os.environ.items()
        if key.startswith("TURBOCIDER_QWEN3_") or key.startswith("TURBOCIDER_Z_QWEN3_")}
    if args.encoder_metadata_cache is not None:
        if not encoder_environment.get("TURBOCIDER_Z_QWEN3_GGUF"): parser.error("encoder metadata cache requires bound GGUF encoder")
        key="TURBOCIDER_QWEN3_GGUF_METADATA_CACHE";value="1" if args.encoder_metadata_cache=="on" else "0"
        if key in os.environ and os.environ[key]!=value: parser.error("conflicting encoder metadata cache environment")
        os.environ[key]=value;encoder_environment[key]=value
    if args.output.exists() or args.output.is_symlink(): parser.error("output already exists")
    if args.cancel_once_at_block is not None and (not 0<=args.cancel_once_at_block<30 or args.prefetch[0]<0):
        parser.error("cancellation requires a bounded first request and block 0..29")
    if args.cancel_once_at_encoder_layer is not None and (not 0<=args.cancel_once_at_encoder_layer<35 or args.cancel_once_at_block is not None):
        parser.error("encoder cancellation requires layer 0..34 and no block cancellation")
    if args.cancel_once_at_refiner is not None and (not 0<=args.cancel_once_at_refiner<4 or args.cancel_once_at_block is not None or
            args.cancel_once_at_encoder_layer is not None or args.source_residency!="packed_streamed" or args.prefetch[0]<0):
        parser.error("refiner cancellation requires unit0..3, streamed source and no other cancellation")
    args.output.mkdir(parents=True)
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        components,component_states,binding_seconds=bind_components(args.model,args.baseline_bf16,encoder_environment) if args.bind_components else (None,{},0)
        library=C.CDLL(str(args.library.resolve()))
        library.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        library.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        library.tc_engine_free.argtypes=[C.c_void_p]
        library.tc_engine_cancel.argtypes=[C.c_void_p]
        library.tc_string_free.argtypes=[C.c_void_p]
        library.tc_system_json.argtypes=[]
        library.tc_system_json.restype=C.c_void_p
        def take(pointer):
            if not pointer.value: return None
            value=C.string_at(pointer).decode(); library.tc_string_free(pointer); return value
        def create_engine():
            engine,error=C.c_void_p(),C.c_void_p()
            status=library.tc_engine_create_model(b"z-image-turbo" if args.baseline_bf16 else b"z-image-turbo-gguf",str(args.model.resolve()).encode(),C.byref(engine),C.byref(error))
            message=take(error)
            if status: raise RuntimeError(message)
            return engine
        engine=create_engine()
        report={"schema_version":1,"scope":"local pure-GPU BF16/quantized execution screen; not whole-request memory/performance qualification",
            "hardware":json.loads(take(C.c_void_p(library.tc_system_json()))),"system_state":system_state_metadata(),
            "library_sha256":hashlib.sha256(args.library.read_bytes()).hexdigest(),"loaded_libraries":loaded_runtime_libraries(),
            "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_packed_math_profile":None if args.baseline_bf16 else "z-mlx-compat-affine-v1",
            "native_import":None if args.baseline_bf16 else native_import,
            "gpu_eval_policy":gpu_eval_policy,
            "native_import_environment":{k:v for k,v in os.environ.items() if k.startswith("TURBOCIDER_Z_GGUF_")},
            "baseline_bf16":args.baseline_bf16,"runtime_environment":{k:v for k,v in os.environ.items() if k.startswith("TURBOCIDER_")},
            "component_binding":components,"component_binding_seconds":binding_seconds,
            "measurement":args.measurement,"warmup_requests_per_layout":args.warmup,
            "prompt_cache_policy":args.prompt_cache,"encoder_environment":encoder_environment,
            "runs":[],"status":"running"}
        indexes={}; cancel_used=False
        try:
            for p in args.prefetch:
                for run in range(args.runs+args.warmup):
                    if args.prompt_cache=="miss" and report["runs"]:
                        library.tc_engine_free(engine);engine=C.c_void_p();engine=create_engine()
                    index=indexes.get(p,0);indexes[p]=index+1
                    name=f"p{p}-{index}" if p>=0 else f"packed-{index}"; image=args.output/(name+".png")
                    request={"schema_version":2,"model":"z-image-turbo" if args.baseline_bf16 else "z-image-turbo-gguf","operation":"image.generate",
                        "inputs":[{"kind":"text","role":"prompt","text":args.alternate_prompt if args.alternate_prompt and run%2 else args.prompt}],
                        "outputs":[{"kind":"image","path":str(image.resolve()),"width":args.size,"height":args.size,"audio":False}],
                        "sampling":{"seed":args.seed,"steps":args.steps},
                        "execution":{"policy":"gpu","quantized_execution":{"schema_version":1,"enabled":True,"prefetch_layers":p,"precision_profile":args.precision,"source_residency":args.source_residency}},
                        "parameters":{"dynamic_text":True}}
                    if args.dump: request["dump_tensors"]=str((args.output/(name+"-tensors")).resolve())
                    if p<0: request["execution"].pop("quantized_execution")
                    if dense_bf16 or compile_packed or raw_gpu:
                        request["execution"]["allow_approximation"]=True
                        request["parameters"]["compile_gpu"]=True
                    samples=[]; errors=[]; stopped=threading.Event()
                    def sample():
                        while not stopped.is_set():
                            try: samples.append({"time":time.monotonic(),**process_memory()})
                            except Exception as exception: errors.append(str(exception))
                            stopped.wait(.01 if args.measurement=="memory" else .02)
                    sampler=threading.Thread(target=sample,daemon=True)
                    before=vm_counters()
                    if args.measurement!="timing": sampler.start()
                    with (args.output/(name+"-events.jsonl")).open("x") as events:
                        should_cancel=any(c is not None for c in cancellations) and not cancel_used
                        triggered=[False]
                        @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                        def event(raw,_):
                            progress=json.loads(raw)
                            if args.measurement!="timing":
                                events.write(json.dumps({**progress,"time_monotonic":time.monotonic()})+"\n"); events.flush()
                            cancel_phase = "load_gguf_direct_tensor" if args.cancel_once_at_import_tensor is not None else "z_image_gguf_refiner" if args.cancel_once_at_refiner is not None else "z_image_text_encode" if args.cancel_once_at_encoder_layer is not None else "z_image_denoise_block"
                            cancel_at = args.cancel_once_at_import_tensor if args.cancel_once_at_import_tensor is not None else args.cancel_once_at_refiner if args.cancel_once_at_refiner is not None else args.cancel_once_at_encoder_layer if args.cancel_once_at_encoder_layer is not None else args.cancel_once_at_block
                            if should_cancel and not triggered[0] and progress.get("phase")==cancel_phase and progress.get("completed")==cancel_at:
                                library.tc_engine_cancel(engine);triggered[0]=True
                        value,error=C.c_void_p(),C.c_void_p(); start=time.perf_counter()
                        try:
                            print("START",name,flush=True)
                            status=library.tc_engine_generate(engine,json.dumps(request).encode(),
                                None if args.measurement=="timing" else event,None,C.byref(value),C.byref(error))
                        finally:
                            stopped.set()
                            if args.measurement!="timing": sampler.join()
                    elapsed=time.perf_counter()-start; raw=take(value); message=take(error); after=vm_counters()
                    revalidate_components(component_states)
                    row={"prefetch":p,"index":index,"request":request,"status":status,"error":message,
                        "warmup":run<args.warmup,"measurement":args.measurement,
                        "wall_seconds":elapsed,"metrics":json.loads(raw) if raw else None,"memory_samples":samples,
                        "sampling_errors":errors,"vm_deltas":{key:after[key]-before[key] for key in before},
                        "png_sha256":hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else None}
                    row["cancellation_triggered"]=triggered[0]
                    if status==0:
                        source_identity_started=time.perf_counter()
                        qe=row["metrics"].get("quantized_execution") or row["metrics"].get("gguf_import")
                        if qe: report["dit_source_sha256"]=qe["source_sha256"]
                        elif "dit_source_sha256" not in report:
                            if args.baseline_bf16:
                                checkpoint=Path(os.environ["TURBOCIDER_Z_IMAGE_TRANSFORMER"]) if os.environ.get("TURBOCIDER_Z_IMAGE_TRANSFORMER") else args.model/"split_files/diffusion_models/z_image_turbo_bf16.safetensors"
                            else:
                                checkpoint=args.model/row["metrics"]["checkpoint"] if args.model.is_dir() else args.model
                            with checkpoint.open("rb") as source:
                                report["dit_source_sha256"]=hashlib.file_digest(source,"sha256").hexdigest()
                        row["identity_verification_seconds"]=time.perf_counter()-source_identity_started
                    if samples:
                        row["memory_sample_max_gap_seconds"]=max(
                            (b["time"]-a["time"] for a,b in zip(samples,samples[1:])),default=0)
                    report["runs"].append(row)
                    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
                    print("DONE",name,status,round(elapsed,3),message,flush=True)
                    if triggered[0]:
                        cancel_used=True
                        if not status or not message or "cancel" not in message.lower() or image.exists():
                            raise RuntimeError("cancellation did not fail cleanly")
                    elif status: raise RuntimeError(message)
            report["status"]="completed"
            if hashlib.sha256(args.library.read_bytes()).hexdigest()!=report["library_sha256"]:
                raise RuntimeError("library changed during probe; evidence invalid")
            successful=[row for row in report["runs"] if row["status"]==0]
            report["png_bytes_exact"]=len({row["png_sha256"] for row in successful})==1 if len(successful)>1 else None
            bounded=[row for row in successful if row["prefetch"]>=0]
            report["bounded_png_bytes_exact"]=len({row["png_sha256"] for row in bounded})==1 if len(bounded)>1 else None
            (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
            print("PNG exact:",report["png_bytes_exact"],flush=True)
        finally: library.tc_engine_free(engine)

if __name__=="__main__": main()
