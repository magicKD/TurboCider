"""Read-only content binding for local Comfy Z probes; not native source proof."""
import hashlib
from pathlib import Path
import time


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def snapshot(path):
    state = path.stat()
    return (str(path.resolve()), state.st_dev, state.st_ino, state.st_size,
            state.st_mtime_ns, state.st_ctime_ns)


def safetensors(path):
    if path.is_file():
        return [path]
    regular = sorted(p for p in path.glob("*.safetensors") if p.is_file() and not p.is_symlink())
    links = sorted(p for p in path.glob("*.safetensors") if p.is_file() and p.is_symlink())
    if regular:
        return regular
    if len(links) == 1:
        return links
    raise ValueError("missing/ambiguous component safetensors: " + str(path))


def bind_components(model, baseline_bf16, environment):
    root = model if model.is_dir() else model.parent
    comfy_text = root / "split_files/text_encoders/qwen_3_4b.safetensors"
    comfy_vae = root / "split_files/vae/ae.safetensors"
    if not comfy_vae.is_file():
        raise ValueError("component binding currently supports Comfy VAE layouts only")
    paths = {"vae": [comfy_vae], "tokenizer": [root / "tokenizer/tokenizer.json"]}
    if environment:
        paths["encoder"] = [Path(environment["TURBOCIDER_Z_QWEN3_GGUF"])]
        paths["encoder_config"] = [Path(environment["TURBOCIDER_Z_QWEN3_GGUF_CONFIG"])]
    else:
        # Match the native constructor's precedence, not the directory label.
        text = root / "text_encoder" if baseline_bf16 and (root / "text_encoder").is_dir() else comfy_text
        if not text.is_file() and not text.is_dir():
            text = root / "text_encoder"
        paths["encoder"] = safetensors(text)
    bindings, states = {}, {}
    start = time.perf_counter()
    for role, files in paths.items():
        bindings[role] = []
        for path in files:
            before = snapshot(path)
            content = digest(path)
            if before != snapshot(path):
                raise ValueError("component changed during content binding")
            states[path] = before
            bindings[role].append({"bytes": before[3], "sha256": content})
        bindings[role].sort(key=lambda entry: (entry["sha256"], entry["bytes"]))
    return bindings, states, time.perf_counter() - start


def revalidate_components(states):
    if any(before != snapshot(path) for path, before in states.items()):
        raise ValueError("bound component/path generation changed during probe")
