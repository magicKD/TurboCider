"""Host header/required-source gate: file availability is not loaded-model proof."""
from pathlib import Path

from mflux_original_layout import read_header


def verify_required_sources(header, flat_mapping):
    expected, available = set(), set()
    for source, targets in flat_mapping.items():
        expected.update(target for target, _ in targets)
        if source in header:
            available.update(target for target, _ in targets)
    if not expected:
        raise ValueError("required model weight targets must not be empty")
    missing = sorted(expected - available)
    if missing:
        raise ValueError("required model weights missing from source mapping: " + ", ".join(missing[:8]) + f" (total {len(missing)})")
    return dict(required_targets=len(expected), covered_targets=len(available), missing=[])


def inspect_reference_coverage(root, definition, models=None, selected_components=None):
    # Uses reference definitions, not weight data or GPU arrays. The stock
    # mapper silently skips missing fields; reject those before construction.
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from mlx.utils import tree_flatten
    result = {}
    for component in definition.get_components():
        if selected_components is not None and component.name not in selected_components:
            continue
        header = {}
        for path in sorted((Path(root) / component.hf_subdir).glob("*.safetensors")):
            values, _ = read_header(path)
            for name, value in values.items():
                if name != "__metadata__":
                    if name in header:
                        raise ValueError("duplicate source tensor across reference component shards")
                    header[name] = value
        all_targets = component.mapping_getter()
        required = [target for target in all_targets if target.required]
        blocks = component.num_blocks if component.num_blocks is not None else WeightMapper._detect_num_blocks(header)
        layers = component.num_layers if component.num_layers is not None else WeightMapper._detect_num_layers(header)
        flat = WeightMapper._build_flat_mapping(required, blocks, layers)
        if models is not None:
            # Generic VAE placeholder expansion includes a third encoder
            # resnet which the actual architecture does not have. Use actual
            # parameter paths, not an exemption for guessed missing names.
            actual = {name for name, _ in tree_flatten(models[component.name].parameters())}
            flat = WeightMapper._build_flat_mapping(all_targets, blocks, layers)
            learned = {name for name in actual if name.endswith((".weight", ".bias"))}
            mapped = {target for targets in flat.values() for target, _ in targets}
            if learned - mapped:
                raise ValueError("actual learned model parameters lack reference mapping: " + ", ".join(sorted(learned - mapped)[:8]))
            # Also retain mapped buffers (e.g. FLUX VAE BN mean/variance).
            # Only architecture constants with no source mapping are excluded.
            flat = {source: [(target, transform) for target, transform in targets if target in actual]
                    for source, targets in flat.items()}
        result[component.name] = verify_required_sources(header, flat)
    return result


def verify_saved_parameter_descriptions(header, parameters):
    """Saved-format uses actual module paths, never guessed HF aliases."""
    stored = {key:value for key,value in header.items() if key != "__metadata__"}
    if not parameters or set(stored) != set(parameters):
        raise ValueError("all actual saved module parameters must be present exactly once")
    for key,actual in parameters.items():
        source = stored[key]
        if source["shape"] != actual["shape"] or source["dtype"] != actual["dtype"]:
            raise ValueError("actual saved parameter dtype/shape differs: "+key)
    return dict(required_targets=len(parameters),covered_targets=len(parameters),missing=[])


def inspect_saved_dit_coverage(root, definition, models):
    """Exact DiT-only g32/Q4 save; encoder/VAE must remain original dense BF16."""
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten
    result = inspect_reference_coverage(root,definition,models,selected_components=("text_encoder","vae"))
    files = sorted((Path(root)/"transformer").glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("one complete matched saved DiT artifact required")
    header,_ = read_header(files[0])
    metadata = header.get("__metadata__",{})
    if metadata.get("quantization_level") != "4" or metadata.get("group_size") != "32":
        raise ValueError("explicit matched Q4/g32 saved DiT metadata required")
    dtypes = {str(mx.uint32):"U32",str(mx.bfloat16):"BF16"}
    actual = {}
    for key,value in tree_flatten(models["transformer"].parameters()):
        if str(value.dtype) not in dtypes:
            raise ValueError("packed U32 or original BF16 parameters required")
        actual[key] = dict(shape=list(value.shape),dtype=dtypes[str(value.dtype)])
    result["transformer"] = verify_saved_parameter_descriptions(header,actual)
    linears = [(path,layer) for path,layer in models["transformer"].named_modules() if isinstance(layer,nn.QuantizedLinear)]
    if len(linears) != 153 or any(layer.bits != 4 or layer.group_size != 32 for _,layer in linears):
        raise ValueError("all153 actual DiT linears must load exact Q4/g32")
    for name in ("text_encoder","vae"):
        if any(key.endswith((".scales",".biases")) or
               (key.endswith((".weight",".bias")) and value.dtype != mx.bfloat16)
               for key,value in tree_flatten(models[name].parameters())):
            raise ValueError("unconverted encoder/VAE must remain original dense BF16")
    result["transformer"].update(quantized_linears=153,bits=4,group_size=32)
    return result
