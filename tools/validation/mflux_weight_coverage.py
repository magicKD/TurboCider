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


def inspect_reference_coverage(root, definition, models=None):
    # Uses reference definitions, not weight data or GPU arrays. The stock
    # mapper silently skips missing fields; reject those before construction.
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from mlx.utils import tree_flatten
    result = {}
    for component in definition.get_components():
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
            flat = {source: [(target, transform) for target, transform in targets if target in learned]
                    for source, targets in flat.items()}
        result[component.name] = verify_required_sources(header, flat)
    return result
