"""PE multimodal layout vs official Qwen3.5 position builder; no vision-quality claim."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace, MethodType

import torch
import transformers
from safetensors.torch import load_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # These official position methods use geometry only, so no 9B model is
    # needed and no model method is replaced by a reimplemented oracle.
    model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    model.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, model)
    cases = []
    with tempfile.TemporaryDirectory(prefix="tc-pe-layout-") as directory:
        tensors = Path(directory) / "layouts.safetensors"
        subprocess.run([str(args.probe.resolve()), str(tensors)], check=True)
        data = load_file(str(tensors))
        for i in range(12):
            prefix = f"case{i}."
            ids, types = (data[prefix+k].long() for k in ("ids", "types"))
            grids = data.get(prefix+"grids", torch.empty((0, 3), dtype=torch.int32)).long()
            expected, delta = Qwen3_5Model.get_rope_index(model, ids, types, grids)
            assert torch.equal(expected[:, 0], data[prefix+"positions"].long())
            assert int(delta.item()) == int(data[prefix+"delta"].item())
            assert torch.equal(types == 1, ids == 248056)
            cases.append(dict(images=grids.shape[0], tokens=ids.numel(), rope_delta=int(delta.item())))
    args.output.write_text(json.dumps(dict(passed=True, cases=cases,
        transformers_version=transformers.__version__, torch_version=torch.__version__,
        scope="PE expanded image IDs, token types, mRoPE positions, continuation delta; no image processing or generation"), indent=2)+"\n")
    print(f"PASS exact official PE mRoPE/token-type parity: {len(cases)} cases")


if __name__ == "__main__":
    main()
