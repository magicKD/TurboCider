# Extending TurboCider

TurboCider has two extension levels. A new checkpoint that already speaks an
existing H3, LTX, FLUX.2, or FastMetal contract needs only a model pack. A new
runtime family needs a small Python adapter package as well.

## Add a model pack

A model pack declares public capabilities, relocatable path candidates, and
one or more execution plans. The built-in engine families may also use
TurboCider's pinned download/conversion recipes; an external runtime package
owns any runtime-specific preparation that has not been added to the common
model lifecycle yet.
Put its JSON file in a directory listed by `TURBOCIDER_MODEL_PACKS` rather than
editing the bundled catalog.

Plans are the admission boundary for placement. A GPU+ANE plan should declare:

- every required native executable, checkpoint, manifest, and bridge path;
- fixed width, height, frame-count, FPS, and model dimensions when those are
  part of the compiled Core ML ABI;
- the device profiles on which quality and performance were measured;
- whether a persistent worker is required;
- `production: true` only after direct/integrated parity and quality gates pass.

The App discovers tasks, inputs, audio support, profiles, execution modes, and
recommended generation values from the pack. No App change is needed for
another checkpoint using an existing adapter.

Set `capabilities.recommended_persistent` when the normal App path should keep
an engine worker alive between requests. This is a model/runtime property, not
a hard-coded FLUX-only behavior.

## Add a runtime adapter

An adapter translates `GenerationRequest` plus the selected `ExecutionPlan`
into a `CommandSpec`. The native process remains responsible for the actual
GPU/ANE graph and may emit normal logs, recognized progress lines, and an
optional `turbocider_result=<json>` metrics record.

```python
from pathlib import Path

from turbocider.adapters import EngineAdapter
from turbocider.adapters.base import CommandSpec


class OrchardAdapter(EngineAdapter):
    name = "orchard"

    def build_command(self, model, request, plan, output_path: Path):
        return CommandSpec(
            argv=[
                model.config["executable_path"],
                "--prompt", request.prompt,
                "--output", str(output_path),
            ],
            cwd=Path(model.config["engine_root"]),
            output_paths=[output_path],
            metadata={
                "engine": self.name,
                "plan": plan.id,
                "expected_seconds": 10.0,
                "progress_phases": [
                    {"name": "denoise", "weight": 0.9},
                    {"name": "decode", "weight": 0.1},
                ],
            },
        )

    def doctor(self, model):
        executable = Path(model.config["executable_path"])
        return {"engine": self.name, "available": executable.is_file()}
```

Publish the class as a Python entry point whose name exactly matches the model
pack's `engine` value:

```toml
[project.entry-points."turbocider.adapters"]
orchard = "orchard_turbocider:OrchardAdapter"
```

TurboCider loads only the entry point requested by a registered model pack.
Installing an adapter package grants it normal Python code execution, whereas
a model-pack JSON file alone cannot register executable code. Embedded hosts
and tests may call `register_adapter("orchard", OrchardAdapter)` directly.
When the Swift daemon helper should use a virtual environment containing that
package, set `TURBOCIDER_PYTHON=/path/to/venv/bin/python` before launching the
App.

## Preserve native-engine performance

Keep adapters thin: translate arguments and environment, do not copy model
weights or tensors into the control plane. For persistent runtimes, include
all model-, precision-, cache-, attention-, and artifact-changing values in the
worker identity. Supply a benchmark JSON comparing the direct command with the
TurboCider path, and require byte identity within one backend or explicit
numeric/media thresholds when comparing GPU with an approximate GPU+ANE path.
