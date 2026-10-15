#!/usr/bin/env python3
"""Re-test a Python INT8 feature bridge, without assuming it will always fail."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import platform
import sys
import tempfile

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
from common import emit, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-overlay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("report already exists")
    if args.coreml_overlay:
        if not (args.coreml_overlay / "coremltools/__init__.py").is_file():
            parser.error("overlay package missing")
        sys.path.insert(0, str(args.coreml_overlay.resolve()))
    with redirect_stdout(sys.stderr):
        import numpy as np
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    if platform.system() != "Darwin" or not hasattr(ct.target, "macOS26"):
        emit(report("python-int8-bridge-probe", {"status":"unsupported_toolchain", "coremltools":ct.__version__}), args.output)
        return 3
    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.int8)], opset_version=ct.target.macOS26)
    def program(x):
        return mb.cast(x=x, dtype="fp16", name="y")
    with redirect_stdout(sys.stderr):
        model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS26,
            inputs=[ct.TensorType(name="x", shape=(2,3), dtype=np.int8)],
            compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    with tempfile.TemporaryDirectory(prefix="tc-python-int8-bridge-") as temporary:
        folder = Path(temporary)
        model.save(str(folder / "graph.mlpackage"))
        ct.models.utils.compile_model(str(folder / "graph.mlpackage"), destination_path=str(folder / "graph.mlmodelc"))
        loaded = ct.models.CompiledMLModel(str(folder / "graph.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
        source = np.array([[-128,-1,0],[1,7,127]], np.int8)
        try:
            result = loaded.predict({"x":source})
        except RuntimeError as error:
            bridge_limit = "value type not convertible" in str(error)
            details = {"status":"python_feature_bridge_limited" if bridge_limit else "prediction_failed",
                       "graph_built":True,"python_bridge_supported":False,"coremltools":ct.__version__,"error":str(error)}
            status = 3 if bridge_limit else 1
        else:
            actual = np.asarray(result["y"])
            if actual.shape != source.shape or not np.array_equal(actual, source.astype(np.float16)):
                raise ValueError("INT8 cast witness failed")
            details = {"status":"bridge_supported", "graph_built":True,"python_bridge_supported":True,"coremltools":ct.__version__}
            status = 0
        del loaded, model
    emit(report("python-int8-bridge-probe", details,
        scope="actual tiny public CPU_ONLY model; re-test Python INT8 conversion, not ANE arithmetic", backend="public_coreml_python"), args.output)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
