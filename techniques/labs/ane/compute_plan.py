"""Normalize public Core ML plan data without logs, cache deletion or inference.

An anticipated device and relative cost are predictions, never hardware timing.
This module has no framework import until the caller explicitly requests a plan.
"""
import math


def summarize(plan):
    program = plan.model_structure.program
    if program is None:
        raise ValueError("this probe requires an ML Program")
    operations = []

    def block_items(block, prefix):
        for ordinal, operation in enumerate(block.operations):
            locator = prefix + "/" + str(ordinal)
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
            weight = None if cost is None else float(cost.weight)
            if weight is not None and (not math.isfinite(weight) or not 0 <= weight <= 1):
                raise ValueError("invalid estimated compute-plan weight")
            operations.append({"locator": locator, "operator": operation.operator_name,
                               "outputs": [value.name for value in operation.outputs],
                               "preferred_device": None if usage is None else type(usage.preferred_compute_device).__name__,
                               "supported_devices": None if usage is None else [type(device).__name__ for device in usage.supported_compute_devices],
                               "estimated_cost_weight": weight, "source_kind": "predicted"})
            for index, block in enumerate(operation.blocks):
                block_items(block, locator + "/block-" + str(index))

    for name, function in sorted(program.functions.items()):
        block_items(function.block, name)
    return {"schema": "tc-coreml-plan-v1", "operations": operations,
            "operation_count": {"value": len(operations), "source_kind": "derived"},
            "physical_ane_placement": "unverified", "physical_operation_timing": None,
            "scope": "public MLComputePlan predictions; not physical device attribution or per-op latency"}


def inspect(compiled, compute_units):
    from coremltools.models.compute_plan import MLComputePlan
    return summarize(MLComputePlan.load_from_path(str(compiled), compute_units=compute_units))
