import json
from coremltools.models.compute_plan import MLComputePlan
import coremltools as ct

def inspect(path):
    p=MLComputePlan.load_from_path(str(path),compute_units=ct.ComputeUnit.CPU_AND_NE)
    out=[]
    def walk(block):
        for op in block.operations:
            u=p.get_compute_device_usage_for_mlprogram_operation(op)
            c=p.get_estimated_cost_for_mlprogram_operation(op)
            out.append({'operator':op.operator_name,'preferred':type(u.preferred_compute_device).__name__ if u else None,'supported':[type(d).__name__ for d in u.supported_compute_devices] if u else [],'estimated_cost':c.weight if c else None})
            for b in op.blocks:walk(b)
    for name,fn in p.model_structure.program.functions.items():walk(fn.block)
    return out
if __name__=='__main__':
    import sys
    print(json.dumps(inspect(sys.argv[1]),indent=2))
