"""Single real Z-Image FFN partition probe; CPU+NE configuration is not residency proof."""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--runs',type=int,default=20)
a=p.parse_args()
report=[]
rng=np.random.default_rng(42)
base=rng.normal(0,.3,(1,3840,1,1536)).astype(np.float16)
for label,path,shapes in [('fixed1056',a.baseline,[1056]),('fixed1536',a.root/'probe-fixed1536',[1536]),('enumerated',a.root/'probe-enumerated',[1056,1536,1088,1056]),('range',a.root/'probe-range',[1056,1536,1088,1056])]:
 package=path/'block0_mlp_branch.int8_pc.mlpackage'
 start=time.perf_counter()
 compiled=a.root/(label+'-probe.mlmodelc')
 if not compiled.exists(): ct.models.utils.compile_model(str(package),destination_path=str(compiled))
 compile_seconds=time.perf_counter()-start
 for hint in ([False,True] if label in ['enumerated','range'] else [False]):
  start=time.perf_counter()
  model=ct.models.CompiledMLModel(str(compiled),compute_units=ct.ComputeUnit.CPU_AND_NE,optimization_hints={'reshapeFrequency':ct.ReshapeFrequency.Infrequent} if hint else None)
  entry={'mode':label,'infrequent':hint,'compile_seconds':compile_seconds,'load_seconds':time.perf_counter()-start,'shapes':[]}
  for n in shapes:
   x=np.ascontiguousarray(base[:,:,:,:n]);samples=[]
   for i in range(a.runs+1):
    start=time.perf_counter();y=model.predict({'x':x})['y'];samples.append(time.perf_counter()-start)
   np.save(a.root/f'{label}-{hint}-{n}.npy',y)
   entry['shapes'].append({'rows':n,'first_seconds':samples[0],'median_seconds':float(np.median(samples[1:])),'samples':samples,'finite':bool(np.isfinite(y).all()),'output_shape':list(y.shape)})
   print(label,hint,n,entry['shapes'][-1]['median_seconds'],flush=True)
  del model;gc.collect()
  report.append(entry)
  (a.root/'probe-report.json').write_text(json.dumps(report,indent=2))
 try:
  plan=MLComputePlan.load_from_path(str(compiled),compute_units=ct.ComputeUnit.CPU_AND_NE)
  placement=[]
  for name,fn in plan.model_structure.program.functions.items():
   for op in fn.block.operations:
    usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
    placement.append({'operator':op.operator_name,'preferred':type(usage.preferred_compute_device).__name__ if usage else None})
  (a.root/(label+'-plan.json')).write_text(json.dumps(placement,indent=2))
 except Exception as ex: print('plan error',str(ex),flush=True)
