"""Host-only full-RGBA, exact one-option and actual-prefix comparison guards."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/profiling/qwen_base'))
import function_request_audit as audit


def pair():
    values=[]
    for enabled in (False,True):
        values.append(dict(summary=dict(compiled_complement=enabled,environment=dict(partition='6144',
            TURBOCIDER_QWEN21_COMPILED_COMPLEMENT='1' if enabled else '0')),
            images=[np.full((16,16,4),i,dtype=np.uint8) for i in (1,2,1)],
            report=dict(runs=[dict(metrics=dict(qwen_runtime_prefix_snapshot=dict(hit=i>0,recipe_sha256=('b' if enabled else 'a')*64))) for i in range(3)])))
    return values


class CompiledComplementRequestAuditTests(unittest.TestCase):
    def test_complete_denominators_recomputed_even_without_speed_qualification(self):
        summary=dict(requests=3,load_error='external activity',analysis=dict(complete_requests=3,first_request_seconds=40.,
            hot_request_seconds=[28.,30.],hot_median_seconds=29.))
        report=dict(runs=[dict(request_wall_including_export=v) for v in (40.,28.,30.)])
        self.assertEqual(audit.timing_contract(summary,report),[40.,28.,30.])
        for change in (dict(first_request_seconds=28.),dict(hot_request_seconds=[28.]),dict(hot_median_seconds=28.),dict(complete_requests=2)):
            bad=copy.deepcopy(summary);bad['analysis'].update(change)
            with self.assertRaises(ValueError):audit.timing_contract(bad,report)
        for value in (0.,float('nan'),True):
            bad=copy.deepcopy(report);bad['runs'][1]['request_wall_including_export']=value
            with self.assertRaises(ValueError):audit.timing_contract(summary,bad)

    def test_exact_single_option_complete_images_and_isolated_prefixes(self):
        self.assertEqual(audit.complement_pair_contract(*pair()),'TURBOCIDER_QWEN21_COMPILED_COMPLEMENT')
        for change in (lambda x:x['summary'].update(compiled_complement=False),
            lambda x:x['summary']['environment'].update(partition='5120'),
            lambda x:x['images'][2].__setitem__((0,0,3),9),lambda x:x['images'].pop(),
            lambda x:x['report']['runs'].pop(),
            lambda x:x['report']['runs'][1]['metrics']['qwen_runtime_prefix_snapshot'].update(hit=False),
            lambda x:x['report']['runs'][0]['metrics']['qwen_runtime_prefix_snapshot'].update(recipe_sha256='a'*64),
            lambda x:x['report']['runs'][0]['metrics'].pop('qwen_runtime_prefix_snapshot')):
            off,on=pair();bad=copy.deepcopy(on);change(bad)
            with self.assertRaises(ValueError):audit.complement_pair_contract(off,bad)

    def test_no_prefix_pair_is_allowed_but_not_asymmetric_receipts(self):
        off,on=pair()
        for item in (off,on):
            for run in item['report']['runs']:run['metrics'].clear()
        audit.complement_pair_contract(off,on)
        off['report']['runs'][0]['metrics']['qwen_runtime_prefix_snapshot']=dict(hit=False,recipe_sha256='a'*64)
        with self.assertRaises(ValueError):audit.complement_pair_contract(off,on)


if __name__=='__main__':unittest.main()
