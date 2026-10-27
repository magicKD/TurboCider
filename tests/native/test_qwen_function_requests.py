"""Host-only explicit family environment and exact padded prediction coverage."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import collect


class FunctionRequestsTests(unittest.TestCase):
    def test_default_environment_unchanged_and_explicit_family(self):
        base=collect.environment('runtime','stock',6144,fixed_program=True,ffn_activation='sigmoid-complement')
        self.assertNotIn('TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS',base)
        family=collect.environment('runtime','stock',6144,fixed_program=True,ffn_activation='sigmoid-complement',function_rows='512,1024,1536,2048')
        self.assertEqual({k:v for k,v in family.items() if k not in base},{'TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS':'512,1024,1536,2048'})
        for route,mode,fixed in (('gpu','stock',False),('runtime','runtime-layers',True),('runtime','stock',False)):
            with self.assertRaises(ValueError):collect.environment(route,mode,4096,fixed_program=fixed,function_rows='512,1024')

    def test_actual_rows_call_coverage_not_maximum_bucket_padding(self):
        def row(functions):return dict(hybrid=dict(runtime_weight=dict(function_rows=functions,row_alignment=512)))
        family=row([512,1024,1536,2048]);small=row([512,1024])
        self.assertEqual([collect.function_calls(family,n) for n in (1024,1048,2560,5144)],[1,1,2,3])
        self.assertEqual([collect.function_calls(small,n) for n in (1024,1048,2560,5144)],[1,2,3,6])
        self.assertEqual(collect.function_calls({},5144),6)
        with self.assertRaises(ValueError):collect.function_calls(row([512,2048]),1048)


if __name__=='__main__':unittest.main()
