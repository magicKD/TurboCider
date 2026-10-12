import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "tools/validation"))
from z_unsloth_framework_screen import ORDER,ROUTES,compiled_source_files,validate_cohort,validate_trial


class ZUnslothProtocolTests(unittest.TestCase):
    def test_compiler_locks_are_not_model_parameters(self):
        with tempfile.TemporaryDirectory(prefix="tc-z-compiled-source-") as raw:
            root=Path(raw)
            (root / "cache.lock").touch()
            weights=root / "weights.bin";weights.write_bytes(b"weights")
            self.assertEqual(compiled_source_files(root),[weights])

    def trial(self):
        return dict(route="unsloth-gpu-text",memory=dict(verified=dict(complete=True)),
            report=dict(status="complete_precheck_only",server_terminal=True,
                runs=[dict(index=i,prompt_is_repeat=i==3,request_wall_including_export=1.) for i in range(4)]))

    def test_reference_successful_server_and_complete_memory_required(self):
        validate_trial(self.trial())
        for edit in (lambda t:t["memory"]["verified"].update(complete=False),
                     lambda t:t["report"].update(server_terminal=False),
                     lambda t:t["report"].update(status="failed")):
            trial=self.trial();edit(trial)
            with self.assertRaises(ValueError):validate_trial(trial)

    def test_short_or_invalid_reference_wall_cannot_be_a_complete_trial(self):
        trial=self.trial();trial["report"]["runs"].pop()
        with self.assertRaises(ValueError):validate_trial(trial)
        for value in (0,True,float("nan")):
            trial=self.trial();trial["report"]["runs"][0]["request_wall_including_export"]=value
            with self.assertRaises(ValueError):validate_trial(trial)

    def test_full_five_route_order_and_own_samples_are_not_optional(self):
        data=dict(status="complete_diagnostic",order=list(ORDER),
            trials=[dict(index=i,route=route,report=dict(runs=[dict(request_wall_including_export=1.)]*4)) for i,route in enumerate(ORDER)],
            medians={route:dict(samples=[1.]*4,fresh_warm_seconds=1.) for route in ROUTES})
        with patch("z_unsloth_framework_screen.validate_trial"):
            validate_cohort(data)
            wrong=copy.deepcopy(data);wrong["order"]=list(reversed(ROUTES))
            with self.assertRaises(ValueError):validate_cohort(wrong)
            wrong=copy.deepcopy(data);wrong["medians"][ROUTES[0]]["fresh_warm_seconds"]=.01
            with self.assertRaises(ValueError):validate_cohort(wrong)


if __name__ == "__main__":unittest.main()
