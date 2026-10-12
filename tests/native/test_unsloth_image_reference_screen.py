from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "tools/validation"))
from unsloth_image_reference_screen import placement,request,sampler_roles,validate_generation


class UnslothImageReferenceTests(unittest.TestCase):
    def log(self):
        return "\r".join(f"|==> | {i}/8 - 1.00s/it" for i in range(1,9))+"\nget_learned_condition completed, taking 1.0s\nsampling completed, taking 8.0s\ngenerate_image completed in 9.1s"

    def test_family_steps_and_complete_actual_progress(self):
        payload = request("z-image-turbo","fox")
        self.assertEqual(payload["steps"],8)
        self.assertEqual(request("flux2-klein-4b","fox")["steps"],4)
        self.assertEqual(validate_generation(payload,payload,self.log())["sampling_seconds"],8.)

    def test_echo_only_missing_steps_wrong_seed_and_duplicate_steps_rejected(self):
        payload = request("z-image-turbo","fox")
        for info,log in ((dict(payload,seed=7),self.log()),(payload,self.log().replace("8/8","7/8")),
                         (payload,self.log()+"\r|==> | 8/8 - 1.00s/it")):
            with self.assertRaises(ValueError):
                validate_generation(info,payload,log)

    def test_gpu_text_cpu_text_and_dit_placement_must_be_actual(self):
        gpu = "Conditioner params100 -> compute MTL0, params MTL0\nDiT params100 -> compute MTL0, params MTL0"
        placement(gpu,"gpu")
        cpu = gpu.replace("Conditioner params100 -> compute MTL0, params MTL0","Conditioner params100 -> compute CPU, params cpu")
        placement(cpu,"default")
        with self.assertRaises(ValueError):
            placement(cpu,"gpu")
        with self.assertRaises(ValueError):
            placement(gpu.replace("DiT","unrelated"),"gpu")

    def test_owned_listener_checks_are_declared_children_not_unknown_processes(self):
        self.assertEqual(sampler_roles(Path("/isolated/sd-server")),
                         (("reference_inference","/isolated/sd-server"),("listener_probe","/usr/sbin/lsof")))


if __name__ == "__main__":
    unittest.main()
