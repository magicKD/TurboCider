from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_spatial_decode_screen import validate_projection_observation


class CompleteVAEProjectionGuards(unittest.TestCase):
    def observation(self):
        return dict(vae_upscale_recipe="coreml_projection_probe",projection_backend="ne",projection_frames=25,
            projection_coreml_calls=1600,projection_backing_hits=1600,projection_slot_bytes=21037056,
            qualified=False,physical_ane_occupancy_measured=False)

    def test_all_frames_and_tiles_are_required(self):
        validate_projection_observation(self.observation(),"ne",dict(rows=64,cols=64),25,1024,1024)
        for key,value in (("vae_upscale_recipe","native"),("projection_backend","cpu"),("projection_frames",1),
            ("projection_coreml_calls",64),("projection_backing_hits",1599),("projection_slot_bytes",0),
            ("qualified",True),("physical_ane_occupancy_measured",True)):
            observation=self.observation();observation[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_projection_observation(observation,"ne",dict(rows=64,cols=64),25,1024,1024)

    def test_cpu_control_is_not_relabelled_as_ne(self):
        observation=self.observation();observation["projection_backend"]="cpu"
        validate_projection_observation(observation,"cpu",dict(rows=64,cols=64),25,1024,1024)

    def test_actual_shader_and_probe_are_release_guarded(self):
        source=(ROOT/"tools/native/check_release_binary.py").read_text()
        self.assertIn("tc_seedvr2_ane_projection_pack",source)
        self.assertIn("coreml_projection_probe",source)
        runtime=(ROOT/"tools/native/seedvr2_vae_ane_projection.mm").read_text()
        self.assertNotIn("output.dataPointer",runtime)
        self.assertIn("output.pixelBuffer==slot.output.pixel",runtime)


if __name__=="__main__":unittest.main()
