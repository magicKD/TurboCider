from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from compile_flux_manifest import validate_manifest


class FluxCompileManifestTests(unittest.TestCase):
    def fixture(self):
        return dict(schema_version=2,artifacts={str(i):dict(int8_pc=f"block{i}.mlpackage") for i in range(24)},
            shape=dict(K=4096,N=4096,mlp_width=12288),
            source=dict(blocks=list(range(24)),checkpoint="model.safetensors.index.json",
                checkpoint_shards={"part1.safetensors":dict(bytes=10,sha256="a"*64)}))

    def test_actual9b_not_relabelled4b(self):
        validate_manifest(self.fixture(),"flux2-klein-9b")
        with self.assertRaises(ValueError):validate_manifest(self.fixture(),"flux2-klein-4b")

    def test_missing_partition_or_sharded_provenance_rejected(self):
        for key,value in (("artifacts",{}),("source",dict(blocks=list(range(24)),checkpoint="model.safetensors.index.json"))):
            data=self.fixture();data[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_manifest(data,"flux2-klein-9b")


if __name__=="__main__":unittest.main()
