"""CPU-only negative controls for fixed-share and multi-prompt audits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import joint_audit
import quality_audit
import quality_collect
import share_audit
import tiling_audit
from collect import sha256_file


class ShareQualityTests(unittest.TestCase):
    def test_completion_split_differences_are_per_request_not_session_totals(self):
        runs=[]
        for i in range(3):
            groups=dict(hybrid={},runtime_weight={},qwen_private_qkv={})
            for name,group,field,integer in share_audit.COMPLETION_COUNTERS:
                groups[group][field]=(i+1)*(100 if integer else 0.5)
            groups["hybrid"]["runtime_weight"]=groups["runtime_weight"]
            runs.append(dict(index=i,metrics=dict(hybrid=groups["hybrid"],qwen_private_qkv=groups["qwen_private_qkv"])))
        result=share_audit.completion_deltas(runs)
        for i,row in enumerate(result):
            self.assertEqual(row["index"],i)
            for name,_,_,integer in share_audit.COMPLETION_COUNTERS:
                self.assertEqual(row[name],100 if integer else 0.5)
        self.assertIn("must not be summed",share_audit.COMPLETION_SCOPE)
        with self.assertRaises(ValueError):share_audit.completion_deltas([])
        for name,group,field,integer in share_audit.COMPLETION_COUNTERS:
            for value in (-1,float("nan"),float("inf"),True,"1",None):
                changed=copy.deepcopy(runs)
                target=changed[1]["metrics"]["hybrid"][group] if group=="runtime_weight" else changed[1]["metrics"][group]
                target[field]=value
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    share_audit.completion_deltas(changed)
            changed=copy.deepcopy(runs)
            target=changed[1]["metrics"]["hybrid"][group] if group=="runtime_weight" else changed[1]["metrics"][group]
            del target[field]
            with self.subTest(missing=name),self.assertRaises(ValueError):share_audit.completion_deltas(changed)
            if integer:
                changed=copy.deepcopy(runs);changed[0]["metrics"][group][field]=1.5
                with self.subTest(fraction=name),self.assertRaises(ValueError):share_audit.completion_deltas(changed)
        for value in (True,-1,2):
            changed=copy.deepcopy(runs);changed[1]["index"]=value
            with self.subTest(index=value),self.assertRaises(ValueError):share_audit.completion_deltas(changed)

    def test_complement_share_controls_keep_fixed_owner_and_exact_replays(self):
        with tempfile.TemporaryDirectory(prefix="tc-complement-share-") as directory:
            records=[]
            for channels,mode in ((5120,"stock"),(6144,"stock"),(5120,"timeline")):
                root=Path(directory)/(str(channels)+mode);(root/"trial").mkdir(parents=True)
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],steps=40,output=str(root/f"{i}.png"))))
                records.append(dict(root=root,images=[np.zeros((2,2,4),dtype=np.uint8) for _ in range(3)],
                    summary=dict(case="generate",mode=mode,channels=channels,private_qkv=True,runtime_prefix_snapshot=False,
                        fixed_program=True,ffn_activation="sigmoid-complement",ffn_projection="descriptor",
                        environment=collect.environment("runtime",mode,channels,False,False,True,True,"descriptor","sigmoid-complement"),
                        assets=dict(library="same",manifest="same",reference="same"))))
            options=dict(fixed_program=True,activation="sigmoid-complement")
            share_audit.matched(*records[:2],**options)
            share_audit.replay(records[0],records[2],"timeline",**options)
            with self.assertRaises(ValueError):share_audit.matched(*records[:2])
            for key,value in (("fixed_program",False),("ffn_activation","exp"),("ffn_projection","single")):
                changed=copy.deepcopy(records[:2]);changed[1]["summary"][key]=value
                with self.assertRaises(ValueError):share_audit.matched(*changed,**options)
            changed=copy.deepcopy(records[:2]);changed[1]["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM"]="0"
            with self.assertRaises(ValueError):share_audit.matched(*changed,**options)
            changed=copy.deepcopy(records[2]);changed["images"][2][0,0,0]=1
            with self.assertRaises(ValueError):share_audit.replay(records[0],changed,"timeline",**options)

    def test_quality_recipe_covers_text_adult_portrait_materials_and_edit(self):
        tasks=quality_collect.TASKS
        self.assertEqual([r[0] for r in tasks],["poster","portrait","materials","matte-edit"])
        self.assertEqual([r[1] for r in tasks],["generate","generate","generate","edit"])
        self.assertIn("adult",tasks[1][2])
        self.assertTrue(all(0<len(r[2].encode())<=32768 for r in tasks))

    def test_only_ffn_share_changes_the_complete_request(self):
        with tempfile.TemporaryDirectory(prefix="tc-share-audit-") as directory:
            records=[]
            for channels in (5120,6144):
                root=Path(directory)/str(channels);(root/"trial").mkdir(parents=True)
                for i in range(3):(root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],steps=40,output=str(root/f"{i}.png"))))
                records.append(dict(root=root,summary=dict(case="generate",runtime_prefix_snapshot=False,private_qkv=True,
                    channels=channels,environment=collect.environment("runtime","stock",channels,False,False,True),
                    assets=dict(library="same",manifest="same",reference="same"))))
            share_audit.matched(*records)
            for key,value in (("private_qkv",False),("channels",5120),("case","edit"),("runtime_prefix_snapshot",True)):
                changed=copy.deepcopy(records);changed[1]["summary"][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):share_audit.matched(*changed)
            changed=copy.deepcopy(records);changed[1]["summary"]["assets"]["reference"]="other"
            with self.assertRaises(ValueError):share_audit.matched(*changed)
            changed=copy.deepcopy(records);changed[1]["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_PREFETCH"]="1"
            with self.assertRaises(ValueError):share_audit.matched(*changed)

    def test_audit_partition_is_explicit_and_rejects_unsupported_before_io(self):
        for channels in (True,0,8192,"5120"):
            with self.subTest(channels=channels),self.assertRaises(ValueError):
                joint_audit.window(Path("/synthetic"),None,None,True,"stock","generate",False,channels=channels)

    def test_quality_controls_change_only_execution_and_not_prompts(self):
        gpu=dict(model="qwen-image-2.1",operation="image.generate",prompt='Poster: "QWEN IMAGE"',seed=73,
                 width=512,height=512,steps=40,frames=1,audio=False,execution="gpu",allow_approximation=True,output="a.png")
        joint={**gpu,"execution":"gpu_ane","hybrid_mlp_mode":"runtime","ane_manifest":"fixed.json","output":"b.png"}
        quality_audit.matched_request(gpu,joint)
        for key,value in (("prompt","other prompt"),("seed",74),("steps",6),("width",1024),("allow_approximation",False),
                          ("execution","gpu"),("hybrid_mlp_mode","auto"),("ane_manifest","")):
            with self.subTest(key=key),self.assertRaises(ValueError):
                quality_audit.matched_request(gpu,{**joint,key:value})

    def test_k_split_changes_only_the_pinned_descriptor_field(self):
        with tempfile.TemporaryDirectory(prefix="tc-k-split-audit-") as directory:
            records=[]
            for tile in (4096,1024):
                root=Path(directory)/str(tile);(root/"trial").mkdir(parents=True)
                manifest=root/"shape.json"
                manifest.write_text(json.dumps(dict(rows=1024,hidden=4096,width=12288,tile_k=tile,tile_n=512)))
                for i in range(3):
                    (root/f"trial/request-{i}.json").write_text(json.dumps(dict(seed=(73,74,73)[i],steps=40,ane_manifest=str(manifest),output=str(root/f"{i}.png"))))
                records.append(dict(root=root,summary=dict(case="generate",channels=6144,private_qkv=True,
                    runtime_prefix_snapshot=False,environment=collect.environment("runtime","stock",6144,False,False,True),
                    assets=dict(library="same",reference="same",manifest=sha256_file(manifest)))))
            self.assertEqual([r["tile_k"] for r in tiling_audit.matched(*records,1024)],[4096,1024])
            with self.assertRaises(ValueError):tiling_audit.matched(*records,512)
            records[1]["summary"]["environment"]["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="5120"
            with self.assertRaises(ValueError):tiling_audit.matched(*records,1024)


if __name__=="__main__":unittest.main()
