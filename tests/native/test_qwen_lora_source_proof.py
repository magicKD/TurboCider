import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("source_proof",ROOT/"tools/validation/qwen_lora_source_proof.py")
PROOF=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PROOF)


class SourceProofTests(unittest.TestCase):
    def rows(self):
        return [dict(bytes_read=1234 if i==0 else 0,native_cache_hits=0 if i==0 else 1,source_files=1,
                     bound_this_request=i==0,scope=PROOF.SCOPE) for i in range(3)]
    def lines(self,rows):return [json.dumps(dict(qwen_lora_source_verification=r)) for r in rows]
    def test_native_cold_and_warm_source_progress(self):
        rows=self.rows();before=copy.deepcopy(rows)
        self.assertEqual(PROOF.validate_source_proof(["unrelated diagnostic",*self.lines(rows)],1234),rows)
        self.assertEqual(rows,before)
    def test_metadata_omitted_hash_changed_source_or_false_counts_cannot_pass(self):
        for index,key,value in ((0,"bytes_read",0),(1,"bytes_read",1234),(1,"native_cache_hits",0),
            (1,"native_cache_hits",True),(1,"native_cache_hits",2),(1,"bound_this_request",True),
            (0,"bound_this_request",1),(0,"source_files",0),(1,"scope","metadata-only")):
            rows=self.rows();rows[index][key]=value
            with self.subTest(index=index,key=key,value=value),self.assertRaises(ValueError):
                PROOF.validate_source_proof(self.lines(rows),1234)
        for rows in ([],self.rows()[:2],self.rows()*2):
            with self.assertRaises(ValueError):PROOF.validate_source_proof(self.lines(rows),1234)
        with self.assertRaises(ValueError):PROOF.validate_source_proof(self.lines(self.rows()),True)
        with self.assertRaises(ValueError):PROOF.validate_source_proof(["qwen_lora_source_verification not JSON"],1234)


if __name__=="__main__":unittest.main()
