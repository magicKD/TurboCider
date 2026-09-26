#!/usr/bin/env python3
"""Acceptance inventory fixtures, not real product acceptance evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/native'))
import verify_streaming_acceptance as acceptance
from streaming_release_policy import freeze, CALIBRATED, canonical


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.frozen=freeze(CALIBRATED);self.binding=dict(release_policy_revision=CALIBRATED,workload='fixture')
        self.base=dict(policy_sha256=self.frozen['sha256'],catalog_binding_sha256=hashlib.sha256(canonical(self.binding)).hexdigest(),reviewed_commit='a'*40)
        (self.root/'raw.log').write_text('fixture test trace, not actual model evidence')
        self.reports={}
        for gate, checks in acceptance.CHECKS.items():
            report=dict(schema=acceptance.REPORT_SCHEMA,gate=gate,**self.base,
                        checks={c:dict(verdict='PASS',artifacts=[self.ref('raw.log')]) for c in checks})
            self.reports[gate]=report
        self.save()

    def ref(self,path):
        data=(self.root/path).read_bytes()
        return dict(path=path,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))

    def save(self):
        for gate,report in self.reports.items():(self.root/(gate+'.json')).write_text(json.dumps(report))
        review=dict(schema=acceptance.SCHEMA,status='approved',**self.base,
                    reports={g:self.ref(g+'.json') for g in acceptance.CHECKS},
                    reviewers={r:'fixture reviewer' for r in ('runtime','model','performance','release')})
        review['review_digest']=hashlib.sha256(canonical(review)).hexdigest()
        (self.root/'review.json').write_text(json.dumps(review))

    def verify(self): return acceptance.verify(self.root,self.binding,self.frozen)

    @unittest.skipUnless(sys.platform == 'darwin', 'P2 uses Darwin process sampling')
    def test_reverify_real_synthetic_campaigns_and_reject_changed_summary(self):
        import verify_streaming_release_evidence as release
        from test_streaming_campaign_verifier import CampaignTests, policy, passed_audit
        self.binding = dict(release_policy_revision=CALIBRATED, **{k:{} for k in ('source','workload','runtime','device','plan','performance_profile')})
        self.base['catalog_binding_sha256'] = hashlib.sha256(canonical(self.binding)).hexdigest()
        for report in self.reports.values(): report.update(self.base)
        self.save()
        bundles = {}
        for gate in ('P0','P1','P2'):
            campaign = policy(); campaign['comparison_kind'] = gate
            campaign['catalog_binding'] = copy.deepcopy(self.binding)
            for variant in campaign['variants'].values():
                variant['source_identity'] = dict(commit='a'*40, source_manifest_sha256='b'*64, clean=True)
            if gate == 'P0':
                campaign.pop('semantic_equivalence', None)
                campaign['thresholds'] = {'P0_legacy': dict(wall_median_ratio_max=1.02, wall_p95_ratio_max=1.05,
                    denoise_median_ratio_max=1.02, **{k:v for k,v in passed_audit().items() if k.startswith('new_')})}
            if gate == 'P2':
                campaign['protocol']['restart_workers_between_blocks'] = True
                campaign['memory_sampling'] = dict(enabled=True, interval_ms=5, max_gap_ms=100,
                    required_variants=['candidate'], target_bytes=8<<30, headroom_policy_revision='tc-public-headroom-v1')
            bundles[gate] = CampaignTests().run_bundle(campaign)
            self.addCleanup(shutil.rmtree, bundles[gate].parent)
        result = release.verify(self.frozen,self.binding,bundles,self.root)
        self.assertEqual(result['assessment']['status'],'READY_FOR_REVIEW')
        self.assertEqual(result['assessment']['p3']['verdict'],'NOT_RUN')
        self.assertFalse(result['production_authorized'])
        self.reports['app']['checks']['restart_recovery']['verdict']='FAIL';self.save()
        blocked=release.verify(self.frozen,self.binding,bundles,self.root)
        self.assertEqual(blocked['assessment']['status'],'BLOCKED')
        self.assertIn('app',blocked['assessment']['blocking_gates'])
        self.reports['app']['checks']['restart_recovery']['verdict']='PASS';self.save()
        path=bundles['P1']/'summary.json';summary=json.loads(path.read_text());summary['matched_pairs']+=1
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'persisted summary'):
            release.verify(self.frozen,self.binding,bundles,self.root)

    def test_complete_bound_review_with_raw_inventory(self):
        r=self.verify();self.assertTrue(all(v=='PASS' for v in r['verdicts'].values()));self.assertFalse(r['production_authorized'])
        self.assertEqual(len(r['artifacts']),8)

    def test_fifo_and_oversized_manifest_rejected_without_blocking(self):
        import verify_streaming_release_evidence as release
        path=self.root/'fifo';os.mkfifo(path)
        with self.assertRaises(acceptance.AcceptanceError):release.read(path)
        path=self.root/'large.json';path.write_bytes(b' '*((1<<20)+1))
        with self.assertRaises(acceptance.AcceptanceError):release.read(path)

    def test_raw_mutation_rejected(self):
        (self.root/'raw.log').write_text('changed')
        with self.assertRaisesRegex(acceptance.AcceptanceError,'digest or size'):self.verify()

    def test_report_mutation_rejected_even_though_pass(self):
        (self.root/'app.json').write_text('{}')
        with self.assertRaises(acceptance.AcceptanceError):self.verify()

    def test_missing_coverage_and_status_only_rejected(self):
        del self.reports['app']['checks']['restart_recovery'];self.save()
        with self.assertRaisesRegex(acceptance.AcceptanceError,'checks'):self.verify()
        self.reports['app']['checks']['restart_recovery']=dict(verdict='PASS',artifacts=[]);self.save()
        with self.assertRaisesRegex(acceptance.AcceptanceError,'no raw evidence'):self.verify()

    def test_failure_and_not_run_are_not_upgraded(self):
        self.reports['lifecycle']['checks']['unsafe_drain_quarantine']['verdict']='FAIL'
        self.reports['quality']['checks']['two_prompts_three_seeds']=dict(verdict='NOT_RUN',artifacts=[]);self.save()
        r=self.verify();self.assertEqual(r['verdicts']['lifecycle'],'FAIL');self.assertEqual(r['verdicts']['quality'],'NOT_RUN')

    def test_report_binding_cannot_be_reviewed_for_another_commit(self):
        self.reports['app']['reviewed_commit']='b'*40;self.save()
        with self.assertRaisesRegex(acceptance.AcceptanceError,'binding'):self.verify()

    def test_symlinks_and_traversal_rejected(self):
        (self.root/'link').symlink_to(self.root/'raw.log')
        for path in ['link','../raw.log','/tmp/raw.log','./raw.log','bad//raw.log']:
            self.reports['app']['checks']['worker_identity']['artifacts'][0]['path']=path;self.save()
            with self.assertRaises(acceptance.AcceptanceError):self.verify()

    def test_duplicate_keys_and_boolean_size_rejected(self):
        with self.assertRaises(acceptance.AcceptanceError): acceptance.Inventory.decode(b'{"status":1,"status":2}')
        self.reports['app']['checks']['worker_identity']['artifacts'][0]['bytes']=True;self.save()
        with self.assertRaises(acceptance.AcceptanceError):self.verify()


if __name__=='__main__':unittest.main(verbosity=2)
