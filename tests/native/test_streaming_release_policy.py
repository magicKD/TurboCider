#!/usr/bin/env python3
"""Policy decisions on already-verified inputs; no model qualification claims."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/native'))
import streaming_release_policy as policy
import build_streaming_catalog as builder


def p3(verdict='INCONCLUSIVE'):
    return dict(comparison_kind='P3', overall=verdict, hard_failure_samples=[], quality_failures=[],
        audit_passed=True, source_provenance_complete=True, quality_status='complete',
        memory_evidence=dict(qualification='PASS'))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.frozen = policy.freeze(policy.CALIBRATED)
        self.campaigns = {g: dict(comparison_kind=g, overall='PASS') for g in policy.STAGING_GATES}
        self.acceptance = {g: 'PASS' for g in policy.ACCEPTANCE_GATES}
        self.bindings = {g: dict(release_policy_revision=policy.CALIBRATED, workload='fixture') for g in policy.STAGING_GATES}

    def assess(self, frozen, campaigns, acceptance, p3=None):
        return policy.assess_calibrated(frozen, campaigns, acceptance, p3, campaign_bindings=self.bindings)

    def test_frozen_contract_golden_fixtures(self):
        root = Path(__file__).resolve().parents[1]/'fixtures/streaming'
        for name, revision in [('strict',policy.STRICT),('calibrated',policy.CALIBRATED)]:
            frozen = json.loads((root/f'release-policy-{name}-v1.json').read_text())
            self.assertEqual(policy.validate_frozen(frozen), policy.contract(revision))

    def test_legacy_strict_and_explicit_channels(self):
        for channel in ['public-stable', 'public-experimental']:
            self.assertEqual(policy.required_gates(channel), ('P0','P1','P2','P3'))
            self.assertEqual(policy.required_gates(channel, policy.STRICT), policy.PUBLIC_GATES)
            with self.assertRaises(policy.ReleasePolicyError): policy.required_gates(channel, policy.CALIBRATED)
        self.assertEqual(policy.required_gates('public-calibrated', policy.CALIBRATED), policy.STAGING_GATES)
        with self.assertRaises(policy.ReleasePolicyError): policy.required_gates('public-calibrated')
        with self.assertRaises(policy.ReleasePolicyError): policy.freeze('unknown')

    def test_canonical_freeze_rejects_tampering_and_bool_integer_alias(self):
        self.assertEqual(policy.validate_frozen(self.frozen), policy.contract(policy.CALIBRATED))
        for key, value in [('hard_memory_cap', True), ('hard_memory_cap', 0), ('swap_speedup_requires_p3', False)]:
            modified = copy.deepcopy(self.frozen); modified['contract']['claims'][key] = value
            with self.assertRaises(policy.ReleasePolicyError): policy.validate_frozen(modified)
        modified = copy.deepcopy(self.frozen); modified['contract']['required_campaign_gates'].remove('P2')
        with self.assertRaises(policy.ReleasePolicyError): policy.validate_frozen(modified)

    def test_p2_and_all_acceptance_remain_required(self):
        del self.campaigns['P2']
        with self.assertRaises(policy.ReleasePolicyError): self.assess(self.frozen, self.campaigns, self.acceptance)
        self.campaigns['P2'] = dict(comparison_kind='P2', overall='INCONCLUSIVE')
        self.assertIn('P2', self.assess(self.frozen, self.campaigns, self.acceptance)['blocking_gates'])
        self.campaigns['P2']['overall'] = 'PASS'
        for gate in policy.ACCEPTANCE_GATES:
            modified = dict(self.acceptance); modified[gate] = 'NOT_RUN'
            self.assertIn(gate, self.assess(self.frozen, self.campaigns, modified)['blocking_gates'])

    def test_not_run_and_performance_verdict_preserved_without_claim(self):
        for supplied, expected in [(None,'NOT_RUN'), (p3(),'INCONCLUSIVE'), (p3('FAIL'),'FAIL')]:
            result = self.assess(self.frozen, self.campaigns, self.acceptance, supplied)
            self.assertEqual(result['status'], 'READY_FOR_REVIEW')
            self.assertEqual(result['p3']['verdict'], expected)
            self.assertFalse(result['claims']['swap_speedup'])
            self.assertFalse(result['production_authorized'])

    def test_p3_correctness_source_lifecycle_failures_block(self):
        for key, value in [('hard_failure_samples',[{'status':'worker_error'}]), ('quality_failures',['pair']),
                           ('audit_passed',False), ('source_provenance_complete',False), ('quality_status','partial'),
                           ('memory_evidence', {'qualification':'FAIL'})]:
            bad=p3('FAIL');bad[key]=value
            result=self.assess(self.frozen,self.campaigns,self.acceptance,bad)
            self.assertEqual(result['status'],'BLOCKED')
            self.assertIn('P3_reliability',result['blocking_gates'])
        unknown=p3();del unknown['hard_failure_samples']
        with self.assertRaises(policy.ReleasePolicyError):policy.p3_observation(unknown)

    def test_strict_evidence_cannot_be_relabelled_calibrated(self):
        for gate in policy.STAGING_GATES:
            saved = dict(self.bindings[gate])
            self.bindings[gate]['release_policy_revision'] = policy.STRICT
            with self.assertRaises(policy.ReleasePolicyError):
                self.assess(self.frozen,self.campaigns,self.acceptance)
            self.bindings[gate] = saved
        self.bindings['P1']['workload'] = 'other'
        with self.assertRaises(policy.ReleasePolicyError):
            self.assess(self.frozen,self.campaigns,self.acceptance)

    def test_only_verified_p3_can_describe_speedup(self):
        value=p3('PASS')
        self.assertFalse(policy.p3_observation(value)['swap_speedup'])
        value['p3_result']=dict(qualification='PASS', swap_status='PASS', speedup_claim_qualified=True, classification='faster_and_lower_swap')
        self.assertTrue(policy.p3_observation(value)['swap_speedup'])
        result=self.assess(self.frozen,self.campaigns,self.acceptance,value)
        self.assertFalse(result['claims']['swap_speedup']) # calibrated policy itself makes no swap-speedup promise
        value['quality_failures']=['pair']
        self.assertFalse(policy.p3_observation(value)['swap_speedup'])


if __name__ == '__main__': unittest.main(verbosity=2)
