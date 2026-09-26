#!/usr/bin/env python3
"""Reverify calibrated campaigns and reviewed acceptance before catalog integration.

Returns review readiness only. This does not create or install a public record.
"""
import argparse
import json
from pathlib import Path
from streaming_release_policy import canonical, assess_calibrated, validate_frozen
from verify_streaming_acceptance import Inventory, verify as verify_acceptance
from verify_streaming_campaign import verify as verify_campaign
from build_streaming_catalog import _validate_commit_identity, _validate_p2_summary


def read_evidence(path):
    inventory = Inventory(path.parent.resolve())
    try:
        data, observed = inventory.read(path.name, json_limit=True)
        return Inventory.decode(data), observed['sha256']
    finally:
        inventory.close()


def read(path):
    return read_evidence(path)[0]


def verify(frozen, binding, bundles, acceptance):
    validate_frozen(frozen)
    if not isinstance(binding, dict) or set(binding) != {'release_policy_revision','source','workload','runtime','device','plan','performance_profile'}:
        raise ValueError('incomplete catalog binding')
    if set(bundles) not in ({'P0','P1','P2'}, {'P0','P1','P2','P3'}):
        raise ValueError('release requires P0/P1/P2, with optional P3')
    accepted = verify_acceptance(acceptance, binding, frozen)
    summaries, policies, evidence = {}, {}, {}
    for gate, bundle in bundles.items():
        policy, policy_digest = read_evidence(bundle/'campaign-policy.json')
        if policy.get('comparison_kind') != gate or canonical(policy.get('catalog_binding')) != canonical(binding):
            raise ValueError('campaign policy binding differs')
        independently_verified = verify_campaign(bundle)
        summary, summary_digest = read_evidence(bundle/'summary.json')
        if canonical(summary) != canonical(independently_verified) or summary.get('policy_sha256') != policy_digest:
            raise ValueError('persisted summary differs from independent verifier')
        _validate_commit_identity(summary, accepted['reviewed_commit'], gate, True)
        if gate == 'P2' and summary.get('overall') == 'PASS':
            _validate_p2_summary(bundle, summary, policy)
        summaries[gate] = summary
        policies[gate] = policy['catalog_binding']
        evidence[gate] = dict(summary_sha256=summary_digest, policy_sha256=policy_digest)
    assessment = assess_calibrated(frozen, {g:summaries[g] for g in ('P0','P1','P2')}, accepted['verdicts'], summaries.get('P3'),
                                  campaign_bindings={g:policies[g] for g in ('P0','P1','P2')})
    return dict(schema='tc-streaming-release-evidence-verification-v1', assessment=assessment,
                acceptance=accepted, campaign_evidence=evidence, campaign_summaries=summaries,
                production_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-policy','binding','p0','p1','p2','acceptance','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--p3', type=Path)
    args = parser.parse_args()
    try:
        bundles = {g:getattr(args,g.lower()) for g in ('P0','P1','P2')}
        if args.p3:
            bundles['P3'] = args.p3
        result = verify(read(args.frozen_policy), read(args.binding), bundles, args.acceptance)
        code = 0 if result['assessment']['status'] == 'READY_FOR_REVIEW' else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = dict(status='INVALID', error=str(exc), production_authorized=False)
        code = 2
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
