"""Explicit release gates and claims; policy decisions never grant runtime authority.

The calibrated contract is preparatory. Production record/channel support must
also be enabled in builder, native validation, packaging and UI before use.
Legacy records retain the strict P0-P3 public contract.
"""
from __future__ import annotations
import hashlib
import json

STRICT = 'tc-public-strict-v1'
CALIBRATED = 'tc-public-calibrated-v1'
POLICY_SCHEMA = 'tc-streaming-release-policy-contract-v1'
STAGING_GATES = ('P0', 'P1', 'P2')
PUBLIC_GATES = (*STAGING_GATES, 'P3')
ACCEPTANCE_GATES = ('source_actual', 'quality', 'lifecycle', 'app', 'installation', 'package_revocation')


class ReleasePolicyError(ValueError):
    pass


def contract(revision: str) -> dict:
    if revision not in (STRICT, CALIBRATED):
        raise ReleasePolicyError('unsupported release policy revision')
    calibrated = revision == CALIBRATED
    return dict(schema=POLICY_SCHEMA, policy_revision=revision,
        public_channels=['public-calibrated'] if calibrated else ['public-stable', 'public-experimental'],
        required_campaign_gates=list(STAGING_GATES if calibrated else PUBLIC_GATES),
        required_acceptance_gates=list(ACCEPTANCE_GATES) if calibrated else [],
        optional_campaign_gates=['P3'] if calibrated else [],
        claims=dict(calibrated_scope_only=calibrated, hard_memory_cap=False,
                    guaranteed_zero_swap=False, swap_speedup_requires_p3=True),
        p3_safety_failures_block_release=True)


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def freeze(revision: str) -> dict:
    value = contract(revision)
    return dict(contract=value, sha256=hashlib.sha256(canonical(value)).hexdigest())


def validate_frozen(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {'contract', 'sha256'} or not isinstance(value['contract'], dict):
        raise ReleasePolicyError('invalid frozen release policy envelope')
    expected = freeze(value['contract'].get('policy_revision'))
    if canonical(value) != canonical(expected):
        raise ReleasePolicyError('release policy contents or digest differ from supported contract')
    return expected['contract']


def required_gates(channel: str, revision: str | None = None) -> tuple[str, ...]:
    if channel == 'staging':
        if revision not in (None, STRICT):
            raise ReleasePolicyError('staging cannot stand in for calibrated public qualification')
        return STAGING_GATES
    selected = contract(revision or STRICT)
    if channel not in selected['public_channels']:
        raise ReleasePolicyError('release channel and policy do not match')
    return tuple(selected['required_campaign_gates'])


def p3_observation(summary: dict | None) -> dict:
    """Keep an independently verified P3 verdict; never rewrite it to PASS.

    No-summary means NOT_RUN, with no swap claim. A supplied summary must contain
    explicit reliability fields; unknown/missing reliability is blocking too.
    The caller still has to independently verify the raw campaign and binding.
    """
    if summary is None:
        return dict(verdict='NOT_RUN', safety_blocked=False, swap_speedup=False)
    if not isinstance(summary, dict) or summary.get('comparison_kind') != 'P3' or summary.get('overall') not in ('PASS', 'FAIL', 'INCONCLUSIVE'):
        raise ReleasePolicyError('invalid P3 verifier summary')
    safety_keys = ('hard_failure_samples', 'quality_failures', 'audit_passed', 'source_provenance_complete', 'quality_status', 'memory_evidence')
    if any(key not in summary for key in safety_keys):
        raise ReleasePolicyError('P3 reliability evidence is incomplete')
    if not isinstance(summary['hard_failure_samples'], list) or not isinstance(summary['quality_failures'], list):
        raise ReleasePolicyError('P3 failure evidence has wrong type')
    memory = summary['memory_evidence']
    if not isinstance(memory, dict) or memory.get('qualification') not in ('PASS', 'FAIL', 'INCONCLUSIVE'):
        raise ReleasePolicyError('P3 memory reliability evidence is incomplete')
    blocked = (memory['qualification'] == 'FAIL' or bool(summary['hard_failure_samples']) or bool(summary['quality_failures']) or
               summary['audit_passed'] is not True or summary['source_provenance_complete'] is not True or
               summary['quality_status'] != 'complete')
    p3 = summary.get('p3_result')
    # A PASS label alone cannot become a speedup claim. Require the verifier's
    # natural-swap evidence and explicit passing speedup classification.
    speedup = (not blocked and memory['qualification'] == 'PASS' and summary['overall'] == 'PASS' and isinstance(p3, dict) and
               p3.get('qualification') == 'PASS' and p3.get('swap_status') == 'PASS' and
               p3.get('speedup_claim_qualified') is True and p3.get('classification') == 'faster_and_lower_swap')
    return dict(verdict=summary['overall'], safety_blocked=blocked, swap_speedup=speedup)


def assess_calibrated(frozen: dict, campaign_summaries: dict, acceptance_verdicts: dict, p3: dict | None = None, *, campaign_bindings: dict) -> dict:
    """Aggregate verified inputs only; this is not an evidence loader or builder.

    Callers must verify original evidence and reviewer bindings. Returning ready
    does not construct a record, install a catalog or authorize execution.
    """
    policy = validate_frozen(frozen)
    if policy['policy_revision'] != CALIBRATED:
        raise ReleasePolicyError('calibrated assessment requires its explicit frozen policy')
    if set(campaign_summaries) != set(STAGING_GATES):
        raise ReleasePolicyError('calibrated requires exactly P0, P1 and P2 summaries')
    if set(campaign_bindings) != set(STAGING_GATES):
        raise ReleasePolicyError('campaign policy bindings are required')
    reference = campaign_bindings['P0']
    for binding in campaign_bindings.values():
        if (not isinstance(binding, dict) or binding.get('release_policy_revision') != CALIBRATED or
                canonical(binding) != canonical(reference)):
            raise ReleasePolicyError('campaign bindings must match the explicit calibrated policy and each other')
    failures = []
    for gate in STAGING_GATES:
        value = campaign_summaries[gate]
        if not isinstance(value, dict) or value.get('comparison_kind') != gate:
            raise ReleasePolicyError('campaign kind mismatch')
        if value.get('overall') != 'PASS':
            failures.append(gate)
    if set(acceptance_verdicts) != set(ACCEPTANCE_GATES):
        raise ReleasePolicyError('calibrated acceptance evidence set is incomplete')
    for gate, verdict in acceptance_verdicts.items():
        if verdict not in ('PASS', 'FAIL', 'INCONCLUSIVE', 'NOT_RUN'):
            raise ReleasePolicyError('invalid acceptance verdict')
        if verdict != 'PASS':
            failures.append(gate)
    observation = p3_observation(p3)
    if observation['safety_blocked']:
        failures.append('P3_reliability')
    return dict(status='BLOCKED' if failures else 'READY_FOR_REVIEW', blocking_gates=failures,
                policy_sha256=frozen['sha256'], p3=observation,
                claims=dict(policy['claims'], swap_speedup=False), production_authorized=False)
