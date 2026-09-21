"""Verify reviewed acceptance reports and their raw artifact inventory.

Campaign math remains independently verified by verify_streaming_campaign.
Non-campaign App/installation/lifecycle checks require human review of the raw
artifacts: this validates that review's complete, exact binding, not a new
oracle for arbitrary test logs. Reviewer names and digests are not digital
signatures or authentication. It never authorizes a production catalog.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from streaming_release_policy import CALIBRATED, ACCEPTANCE_GATES, canonical, validate_frozen

SCHEMA = 'tc-streaming-acceptance-review-v1'
REPORT_SCHEMA = 'tc-streaming-acceptance-report-v1'
CHECKS = {
    'source_actual': ('all_sources_verified', 'actual_receipt_verified'),
    'quality': ('same_plan_reference', 'two_prompts_three_seeds'),
    'lifecycle': ('success_cleanup', 'cancel_cleanup', 'recoverable_failure', 'unsafe_drain_quarantine'),
    'app': ('intent_durable', 'worker_identity', 'terminal_artifact_binding', 'restart_recovery'),
    'installation': ('second_installation', 'source_mutation_rejected'),
    'package_revocation': ('packaged_catalog', 'revoked_record_rejected'),
}
VERDICTS = ('PASS', 'NOT_RUN', 'INCONCLUSIVE', 'FAIL')


class AcceptanceError(ValueError):
    pass


def exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise AcceptanceError(f'{label}: missing or unknown fields')


def digest_text(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise AcceptanceError('duplicate JSON key')
        result[key] = value
    return result


class Inventory:
    def __init__(self, root: Path):
        self.root = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.verified = {}

    def close(self):
        os.close(self.root)

    def read(self, relative: str, *, json_limit: bool):
        if (not isinstance(relative, str) or not relative or '\\' in relative or '\0' in relative or
                PurePosixPath(relative).is_absolute() or any(x in ('', '.', '..') for x in relative.split('/'))):
            raise AcceptanceError('artifact path must be normalized and relative')
        directory = os.dup(self.root)
        fd = None
        try:
            parts = relative.split('/')
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or (json_limit and before.st_size > 1 << 20):
                raise AcceptanceError('artifact must be a nonempty regular file; JSON is limited to 1 MiB')
            digest = hashlib.sha256()
            data = bytearray()
            count = 0
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                count += len(block)
                if count > before.st_size:
                    raise AcceptanceError('artifact grew during verification')
                if json_limit and count > 1 << 20:
                    raise AcceptanceError('JSON grew beyond limit')
                digest.update(block)
                if json_limit:
                    data.extend(block)
            after = os.fstat(fd)
            stable = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if stable(before) != stable(after) or count != before.st_size:
                raise AcceptanceError('artifact changed during verification')
            observed = dict(path=relative, sha256=digest.hexdigest(), bytes=count)
            if relative in self.verified and self.verified[relative] != observed:
                raise AcceptanceError('artifact has conflicting identities')
            self.verified[relative] = observed
            return bytes(data), observed
        except OSError as exc:
            raise AcceptanceError(f'cannot read artifact {relative}: {exc}') from exc
        finally:
            if fd is not None:
                os.close(fd)
            os.close(directory)

    def referenced(self, reference, *, json_limit=False):
        exact(reference, ('path','sha256','bytes'), 'artifact reference')
        if not digest_text(reference['sha256']) or type(reference['bytes']) is not int or reference['bytes'] <= 0:
            raise AcceptanceError('invalid artifact digest or size')
        data, observed = self.read(reference['path'], json_limit=json_limit)
        if observed != reference: raise AcceptanceError('artifact digest or size differs')
        return self.decode(data) if json_limit else None

    @staticmethod
    def decode(data):
        def bad_constant(value): raise AcceptanceError(f'non-finite JSON number: {value}')
        return json.loads(data, object_pairs_hook=unique_object, parse_constant=bad_constant)


def verify(bundle: Path, binding: dict, frozen: dict) -> dict:
    policy = validate_frozen(frozen)
    if policy['policy_revision'] != CALIBRATED or binding.get('release_policy_revision') != CALIBRATED:
        raise AcceptanceError('acceptance requires explicit calibrated policy binding')
    binding_digest = hashlib.sha256(canonical(binding)).hexdigest()
    inventory = Inventory(bundle)
    try:
        raw, review_file = inventory.read('review.json', json_limit=True)
        review = inventory.decode(raw)
        exact(review, ('schema','status','policy_sha256','catalog_binding_sha256','reviewed_commit','reports','reviewers','review_digest'), 'review')
        if review['schema'] != SCHEMA or review['status'] != 'approved': raise AcceptanceError('review is not approved')
        unsigned = dict(review)
        unsigned.pop('review_digest')
        if review['review_digest'] != hashlib.sha256(canonical(unsigned)).hexdigest(): raise AcceptanceError('review digest differs')
        if review['policy_sha256'] != frozen['sha256'] or review['catalog_binding_sha256'] != binding_digest:
            raise AcceptanceError('review policy or catalog binding differs')
        commit = review['reviewed_commit']
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise AcceptanceError('review commit must be full lowercase git SHA')
        exact(review['reviewers'], ('runtime','model','performance','release'), 'reviewers')
        if any(not isinstance(v,str) or not v.strip() for v in review['reviewers'].values()): raise AcceptanceError('missing reviewer identity')
        exact(review['reports'], ACCEPTANCE_GATES, 'acceptance reports')
        verdicts = {}
        for gate in ACCEPTANCE_GATES:
            report = inventory.referenced(review['reports'][gate], json_limit=True)
            exact(report, ('schema','gate','policy_sha256','catalog_binding_sha256','reviewed_commit','checks'), 'report')
            if (report['schema'] != REPORT_SCHEMA or report['gate'] != gate or
                    any(report[k] != review[k] for k in ('policy_sha256','catalog_binding_sha256','reviewed_commit'))):
                raise AcceptanceError('report binding differs from review')
            exact(report['checks'], CHECKS[gate], gate+' checks')
            worst = 'PASS'
            for check in report['checks'].values():
                exact(check, ('verdict','artifacts'), 'check')
                verdict = check['verdict']
                artifacts = check['artifacts']
                if verdict not in VERDICTS or not isinstance(artifacts,list) or len(artifacts) > 256:
                    raise AcceptanceError('invalid check verdict/artifacts')
                if verdict != 'NOT_RUN' and not artifacts: raise AcceptanceError('executed check has no raw evidence')
                for artifact in artifacts: inventory.referenced(artifact)
                if VERDICTS.index(verdict) > VERDICTS.index(worst): worst = verdict
            verdicts[gate] = worst
        return dict(schema='tc-streaming-acceptance-verification-v1', verdicts=verdicts,
                    reviewed_commit=commit, review_digest=review['review_digest'], review_file=review_file,
                    catalog_binding_sha256=binding_digest, policy_sha256=frozen['sha256'],
                    artifacts=list(inventory.verified.values()), production_authorized=False)
    finally:
        inventory.close()
