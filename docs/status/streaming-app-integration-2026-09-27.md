# Streaming App integration status

Experiments were stopped at the user's request on September 27, 2026. The coordinator and its generation workers have exited. Do not automatically restart qualification or run the prepared acceptance scripts. Original evidence remains available under `outputs/integration-20260927`; partial results are not release approval.

## Branch integration

- `dev-verify` at `95b1760` includes `dev` (`3421144`) and `main` (`ee06ece`).
- `feat/stream` includes `feat/stream-dev` (`99d3f48`) and the merged `dev-verify` (`bcf8415`).
- Existing local models were used; unavailable optional models received static checks only. No model download was required.

## Implementation map

| Responsibility | Location |
| --- | --- |
| App selection, draft persistence and resident recovery | `apps/macos/StudioState.swift`, `apps/macos/JobStore.swift` |
| Native streaming catalog and request resolution | `native/runtime/streaming/` |
| Calibration and reviewed record construction | `tools/native/build_streaming_catalog.py` |
| Original text-boundary evidence validation | `tools/native/verify_streaming_text_capacity.py` |
| App discovery, generation, cancellation and restart checks | `tests/integration/PublicImageStreamingSmoke.swift` |
| Text-capacity contract regressions | `tests/native/test_streaming_text_capacity_evidence.py` |

Text-capacity case validation is separated into request/result parity, native execution receipts, and process-bound memory sampling. The public validation entry point and its checks are unchanged. Calibration combines P2 memory evidence with text-boundary peaks; the estimator name has one shared definition.

The Z-Image capacity contract covers dynamic BF16 GPU worker requests within the record's token interval, bounded by 1024. It binds the actual encoder rows and aligned caption layout independently. Fixed-text requests retain exact-record matching. Model content, device, resolution, sampling policy and other request features remain part of the binding.

Legacy streaming budgets retain their original meaning. Unavailable or incompatible settings expose an explicit resident-loading recovery action that preserves prompt, LoRA and acceleration settings. Test catalog availability does not establish production availability.

## Last measured build: `17e40ec`

These results belong to that exact build. Subsequent code cleanup has not been rebuilt or measured, and must not inherit its qualification label.

| Check | Result | Evidence under `outputs/integration-20260927` |
| --- | --- | --- |
| Ordinary GPU P0 | PASS; 40 successful requests, 20 identical image pairs; wall medians 19.116 / 19.097 seconds | `stream-qualification-17e40ec/p0-independent-verification.json` |
| 1024-token streaming P1 | PASS; 40 successful requests, 20 identical image pairs; wall medians 40.435 / 40.552 seconds | `stream-qualification-17e40ec/p1-independent-verification.json` |
| Memory P2 | USER STOPPED; 22 successful requests recorded out of 40; no final verdict | `stream-qualification-17e40ec/operator-stop.json` and `p2/raw-samples.jsonl` |
| Current text-boundary qualification | NOT RUN | No current range bundle |
| Current real App/installation/lifecycle acceptance | NOT RUN | Prepared scripts were not executed |

Native/App configuration regressions passed before the code cleanup. The earlier `acd024b` test build completed all automated campaigns, boundaries and App checks; that evidence remains historical and cannot qualify the newer build.

The production catalog `native/runtime/streaming/bundled_catalog.json` remains empty. No production streaming record, release approval or newly qualified App package is claimed.

## Cleanup validation

The behavior-preserving cleanup passed 48 host-only regression tests: 7 text-capacity, 21 catalog builder, 9 bundled catalog and 11 acceptance tests. No model inference, performance experiment or native/App rebuild was run for this cleanup.

## Remaining work

Qualification, acceptance, reviewed catalog construction and final production App packaging remain incomplete. Resume experiments only when requested. Preserve the original evidence and use a fresh output directory for any future run.

Detailed implementation history, earlier performance observations and failed or inconclusive experiments are retained in [the history](../experiments/2026-09-27-streaming-integration-history.md).
