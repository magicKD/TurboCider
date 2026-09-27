# Streaming App integration status

Experiments were stopped at the user's request on September 27, 2026. The coordinator and its generation workers have exited. Do not automatically restart qualification or run the prepared acceptance scripts. Original evidence remains available under `outputs/integration-20260927`; partial results are not release approval.

## Local experimental delivery (September 27 follow-up)

The user subsequently authorized a small smoke test and accepted a **local experimental configuration**, without a full-request memory guarantee. The long qualification campaigns remain stopped.

- The main checkout now owns `feat/stream`. The earlier nested `stream` worktree was detached without removing its files or evidence; it no longer prevents checking out the branch.
- Merge `97d8ab8` incorporates `dev-verify` at `a39f833` without conflicts, including the four-image M4 Pro Metal comparison.
- `assets/config/local-streaming-profile.json` defines the local App shortcut: Apple M4 Pro / 48 GiB, Z-Image Turbo BF16, GPU, no LoRA, 512×512, 9 steps, dynamic text and a 10 GiB **sampling planning budget**. This is not a calibrated catalog record or a process-memory cap.
- In the App's advanced parameters, select **应用本机实验流式配置（512×512 · 9 步）**. Incompatible hardware, weights, ANE or LoRA disable this shortcut with an explanation. Applying it preserves the prompt, seed and model path, persists the settings, and uses the existing experimental request route.
- Packaging now honors the same output-directory overrides as compilation, avoiding accidental packaging of a stale default build after switching branches.

The native release library was incrementally rebuilt after confirming that all reused native source inputs were unchanged; build identity and catalog sources were rebuilt and the library relinked. Test hooks and memory-audit instrumentation are disabled. The production catalog stays empty. The App and Studio regression executable were rebuilt for the local shortcut.

A bounded smoke test using the packaged library generated three 512×512 images at 9 steps and seed 42: resident **20.416 s**, streamed **18.575 s**, streamed **17.385 s**. All three succeeded and their PNG bytes matched. These sequential cold/warm samples establish functionality only, not a speed ranking or memory qualification. Evidence: `outputs/integration-20260927/quick-stream-finalize/packaged-smoke/report.json`.

Studio regression tests passed, including hardware/configuration eligibility, rejected-setting preservation, request routing and budget forwarding, draft persistence and rejection of a claimed memory guarantee. Catalog, text-capacity and acceptance contract regressions also passed; they remain separate from capacity qualification. The final package passed strict code-signature verification and contains the checked profile resource.

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

## Earlier qualification campaign: `17e40ec`

These results belong to that exact build. The later local smoke test above does not complete or inherit this qualification campaign.

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

Full capacity qualification, lifecycle acceptance and reviewed production catalog construction remain incomplete. A locally signed experimental App is available; a capacity-qualified public release is not established. Resume experiments only when requested. Preserve the original evidence and use a fresh output directory for any future run.

Detailed implementation history, earlier performance observations and failed or inconclusive experiments are retained in [the history](../experiments/2026-09-27-streaming-integration-history.md).
