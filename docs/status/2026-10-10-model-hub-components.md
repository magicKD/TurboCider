# Model hub, component installations and LoRA management

Accepted on 2026-10-12, based on `c0155fe` with this milestone's changes.

## Delivered behavior

- A native model hub with Discover / On device, format-aware search, capability
  filters, offline brand monograms and a single selection for model and upscaler details.
  Multiple registered paths remain visible when the active path is removed.
- Separate Encoder / DiT / VAE inventory for Qwen-Image-2.1, FLUX.2 Klein and
  Z-Image. Replacements create a checked, linked installation and retain the
  original weights. Other families retain their complete-install workflow.
- ModelScope-first download preview with selectable Hugging Face, complete or
  individual components, source provenance, destination mapping, cancellation
  and verified cache reuse. The Qwen default recipe now includes its processor
  tokenizer. Partial or unsupported downloads do not replace the active model.
- Seven published Unsloth Qwen GGUF DiT choices from Q4 through Q8. Only the
  existing Q4_K_M approximate text-to-image route is executable; other choices
  are explicitly download-only. BF16 remains the supported editing/LoRA route.
- A separate LoRA management tab for local adapters, model ownership, import,
  discovery, enabled state and strength. All enabling uses creation state
  methods, including the recognized Viggle six-step preset. Scans accept both
  `lora` and `loras`, ignore directories and deduplicate links.
- Local agent commands: `library components` and `library compose`, with bounded
  request validation. No new service, Python runtime or model export is required.

## Validation scope

No extra model weights were downloaded. Published repository metadata was read
to verify Unsloth variant names and ModelScope availability. Download tests use
tiny loopback HTTP fixtures; component tests use bounded sparse header fixtures.
No generation, GPU stress campaign, ANE compilation or Core ML export is part of
this UI milestone.

The existing local Qwen BF16 installation passed read-only checks for its three
weight files and processor. An isolated composition reused those same sources;
the resulting registry and links used 4 KiB of physical disk. These checks do
not establish inference quality or validate an actual downloaded Unsloth tensor
header.

## Acceptance results

- `make test-library`: all eight stages passed against the final model-library
  and creation-state sources. Coverage includes 55 component assertions, model hub presentation,
  LoRA state and persistence, controller queue/cancellation/recovery, library
  integrity, six-family installation inspection, helper transport and tiny
  ModelScope/Hugging Face loopback downloads. Source and binary hashes were
  unchanged during this run.
  A subsequent view-only scroll reset was rebuilt into the App and checked
  interactively; none of the tested backend/state sources changed.
- `tests/native/test_contract.py`: 129 cases completed, with three existing
  local-fixture skips and no failures in the normal Metal-visible environment.
  Repository layout, independence and C++ boundary suites passed all 17 cases.
- The local Qwen installation and a linked composition passed inventory and
  inspection using the packaged helper. All three components were present,
  and the composition retained the original tokenizer bytes.
- Native App acceptance covered Discover / On device, search, capability
  filters, Qwen component cards and configuration cancellation; ModelScope
  default, HF switching, all seven GGUF choices and single-DiT filtering; LoRA
  filtering/search, batch selection and duplicate registration, explicit enable,
  disable/re-enable with strength 0.75 retained, strength reset and removal from
  the draft. The original Z-Image draft ended with zero enabled adapters, and
  registration retained all four existing LoRA records. No generation was run.
  In the final rebuilt App, direct model switching and search fallback both
  reset a fully scrolled detail pane from position 1 to 0; the new model's
  header is visible immediately. On-device and upscaler details were rechecked.
- The native runtime was rebuilt from the current native sources. Its 605
  recorded source hashes matched the checkout. The App and model-library helper
  were rebuilt; the packaged App passed strict deep signature verification.
  A single previous App bundle remains available as a recoverable local backup.

The initial component-test failure was a Foundation directory-URL comparison
in the fixture. The final fixture compares canonical paths and also verifies
the inherited tokenizer bytes. The failure log is retained alongside the
successful rerun; it is not counted as a passing result.

Local receipts are under `outputs/model-hub-20261010/`, including
`library-acceptance-final.json`, `accepted-package-receipt.json`,
`native-contract-metal.log`, `repository-tests.log`, `ui-acceptance-final.json`
and the final `app-build-scroll-reset.log` / `package-scroll-reset.log`.
The deliverable is `dist/TurboCider.app`; generated binaries and local receipts
are deliberately excluded from Git.

### UI automation limitation

Reading the Z-Image download sheet caused the external SkyComputerUseService
to crash in its accessibility traversal (`Array.remove(at:)` / `compactMap`).
TurboCider remained running, and UI automation recovered after the user closed
that sheet. The Z-Image precision/download backend remains covered by offline
fixtures and contract checks, but that sheet's final interactive acceptance is
not claimed. Qwen's download sheet and the LoRA page were inspected successfully.

## Compatibility boundaries

Component file checks validate selected architecture metadata, key shapes and
byte ranges. Native model loading remains responsible for complete tensor
compatibility. Replacing an encoder inherits the base tokenizer/processor.
Linked source files must remain available. Removing a registration retains files.
Adapter ownership is user-declared; registration does not establish LoRA tensor
compatibility. Individual components cannot run as a complete model.

Published GGUF DiT files do not quantize the accompanying BF16 encoder/VAE.
There is no claim that Q5/Q6/Q8 inference or GGUF editing/LoRA works in this build.
Source: [Unsloth Hugging Face files](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/tree/main)
and [ModelScope repository](https://modelscope.cn/models/unsloth/Qwen-Image-2.1-GGUF).

## Build environment

The installed Xcode launcher failed SDK lookup because its CoreDevice framework
does not match the system. Builds use the installed Command Line Tools and SDK
15.2 explicitly, without changing global developer settings. Sandboxed Metal
discovery reports an unavailable GPU; the existing native request contract suite
passes in the normal Metal-visible environment. Initial diagnostic failures are
retained in local logs rather than counted as acceptance results.
