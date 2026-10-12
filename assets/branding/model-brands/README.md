# Model publisher logos

These small images identify the model publishers in TurboCider's model hub and
download sheets. Retrieved on 2026-10-12 from the publishers' official GitHub
organization avatars, using GitHub's 256-pixel image endpoint. Original PNG/JPEG
bytes are retained without recoloring or reconstruction.

| File | Publisher / source account |
|---|---|
| `qwen.png` | [Qwen](https://github.com/QwenLM) |
| `flux.png` | [Black Forest Labs](https://github.com/black-forest-labs) |
| `ltx.jpg` | [Lightricks](https://github.com/Lightricks) |
| `minimax.jpg` | [MiniMax](https://github.com/MiniMax-AI) |
| `wan.png` | [Wan](https://github.com/Wan-Video) |
| `z-image.jpg` | [Tongyi-MAI](https://github.com/Tongyi-MAI) |
| `llada.jpg` | [inclusionAI](https://github.com/inclusionAI) |

`SOURCES.json` records the source page, exact download URL, retrieval date, byte
length and SHA-256 for each file. The App bundles these assets and reads them
once, with no runtime image download. A white presentation tile preserves the
contrast of the original dark marks in either appearance. Missing assets fall
back to the existing native badge; Real-ESRGAN retains its native upscale icon.

Names and logos remain trademarks/artwork of their respective owners. They are
used for publisher identification, do not imply endorsement or affiliation,
and are not covered by TurboCider's MIT license. Upstream brand guidance applies,
including [Black Forest Labs' brand page](https://bfl.ai/brand).
