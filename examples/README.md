# TurboCider examples

Run examples from the repository root. Start with the
[build and model setup guide](../docs/public/GETTING_STARTED.md).

| Example | Use it to… |
| --- | --- |
| [FLUX image request](requests/generate.json) | Inspect a minimal schema 2 text-to-image request. |
| [LTX video request](requests/ltx-video.json) | Inspect a text-to-video request with explicit frames, frame rate and residency. |
| [SeedVR2 image upscale](requests/seedvr2-image-upscale.json) | Replace the sample source and output paths to upscale a local image. |
| [GPU and GPU/ANE routes](requests/README.md) | Choose Qwen or Z-Image templates and matching acceleration manifests. |
| [Qwen reference editing with the local API](local_api/qwen_reference_workflow.py) | Plan an edit with ordered reference images. Set `--run` to submit through the configured local socket. |
| [Core ML artifact examples](coreml/README.md) | Find conversion requests and runtime descriptors with their prerequisites. |

With a native build, check the FLUX request before loading model weights:

```sh
build/native/turbocider plan examples/requests/generate.json
```

Before generating, supply a matching local installation and choose a fresh
output path. See [request and CLI usage](../docs/public/USAGE.md#requests-and-cli)
for command formats and [local API setup](../docs/public/LOCAL_API.md) for
the socket example.
