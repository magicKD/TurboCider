# TurboCider Web mini apps

Open **应用** in the macOS sidebar. **人物工作室** is the bundled app adapted from PoseWear. **创作模板** opens the existing native templates. Community developers can install a local `.tcapp` from **安装应用…**.

## 人物工作室

Use the gallery, full workbench or floating workbench to select references. The full workbench also accepts local person, clothing, pose, depth, expression, hair, scene, lighting, prop and furniture images.

- **使用 Demo** selects a bundled person, outfit and pose. The pose has a matching depth image. The app starts offline and makes no requests to material sources until you enable online materials or start a pack download. The bundled CC0 images are attributed in `NOTICE.txt`.
- Open **在线素材** and select **开启在线素材** to browse the published PoseWear catalog. It includes pose sketches, depth poses, outfits, anime outfits, portraits, expressions, hair, historical costumes, scenes, lighting, props and furniture. Search, filters and angle switching use the original catalog metadata. Previews load from the source as you browse. Submitting a job copies the selected references into that job's local inputs so they can be reused from history. CharaDB images support in-app previews and links to the source. [PoseWear catalog and credits](https://github.com/atuizz/posewear#素材来源与许可)
- **下载到本机** is optional for the CC0 historical costume, scene, lighting, prop and furniture sets. Each pack can be used with online materials turned off after the download finishes. An interrupted download can reuse completed files for the same catalog version. [Shisei asset license](https://github.com/wquguru/shisei/blob/6e22358bab73c51eeae0615030635ace199169f8/docs/LICENSING.md)

- The default engine is the authored eight-step Qwen Image 2.1 Turbo. Choose an installed Turbo variant in the model library or select its folder from the app's toolbar.
- A single edit can combine identity, clothing and pose in that order. Paired pose sketch and depth views retain their own image positions. Additional references follow their slot order.
- **分两步编辑** creates a pose/environment result first and uses that output as the person image for the outfit stage. It uses separately numbered automatic prompts. Both stages appear in the host task center.
- Pose groups can render individually or as a model-generated sheet. Results support comparison, library history, use as the next person image, and native save dialogs. Closing the interface leaves submitted jobs running in the desktop task center.
- **画幅 → 自动** follows the main person image, then a pose or depth image, then the first reference. It targets about 512×512 pixels, with a 768×512 pixel-area limit and 32-pixel alignment. Select **竖 1:2** for 384×768 or **横 2:1** for 768×384. Without references, automatic uses 512×512. The supported Turbo LoRA route also uses 512×512.
- All references in a shot share a total 512×512 pixel budget. The model resizes them proportionally and keeps their original media available in history. Results show the output dimensions and total encoded reference pixels.
- On a compatible ConvRot installation the default Turbo route streams encoder and DiT weights. It uses compact prefix KV in memory while there is headroom and refills one layer at a time from a request-owned temporary file under memory pressure. Ordinary Turbo LoRA and other installations use the component-staged path with the same reference budget.
- Optional transformer LoRAs appear in the workbench's local generation settings when the model library has a compatible registration. The authored schedule remains eight steps. The ordinary adapter route requires a square 512×512 output.

The bundled web source is pinned to PoseWear commit `f9f55db356464af87f97b39f5bb57930ed0cd6df`. See the bundled `LICENSE.txt` and `NOTICE.txt` for attribution.

## Package a community app

Build static HTML, CSS and JavaScript with relative paths. Include a root manifest. The `org.turbocider.*` namespace is reserved by the host.

```json
{
  "schemaVersion": 1,
  "sdkVersion": 1,
  "id": "com.example.lookbook",
  "name": "Lookbook",
  "version": "1.0.0",
  "entry": "index.html",
  "permissions": ["media.import", "media.export", "image.generate"],
  "description": "Create portraits with the local Turbo engine."
}
```

From the repository root, build the sample and open it with **安装应用…**.

```bash
python3 tools/mini-apps/pack.py examples/mini-apps/image-sample /tmp/image-sample.tcapp
```

Use an external script file for app code. UTF-8 files have relative ASCII package paths. The host accepts HTML, JavaScript, CSS, JSON, text, supported raster images and WOFF fonts. ZIPs may use stored or deflated entries. The installer validates paths, entry limits, CRC values and expanded size. The package limit is 64 MiB and 1,024 entries. Users confirm a package's identity and permissions on installation or replacement.

The first release accepts locally installed packages. Each WebView has its own offline runtime and scoped host storage. The host serves its package and media URLs. It blocks external navigation and resource requests. The built-in People Studio can request material data through a dedicated host service that fetches PoseWear's published catalog and media from approved source URLs. The host verifies catalog fingerprints and parses data without executing the upstream scripts. Install updates from sources you trust because the replacement uses that app's scoped data.

## SDK v1

`window.TurboCider` is injected by the macOS host. The repository includes [TypeScript types](../../examples/mini-apps/turbocider-sdk.d.ts) and a working [sample](../../examples/mini-apps/image-sample/main.js).

| Method | Permission | Result |
| --- | --- | --- |
| `call('handshake')` | Default | App identity, SDK version and structural request limits |
| `call('models.list')` | `image.generate` | Supported engine IDs and whether an installation was selected |
| `call('adapters.list', {engine})` | `adapters.use` | Registered compatible adapter IDs, names and default strengths |
| `media.bundled(path)` | Default | Handle for an image in this package |
| `media.import(blob, name)` | `media.import` | Handle for a user-supplied static image, up to 16 MiB |
| `media.get(id)` | Default | Read an image handle owned by this app |
| `media.save(id)`, `media.saveBlob(blob, name)` | `media.export` | Native save panel. Raw-byte exports are limited to 24 MiB |
| `storage.get(key)`, `storage.set(key, value)` | Default | App-scoped string preferences, up to 512 KiB total |
| `workflow.plan(options)` | `image.generate` | Frozen plan ID, ordered shot summaries, seeds, geometry and native memory estimate |
| `jobs.submit(planID)` | `image.generate` | Persisted job snapshots in plan order |
| `jobs.get(id)`, `jobs.watch(id)`, `jobs.list()` | `image.generate` | Status and published output handles for this app |
| `jobs.cancel(id)`, `jobs.remove(id)` | `image.generate` | Cancel a pending job or remove a terminal record |

Plans accept up to 12 shots. A shot has a prompt and an ordered reference list. An empty list generates an image. A reference is either `{mediaID, role}` or `{shotIndex, role}`. The zero-based `shotIndex` must refer to an earlier shot in the same plan. A dependent shot can start after the parent publishes an output. If the output is unavailable, the host records a failed dependency. Engines have their own request limits. Query the host and preflight before submitting.

A shot can specify `width` and `height` or `matchReferenceAspect: true`. Turbo shots can set `referencePixelBudget` to select shared sizing. The value is a multiple of 1,024 pixels and at most `262144`. The plan freezes the canvas dimensions, residency and reference budget. A completed job reports its actual `referencePixels`.

```js
const tc = window.TurboCider;
const requestID = tc.createRequestID();
const plan = await tc.workflow.plan({
  requestID,
  engine: "qwen-image-2.1-turbo",
  workflow: "com.example.lookbook.edit",
  shots: [{
    prompt: "Keep the person’s facial identity. Apply the clothing and pose.",
    references: [
      { mediaID: person.id, role: "person" },
      { mediaID: outfit.id, role: "garment" },
      { mediaID: pose.id, role: "pose" }
    ]
  }]
});
const [job] = await tc.jobs.submit(plan.id);
for await (const update of tc.jobs.watch(job.id)) {
  render(update);
}
```

A plan ID expires after ten minutes or when the app replaces its current plan. Retry delivery with the same plan ID or reconstruct the same options and request ID. An existing request with the same ID and different options returns `requestConflict`. To create a new job after a failure, use a new request ID. On app restart, query job history for terminal or interrupted status.

The SDK returns an opaque media ID and an app-scoped preview URL. Use that ID to feed a published output into another plan owned by the app. Native paths, arbitrary network access and process controls are outside the SDK.

## Test

```bash
make test-mini-apps
```

The test runs the real WebKit host, official gallery and sample package, along with scope and permission checks. Set `TURBOCIDER_TEST_LIVE_PEOPLE_MATERIALS=1` to verify the 12 published catalog collections and load previews from the current upstream sources. Add `TURBOCIDER_TEST_LIVE_PEOPLE_PACK=1` to also download the public props pack into a temporary test directory and reopen it offline. To include native generation, set `TURBOCIDER_MINI_APP_TEST_MODEL` to a complete Turbo installation and `TURBOCIDER_MINI_APP_TEST_GPU=1`. Add `TURBOCIDER_MINI_APP_TEST_STAGED=1` for the dependent two-stage workflow or `TURBOCIDER_MINI_APP_TEST_PORTRAIT=1` for the four-reference 384×768 workflow.
