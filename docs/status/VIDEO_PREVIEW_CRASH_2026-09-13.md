# Video result preview crash, 2026-09-13

## Evidence and diagnosis

On macOS 26.6.2, the desktop App aborted on the main thread after a completed
LTX video job. The crash stack entered `getSuperclassMetadata` from
`_AVKit_SwiftUI`, then SwiftUI's view-representable setup. The job had already
persisted `succeeded`, and the H.264/AAC output decoded successfully: 704×448,
97 frames at 24 FPS, 48 kHz stereo audio, 4.041667 seconds.

Launching the old packaged App with an isolated copy of that history reproduced
the failure immediately, without running inference. Its stderr was:

```text
failed to demangle superclass of VideoPlayerView from mangled name 'So12AVPlayerViewC': unknown error
```

`otool -L` showed that the old executable linked `_AVKit_SwiftUI` but did not
directly link `AVKit`. This is a missing-superclass/framework-loading failure,
not an observed out-of-memory termination. The crash report's virtual-memory
summary must not be interpreted as inference peak RSS or Metal peak memory.

## Fix

Replace SwiftUI `VideoPlayer` with a concrete `NSViewRepresentable` backed by
`AVPlayerView`. Referencing the concrete class causes the App to directly link
AVKit, and avoids the failing `_AVKit_SwiftUI` wrapper. Keep the same player on
same-path view updates, and pause, detach its item and release it on teardown.

The existing GPU memory policy remains unchanged:

- The App awaits unloading the previous image model before starting LTX.
- `component_staged` runs LTX in a disposable process.
- The worker replaces itself with the clean VAE finalizer (`video_vae_isolation=exec`).
- The worker exits after media export; no resident LTX session remains in the UI.

Block streaming is not a remedy for a playback superclass failure. It remains
an explicit option for supported workloads, not a default change for this fix.

## Regression checks

`make test-video-preview VIDEO=/path/to/generated.mp4` hosts the actual preview
in a SwiftUI window without linking the inference runtime. It checks mounting,
frame decoding, playback progress, stable player identity on refresh, item
release and reloading. Build it with `tools/native/build_app.sh`.

Also compare old/fixed packaged Apps using the same isolated job history, and
inspect the fixed executable's `otool -L` output for a direct AVKit dependency.
Do not put old `VideoPlayer` and the fixed view in the same reproducer binary:
the fixed view's AVKit reference supplies the dependency and masks the old bug.
