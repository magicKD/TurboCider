import Foundation
import Combine
import ImageIO

@main struct TinyVAEAppTests {
    @MainActor static func main() async throws {
        func check(_ condition: @autoclosure () -> Bool, _ reason: String) throws {
            if !condition() { throw NativeFailure(message: reason) }
        }
        var draft = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(
            !draft.tinyPreview && draft.tinyPreviewInterval == 2
                && draft.tinyVAEEncoder == "original", "old drafts must retain quality defaults")
        draft.modelID = "qwen-image-2.1"
        draft.tinyPreview = true
        draft.tinyVAEPaths["taeqi2_1"] = "/missing/tiny.safetensors"
        do {
            _ = try draft.tinyVAEConfiguration()
            throw NativeFailure(message: "accepted missing tiny weights")
        } catch let failure as NativeFailure {
            try check(failure.message.contains("请先"), "wrong missing-model error")
        }
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "tiny-app-\(UUID())")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let weights = temporary.appendingPathComponent("taeqi2_1.safetensors")
        try Data([0]).write(to: weights)
        draft.tinyVAEPaths["taeqi2_1"] = weights.path
        draft.tinyVAEEncoder = "tiny"
        draft.tinyVAEDecoder = "tiny"
        draft.operation = "image.edit"
        let config = try draft.tinyVAEConfiguration()
        try check(
            config?.encoder == "tiny" && config?.decoder == "tiny" && config?.interval == 2,
            "configuration lost selection")
        var legacy = NativeRequest(
            prompt: "test", output: temporary.appendingPathComponent("result.png").path)
        legacy.tiny_vae = config
        let v2 = NativeRequestV2(legacy: legacy)
        let roundtrip = try JSONDecoder().decode(
            NativeRequestV2.self, from: JSONEncoder().encode(v2))
        try check(roundtrip.parameters.tiny_vae == config, "schema 2 dropped tiny settings")
        let restored = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
        try check(
            restored.tinyVAEPaths == draft.tinyVAEPaths && restored.tinyPreview
                && restored.tinyVAEDecoder == "tiny", "draft persistence dropped settings")
        draft.streaming.selection = .tier12
        do {
            _ = try draft.tinyVAEConfiguration()
            throw NativeFailure(message: "accepted uncalibrated streaming")
        } catch let failure as NativeFailure {
            try check(failure.message.contains("内存档位"), "wrong streaming error")
        }
        let good = NativeEvent(
            sequence: 1, phase: "preview", completed: 2, total: 6, elapsed_seconds: 1,
            preview_path: TinyPreviewPath.expected(output: legacy.output))
        try check(
            TinyPreviewPath.valid(good, output: legacy.output, steps: 6),
            "valid preview path rejected")
        var bad = good
        bad.preview_path = "/other/private.png"
        try check(
            !TinyPreviewPath.valid(bad, output: legacy.output, steps: 6),
            "unrelated preview path accepted")
        try check(
            !TinyPreviewPath.valid(good, output: legacy.output, steps: 4),
            "preview wrong step count accepted")
        print(
            "Tiny VAE App defaults, persistence, request conversion, memory guard and preview correlation passed"
        )
        let args = CommandLine.arguments
        if args.count == 3 && args[1] == "--download" {
            let library = ModelLibraryController(root: URL(fileURLWithPath: args[2]))
            let studio = StudioState(directory: temporary.appendingPathComponent("download-ui"))
            for asset in TinyVAEAsset.all {
                let originalModel = studio.draft.modelID
                library.downloadTinyVAE(asset, source: .mirror)
                let started = ContinuousClock.now
                while library.busy {
                    try check(
                        started.duration(to: .now) < .seconds(30),
                        "App download helper did not finish")
                    try await Task.sleep(for: .milliseconds(30))
                }
                guard let path = library.tinyVAEPath(asset) else {
                    throw NativeFailure(
                        message: library.message ?? "App download produced no installation")
                }
                library.useTinyVAE(asset, studio: studio)
                try check(
                    studio.draft.modelID == originalModel
                        && studio.draft.tinyVAEPaths[asset.id] == path,
                    "selecting tiny changed base model or lost the file")
                studio.draft.modelID = asset.modelID
                studio.draft.tinyPreview = true
                let selected = try studio.draft.tinyVAEConfiguration()
                try check(
                    selected?.path == path && selected?.preview == true,
                    "download selection did not enter native request")
            }
            try check(
                library.installations.filter { $0.modelID.hasPrefix("tiny-vae-") }.count == 2,
                "cached download duplicated installs")
            print(
                "PASS App Tiny VAE download helper, verified installs, cached reuse, selection and native request binding"
            )
            return
        }
        guard args.count == 6 && ["--gpu", "--gpu-base"].contains(args[1]) else { return }
        let authoredBase = args[1] == "--gpu-base"
        let modelID = args[2], model = URL(fileURLWithPath: args[3]),
            tiny = URL(fileURLWithPath: args[4]).standardizedFileURL.path
        let output = URL(fileURLWithPath: args[5])
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let store = NativeJobStore(directory: output)
        var reports: [[String: Any]] = []
        var baseline: Data?
        var modes = authoredBase ? ["tiny-final"] : ["baseline", "preview", "cancel", "tiny-final"]
        if modelID == "qwen-image-2.1" && !authoredBase { modes.append("tiny-edit") }
        for mode in modes {
            var request = NativeRequest(
                prompt:
                    "A red ceramic teapot on a wooden table, warm sunlight, detailed photography.",
                output: output.appendingPathComponent("\(mode).png").path)
            request.model = modelID
            request.operation = "image.generate"
            request.width = 512
            request.height = 512
            request.steps = modelID == "qwen-image-2.1" ? 8 : 4
            request.audio = false
            request.residency = modelID == "qwen-image-2.1" ? "component_staged" : "resident"
            if mode != "baseline" {
                request.tiny_vae = NativeTinyVAEConfig(
                    path: tiny, decoder: mode == "tiny-final" ? "tiny" : "original",
                    preview: true, interval: mode == "tiny-final" ? 3 : 2)
                if mode == "tiny-final" { request.allow_approximation = true }
            }
            if mode == "tiny-edit" {
                request.operation = "image.edit"
                request.inputs = [
                    NativeInput(
                        kind: "image", role: "reference",
                        path: output.appendingPathComponent("baseline.png").path)
                ]
                request.prompt =
                    "Change the teapot in <image1> to blue. Preserve its shape and the background."
                request.tiny_vae?.encoder = "tiny"
                request.allow_approximation = true
            }
            if authoredBase {
                var settings = StudioDraft()
                settings.modelID = modelID
                settings.modelPaths[modelID] = model.path
                settings.prompt = request.prompt
                settings.operation = "image.generate"
                settings.width = 512
                settings.height = 512
                settings.steps = 20
                settings.audio = false
                settings.residency = "component_staged"
                settings.acceleration = StudioAcceleration(policy: "gpu")
                settings.tinyVAEPaths[settings.tinyArchitecture] = tiny
                settings.tinyVAEDecoder = "tiny"
                settings.tinyPreview = true
                request = try settings.request(output: URL(fileURLWithPath: request.output))
                try check(
                    request.allow_approximation == true && request.tiny_vae?.decoder == "tiny"
                        && request.steps == 20,
                    "ordinary Creation draft lost tiny settings/approximation permission")
            }
            var previews: [Int] = [], phases = Set<String>(), cancelled = false
            let progress = store.$jobs.sink { jobs in
                if let job = jobs.first(where: { $0.request.output == request.output }) {
                    phases.insert(job.phase)
                }
            }
            let observation = store.$livePreview.sink { data in
                guard let data else { return }
                if CGImageSourceCreateWithData(data as CFData, nil) != nil {
                    previews.append(store.livePreviewStep)
                }
                if mode == "cancel", !cancelled {
                    cancelled = true
                    store.cancel()
                }
            }
            print("START \(modelID) \(mode)")
            do {
                let job = try await store.generate(modelURL: model, request: request)
                try check(mode != "cancel", "cancelled job succeeded")
                try check(job.state == "succeeded", "job did not succeed")
                let data = try Data(contentsOf: URL(fileURLWithPath: request.output))
                guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                    let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                else { throw NativeFailure(message: "unreadable result") }
                try check(image.width == 512 && image.height == 512, "output geometry mismatch")
                if mode == "baseline" { baseline = data }
                if mode == "preview" { try check(data == baseline, "preview changed final image") }
                if mode != "baseline" {
                    try check(!previews.isEmpty, "App did not display a live preview")
                }
                if mode == "tiny-edit" {
                    try check(
                        phases.contains("tiny_vae_encode"), "tiny reference encoder not invoked")
                }
                reports.append([
                    "mode": mode, "state": job.state, "preview_steps": previews,
                    "phase_count": phases.count,
                    "preview_unchanged": mode == "preview" ? data == baseline : true,
                ])
            } catch {
                guard mode == "cancel", cancelled else { throw error }
                try check(
                    !FileManager.default.fileExists(atPath: request.output),
                    "cancelled output was published")
                reports.append(["mode": mode, "state": "cancelled", "preview_steps": previews])
            }
            observation.cancel()
            progress.cancel()
            try check(
                store.livePreview == nil && !store.busy,
                "preview or busy state leaked after terminal result")
        }
        try JSONSerialization.data(withJSONObject: reports, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("acceptance.json"))
        print("Tiny VAE App GPU acceptance passed: \(reports)")
    }
}
