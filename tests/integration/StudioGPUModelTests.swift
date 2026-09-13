import AppKit
import AVFoundation
import Foundation

/// Real-weight acceptance through exactly the same draft/store entrypoints as UI.
@main struct StudioGPUModelTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 4 else { throw NativeFailure(message: "studio-gpu-model-tests Z_IMAGE LTX OUTPUT") }
        let root = URL(fileURLWithPath: args[3])
        let studio = StudioState(directory: root)
        let store = NativeJobStore(directory: root)
        func check(_ value: Bool, _ message: String) throws {
            guard value else { throw NativeFailure(message: message) }
        }
        var reports: [String: Any] = [:]
        studio.draft.modelPaths["z-image-turbo"] = URL(fileURLWithPath: args[1]).path
        studio.draft.modelPaths["ltx-2.5-distilled"] = URL(fileURLWithPath: args[2]).path
        studio.selectModel("z-image-turbo")
        studio.draft.prompt = "A red fox standing in a snowy forest, soft morning light, realistic wildlife photograph."
        studio.draft.randomSeed = false; studio.draft.seedText = "42"
        let image = root.appendingPathComponent("outputs/z-image-gpu.png")
        let zRequest = try studio.draft.request(output: image)
        try check(zRequest.execution == "gpu" && zRequest.allow_approximation != true, "Z-Image not exact GPU")
        let zJob = try await store.generate(modelURL: URL(fileURLWithPath: args[1]), request: zRequest)
        guard let bitmap = NSBitmapImageRep(data: try Data(contentsOf: image)) else { throw NativeFailure(message: "Invalid Z-Image PNG") }
        try check(bitmap.pixelsWide == 512 && bitmap.pixelsHigh == 512, "Wrong image dimensions")
        reports["z-image"] = try JSONSerialization.jsonObject(with: Data(zJob.resultJSON!.utf8))
        print("PASS Z-Image GPU 512x512")
        try await store.unload()
        studio.selectModel("ltx-2.5-distilled")
        studio.draft.prompt = "A red fox slowly walking through a snowy forest, gentle camera movement, cinematic morning light."
        for operation in ["video.generate", "video.image"] {
            if operation == "video.image" { await studio.addFiles([image]) }
            studio.changeOperation(operation)
            let name = operation == "video.generate" ? "ltx-t2v-gpu" : "ltx-i2v-gpu"
            let request = try studio.draft.request(output: root.appendingPathComponent("outputs/\(name).mp4"))
            try check(request.execution == "gpu" && request.allow_approximation != true && LTXWorker.accepts(request), "LTX not exact GPU worker")
            let job = try await store.generate(modelURL: URL(fileURLWithPath: args[2]), request: request)
            let result = try JSONSerialization.jsonObject(with: Data(job.resultJSON!.utf8)) as! [String: Any]
            let asset = AVURLAsset(url: URL(fileURLWithPath: request.output))
            let tracks = try await asset.loadTracks(withMediaType: .video)
            try check(!tracks.isEmpty, "Video track missing")
            let size = try await tracks[0].load(.naturalSize)
            let duration = try await asset.load(.duration)
            try check(Int(size.width) == request.width && Int(size.height) == request.height, "Wrong video dimensions")
            try check(abs(duration.seconds - Double(request.frames) / 24) < 0.1, "Wrong video duration")
            try check(!store.canUnload && !store.busy, "LTX worker retained a model session")
            reports[name] = result
            print("PASS \(name) \(request.width)x\(request.height)x\(request.frames)")
        }
        try JSONSerialization.data(withJSONObject: reports, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print("PASS desktop GPU three-operation acceptance")
    }
}
