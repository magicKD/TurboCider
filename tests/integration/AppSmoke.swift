import Foundation
import AppKit

@main
struct AppSmoke {
    @MainActor static func main() async throws {
        guard (3...4).contains(CommandLine.arguments.count) else {
            throw NativeFailure(message: "app-smoke MODEL OUTPUT_DIRECTORY [MODEL_ID]")
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[2])
        let modelID = CommandLine.arguments.count == 4 ? CommandLine.arguments[3] : "flux2-klein-4b"
        guard let model = StudioModel.catalog().first(where: { $0.id == modelID && $0.executor }) else {
            throw NativeFailure(message: "app-smoke does not recognize an executable model: \(modelID)")
        }
        let store = NativeJobStore(directory: root)
        let output = root.appendingPathComponent(model.isVideo ? "app-generated.mp4" : "app-generated.png")
        var request = NativeRequest(prompt: "A red fox sitting in a snowy forest, soft morning light, detailed photography.", output: output.path)
        request.model = modelID
        request.operation = model.operations.first
        request.width = model.default_width; request.height = model.default_height
        request.steps = model.default_steps; request.frames = model.default_frames
        request.fps = model.default_fps; request.audio = model.default_audio
        request.residency = model.default_residency
        let job = try await store.generate(modelURL: URL(fileURLWithPath: CommandLine.arguments[1]), request: request)
        let attributes = try FileManager.default.attributesOfItem(atPath: request.output)
        let outputBytes = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        let mediaValid = model.isVideo || NSImage(contentsOfFile: request.output) != nil
        guard job.state == "succeeded", !store.busy, outputBytes > 0, mediaValid else {
            throw NativeFailure(message: "App generation verification failed")
        }
        let restored = NativeJobStore(directory: root)
        guard restored.jobs.first?.id == job.id, restored.jobs.first?.state == "succeeded" else { throw NativeFailure(message: "App persistence verification failed") }
        print(job.resultJSON ?? "{}")
    }
}
