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
        let store = NativeJobStore(directory: root)
        var request = NativeRequest(prompt: "A red fox sitting in a snowy forest, soft morning light, detailed photography.", output: root.appendingPathComponent("app-generated.png").path)
        request.model = modelID
        request.width = 256; request.height = 256
        let job = try await store.generate(modelURL: URL(fileURLWithPath: CommandLine.arguments[1]), request: request)
        guard job.state == "succeeded", !store.busy, NSImage(contentsOfFile: request.output) != nil else { throw NativeFailure(message: "App generation verification failed") }
        let restored = NativeJobStore(directory: root)
        guard restored.jobs.first?.id == job.id, restored.jobs.first?.state == "succeeded" else { throw NativeFailure(message: "App persistence verification failed") }
        print(job.resultJSON ?? "{}")
    }
}
