import Foundation

@main
struct ModelSourceVerificationTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 3 else { throw NativeFailure(message: "usage: model-source-tests MODEL_ID MODEL_PATH") }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = NativeJobStore(directory: directory)
        let model = URL(fileURLWithPath: args[2])
        store.externalServiceActive = true
        do {
            try await store.verifyModelSources(modelURL: model, modelID: args[1])
            throw NativeFailure(message: "verification admitted while external service was active")
        } catch let failure as NativeFailure {
            guard failure.localizedDescription.contains("空闲") else { throw failure }
        }
        guard !store.busy && store.loadedModelID == nil else { throw NativeFailure(message: "rejected verification mutated session") }
        store.externalServiceActive = false
        let cancelled = Task { try await store.verifyModelSources(modelURL: model, modelID: args[1]) }
        try await Task.sleep(nanoseconds: 100_000_000)
        store.cancel()
        do {
            try await cancelled.value
            throw NativeFailure(message: "verification ignored cancellation")
        } catch is CancellationError {}
        guard !store.busy && store.sessionState == "文件校验已取消" else { throw NativeFailure(message: "cancelled verification left busy state") }
        try await store.verifyModelSources(modelURL: model, modelID: args[1])
        guard !store.busy, store.loadedModelID == args[1], store.sessionReport == nil,
              store.sessionState.hasPrefix("已校验 ") else { throw NativeFailure(message: "verification did not publish a clean completion state") }
        try await store.unload()
        guard store.loadedModelID == nil && !store.busy else { throw NativeFailure(message: "verified session did not unload") }
        print("PASS model-library verification: busy admission, cancellation, retry, clean report state and unload")
    }
}
