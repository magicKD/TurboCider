import Foundation
import Combine

struct NativeJob: Codable, Identifiable, Sendable {
    let id: UUID
    let createdAt: Date
    let request: NativeRequest
    var state: String
    var phase: String
    var completed: Int
    var total: Int
    var elapsed: Double
    var error: String?
    var resultJSON: String?
}

@MainActor
final class NativeJobStore: ObservableObject {
    @Published private(set) var jobs: [NativeJob] = []
    @Published private(set) var busy = false
    @Published private(set) var storageError: String?
    let directory: URL
    private var engine: NativeEngine?
    private var modelPath: String?
    private var modelID: String?
    private var activeID: UUID?
    private var cancelRequested = false
    private var lastPersist = Date.distantPast

    init(directory: URL) {
        self.directory = directory
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let file = directory.appendingPathComponent("jobs.json")
            if FileManager.default.fileExists(atPath: file.path) {
                jobs = try JSONDecoder().decode([NativeJob].self, from: Data(contentsOf: file))
                for i in jobs.indices where ["running", "preparing", "cancelling"].contains(jobs[i].state) {
                    jobs[i].state = "interrupted"
                    jobs[i].error = "The previous engine session stopped before completion."
                }
                try persist()
            }
        } catch { storageError = error.localizedDescription }
    }
    private func persist() throws {
        try JSONEncoder().encode(jobs).write(to: directory.appendingPathComponent("jobs.json"), options: .atomic)
        lastPersist = Date()
    }
    func cancel() {
        guard busy else { return }
        cancelRequested = true
        engine?.cancel()
        if let id = activeID, let i = jobs.firstIndex(where: { $0.id == id }) {
            jobs[i].state = "cancelling"
            try? persist()
        }
    }
    func generate(modelURL: URL, request: NativeRequest) async throws -> NativeJob {
        guard !busy else { throw NativeFailure(message: "An inference job is already running") }
        guard storageError == nil else { throw NativeFailure(message: storageError!) }
        _ = try NativeEngine.plan(request)
        busy = true
        cancelRequested = false
        let id = UUID()
        activeID = id
        let job = NativeJob(id: id, createdAt: Date(), request: request, state: "preparing", phase: "prepare", completed: 0, total: 1, elapsed: 0)
        jobs.insert(job, at: 0)
        defer { busy = false; activeID = nil }
        do {
            try persist()
            if modelPath != modelURL.path || modelID != request.model || engine == nil {
                engine = nil
                engine = try NativeEngine(modelURL: modelURL, modelID: request.model)
                modelPath = modelURL.path
                modelID = request.model
            }
            guard let engine else { throw NativeFailure(message: "Engine unavailable") }
            let result = try await engine.generate(request) { [weak self] event in
                DispatchQueue.main.async { [weak self] in self?.receive(event, id: id) }
            }
            guard let i = jobs.firstIndex(where: { $0.id == id }) else { throw NativeFailure(message: "Missing job") }
            jobs[i].state = "succeeded"
            jobs[i].phase = "complete"
            jobs[i].resultJSON = String(decoding: result, as: UTF8.self)
            try persist()
            return jobs[i]
        } catch {
            if let i = jobs.firstIndex(where: { $0.id == id }) {
                jobs[i].state = error is CancellationError ? "cancelled" : "failed"
                jobs[i].error = error.localizedDescription
                do { try persist() } catch { storageError = error.localizedDescription }
            }
            throw error
        }
    }
    private func receive(_ event: NativeEvent, id: UUID) {
        guard activeID == id, let i = jobs.firstIndex(where: { $0.id == id }),
              ["preparing", "running", "cancelling"].contains(jobs[i].state) else { return }
        if cancelRequested { engine?.cancel() }
        jobs[i].state = cancelRequested ? "cancelling" : "running"
        jobs[i].phase = event.phase
        jobs[i].completed = event.completed
        jobs[i].total = event.total
        jobs[i].elapsed = event.elapsed_seconds
        if Date().timeIntervalSince(lastPersist) > 1 {
            do { try persist() } catch { storageError = error.localizedDescription; cancel() }
        }
    }
}
