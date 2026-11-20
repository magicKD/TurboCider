import Foundation

/// Bounded, correlated telemetry. It never authorizes execution or publication.
/// The lock protects incremental framing when called from the supervisor actor.
final class WorkerEventStream: @unchecked Sendable {
    enum Event: Sendable { case resolved(NativeStreamingResolution), progress(NativeEvent) }
    private struct Header: Decodable {
        let event_schema_version: Int
        let kind: String
        let sequence: Int
        let job_id: String
        let request_id: String
        let request_digest: String
        let execution_container: String
        let runtime_fingerprint: String
    }
    private let lock = NSLock()
    private let jobID: String, reference: PublicImageWorker.Reference, request: NativeRequestV2
    private let callback: @Sendable (Event) -> Void
    private var framing = WorkerEventFraming(invalidMessage: "worker_event_invalid")
    private var sequence = 0
    private var nativeSequence = -1
    private var resolution: NativeStreamingResolution?
    init(jobID: UUID, reference: PublicImageWorker.Reference, request: NativeRequestV2,
         callback: @escaping @Sendable (Event) -> Void) {
        self.jobID = jobID.uuidString.lowercased(); self.reference = reference
        self.request = request; self.callback = callback
    }
    private func invalid() -> NativeFailure { NativeFailure(message: "worker_event_invalid") }
    func consume(_ data: Data) throws {
        lock.lock(); defer { lock.unlock() }
        try framing.consume(data, onEvent: handleEvent)
    }
    private func handleEvent(_ json: Data) throws {
        let h = try JSONDecoder().decode(Header.self, from: json)
        guard h.event_schema_version == 1, h.sequence > sequence,
              h.job_id == jobID, h.request_id == reference.requestID.uuidString.lowercased(),
              h.request_digest == reference.requestDigest, h.execution_container == "cli_worker",
              h.runtime_fingerprint == reference.runtimeFingerprint,
              let object = try JSONSerialization.jsonObject(with: json) as? [String: Any],
              let payload = object["payload"] as? [String: Any] else { throw invalid() }
        let bytes = try JSONSerialization.data(withJSONObject: payload)
        switch h.kind {
        case "resolved":
            guard resolution == nil else { throw invalid() }
            let value = try JSONDecoder().decode(NativeStreamingResolution.self, from: bytes)
            _ = try value.binding(request)
            guard value.selection.execution_container == "cli_worker" else { throw invalid() }
            resolution = value; callback(.resolved(value))
        case "progress":
            let value = try JSONDecoder().decode(NativeEvent.self, from: bytes)
            guard resolution != nil, value.sequence > nativeSequence, !value.phase.isEmpty,
                  value.phase.utf8.count <= 128, value.completed >= 0, value.total >= 0,
                  value.completed <= value.total, value.elapsed_seconds.isFinite, value.elapsed_seconds >= 0 else { throw invalid() }
            nativeSequence = value.sequence; callback(.progress(value))
        default: throw invalid()
        }
        sequence = h.sequence
    }
    func finish(resolution terminal: NativeStreamingResolution?) throws {
        lock.lock(); defer { lock.unlock() }
        try framing.finish()
        // No telemetry is required, but if supplied it must agree with success.
        if let resolution, let terminal {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            guard try encoder.encode(resolution) == encoder.encode(terminal) else { throw invalid() }
        }
    }
}
