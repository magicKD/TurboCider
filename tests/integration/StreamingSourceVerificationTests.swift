import Foundation

@main
struct StreamingSourceVerificationTests {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 3 else {
            throw NativeFailure(message: "usage: source-verification-tests MODEL_ID MODEL_PATH")
        }
        let engine = try await NativeEngine.open(modelURL: URL(fileURLWithPath: args[2]), modelID: args[1])
        let started = Date()
        let first = try await engine.verifyStreamingSources()
        let second = try await engine.verifyStreamingSources()
        guard let initial = try JSONSerialization.jsonObject(with: first) as? [String: Any],
              let cached = try JSONSerialization.jsonObject(with: second) as? [String: Any],
              initial["status"] as? String == "verified",
              initial["model"] as? String == args[1],
              let digest = initial["artifact_manifest_digest"] as? String,
              digest.count == 64,
              digest == cached["artifact_manifest_digest"] as? String,
              let files = initial["files"] as? [[String: Any]], !files.isEmpty,
              (cached["verification_bytes_read"] as? NSNumber)?.uint64Value == 0,
              (cached["verification_cache_hits"] as? NSNumber)?.intValue == files.count else {
            throw NativeFailure(message: "native source proof/cache contract failed")
        }
        print("PASS Swift source verification: model=\(args[1]) files=\(files.count) elapsed=\(Date().timeIntervalSince(started))s cached payload=0")
    }
}
