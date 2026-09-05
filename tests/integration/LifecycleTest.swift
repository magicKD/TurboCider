import Foundation
import AppKit

@main
struct LifecycleTest {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 3 else { throw NativeFailure(message: "lifecycle-test MODEL OUTPUT_DIRECTORY") }
        let root = URL(fileURLWithPath: CommandLine.arguments[2]);try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let model = URL(fileURLWithPath: CommandLine.arguments[1])
        let engine = try NativeEngine(modelURL: model)
        var request = NativeRequest(prompt: "A ceramic blue teapot on a wooden table, studio photography.", output: root.appendingPathComponent("cancelled.png").path)
        request.width = 256; request.height = 256
        var cancelled = false
        do { _ = try await engine.generate(request) { event in if event.phase == "denoise" { engine.cancel() } } }
        catch is CancellationError { cancelled = true }
        guard cancelled, !FileManager.default.fileExists(atPath: request.output) else { throw NativeFailure(message: "Cancellation did not stop before export") }
        var reports: [[String: Any]] = []
        for i in 0..<3 {
            request.output = root.appendingPathComponent("repeat-\(i).png").path
            let result = try await engine.generate(request) { _ in }
            reports.append(try JSONSerialization.jsonObject(with: result) as! [String: Any])
        }
        let first = try Data(contentsOf: root.appendingPathComponent("repeat-0.png"))
        for i in 1..<3 { guard try Data(contentsOf: root.appendingPathComponent("repeat-\(i).png")) == first else { throw NativeFailure(message: "Repeated generation changed image") } }
        guard reports.allSatisfy({ $0["prompt_cache_hit"] as? Bool == true }) else { throw NativeFailure(message: "Warm prompt cache was not reused") }
        let memories = reports.map { ($0["memory"] as! [String:Any])["mlx_active_bytes"] as! UInt64 }
        guard memories.max()! - memories.min()! < 64*1024*1024 else { throw NativeFailure(message: "Repeated session retained unexpected memory") }
        request.output = root.appendingPathComponent("cancel-at-export.png").path
        var exportCancelled = false
        do { _ = try await engine.generate(request) { event in if event.phase == "export" && event.completed == 0 { engine.cancel() } } }
        catch is CancellationError { exportCancelled = true }
        guard exportCancelled, !FileManager.default.fileExists(atPath: request.output) else { throw NativeFailure(message: "Export cancellation left a committed file") }
        request.prompt = "一只橙色的猫坐在窗边，柔和的阳光，摄影。"
        request.output = root.appendingPathComponent("chinese-prompt.png").path
        let changed = try await engine.generate(request) { _ in }
        guard (try JSONSerialization.jsonObject(with: changed) as! [String:Any])["prompt_cache_hit"] as? Bool == false else { throw NativeFailure(message: "Changed prompt reused conditioning") }
        request.dynamic_text = false
        request.output = root.appendingPathComponent("fixed-text.png").path
        let fixed = try await engine.generate(request) { _ in }
        guard (try JSONSerialization.jsonObject(with: fixed) as! [String:Any])["text_tokens"] as? Int == 512 else { throw NativeFailure(message: "Fixed text bucket not honored") }
        let store = NativeJobStore(directory: root.appendingPathComponent("jobs"))
        request.dynamic_text = true;request.output = root.appendingPathComponent("store.png").path
        let job = try await store.generate(modelURL: model, request: request)
        let restored = NativeJobStore(directory: store.directory)
        guard restored.jobs.first?.id == job.id, restored.jobs.first?.state == "succeeded" else { throw NativeFailure(message: "Persistent job not restored") }
        let report: [String:Any] = ["passed":true,"checks":["cancel_before_export","cancel_at_export_boundary","same_session_recovery","three_identical_warm_outputs","bounded_retained_memory","changed_prompt_cache_invalidation","fixed_512_text_bucket","job_persistence"],"warm_reports":reports,"changed_prompt":try JSONSerialization.jsonObject(with: changed),"fixed_text":try JSONSerialization.jsonObject(with: fixed)]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys]);try data.write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print(String(decoding:data,as:UTF8.self))
    }
}
