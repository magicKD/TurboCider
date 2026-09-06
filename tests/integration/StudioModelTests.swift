import AppKit
import Foundation
import Combine

@main
struct StudioModelTests {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 3 else { throw NativeFailure(message: "studio-model-tests MODEL OUTPUT_DIRECTORY") }
        let model = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = URL(fileURLWithPath: CommandLine.arguments[2])
        let store = NativeJobStore(directory: root)
        let studio = StudioState(directory: root)
        func check(_ value: @autoclosure () throws -> Bool, _ reason: String) throws {
            guard try value() else { throw NativeFailure(message: reason) }
        }
        func rejectAsync(_ work: () async throws -> Void) async throws {
            do { try await work() } catch { return }
            throw NativeFailure(message: "Expected busy rejection")
        }
        var checks: [String] = []
        print("Loading actual image weights…")
        try await store.load(modelURL: model)
        try check(store.canUnload && store.loadedPath == model.path && store.sessionState.contains("权重已加载"), "Load did not publish ready weights")
        let loadReport = store.resourceReport!
        let loaded = try JSONSerialization.jsonObject(with: Data(loadReport.utf8)) as! [String: Any]
        try check((loaded["mlx_active_bytes"] as? UInt64 ?? 0) > 7_000_000_000, "Load only registered lazy weights without materializing them")
        checks.append("explicit_image_weight_load")
        studio.draft.modelPaths["flux2-klein-4b"] = model.path
        studio.draft.width = 256; studio.draft.height = 256
        studio.draft.prompt = "A ceramic blue teapot on a wooden table, studio photography."
        studio.draft.randomSeed = false; studio.draft.seedText = "42"
        let request = try studio.draft.request(output: root.appendingPathComponent("outputs/text.png"))
        let work = Task { try await store.generate(modelURL: model, request: request) }
        while !store.busy { await Task.yield() }
        try await rejectAsync { try await store.unload() }
        try await rejectAsync { _ = try await store.generate(modelURL: model, request: request) }
        studio.draft.prompt = "This edit must not alter the running request"
        let original = try await work.value
        try check(original.request.prompt == request.prompt && original.state == "succeeded", "Mutable draft changed generation")
        try check(NSImage(contentsOfFile: request.output) != nil && original.secondsPerStep != nil, "Image or genuine step telemetry missing")
        checks += ["text_to_image", "single_active_job", "busy_unload_rejected", "immutable_request", "completed_step_telemetry"]
        print("Testing single-image transform…")
        await studio.addFiles([URL(fileURLWithPath: original.request.output)])
        studio.changeOperation("image.transform"); studio.draft.prompt = request.prompt; studio.draft.strength = 0.5
        let transformRequest = try studio.draft.request(output: root.appendingPathComponent("outputs/transform.png"))
        var transformTotals: Set<Int> = []
        let observation = store.$jobs.sink { jobs in
            if let job = jobs.first, job.request.operation == "image.transform", job.phase == "denoise" { transformTotals.insert(job.total) }
        }
        let transformed = try await store.generate(modelURL: model, request: transformRequest)
        observation.cancel()
        try check(transformTotals == [2], "UI sampling progress did not use actual transform steps")
        let transformResult = try JSONSerialization.jsonObject(with: Data(transformed.resultJSON!.utf8)) as! [String: Any]
        try check(transformResult["actual_denoise_steps"] as? Int == 2, "Transform strength step count changed")
        checks.append("single_image_transform_actual_steps")
        print("Testing ordered multi-image edit…")
        await studio.addFiles([URL(fileURLWithPath: transformed.request.output)])
        studio.changeOperation("image.edit")
        let editRequest = try studio.draft.request(output: root.appendingPathComponent("outputs/edit.png"))
        let edited = try await store.generate(modelURL: model, request: editRequest)
        let editResult = try JSONSerialization.jsonObject(with: Data(edited.resultJSON!.utf8)) as! [String: Any]
        try check((editResult["reference_tokens"] as? Int ?? 0) > 0 && edited.request.inputs?.count == 2, "Multiple reference images not consumed")
        try check(NSImage(contentsOfFile: edited.request.output) != nil, "Edit result cannot decode")
        checks.append("ordered_multi_reference_edit")
        studio.save()
        let restored = StudioState(directory: root)
        try check(restored.draft.assets.count == 2, "Draft input persistence failed")
        let history = NativeJobStore(directory: root)
        try check(history.jobs.count == 3 && history.jobs.allSatisfy { $0.state == "succeeded" }, "Job history persistence failed")
        checks.append("asset_and_job_persistence")
        print("Releasing session, checking actual allocator resources…")
        try await store.unload()
        let unloadReport = store.resourceReport!
        let memory = try JSONSerialization.jsonObject(with: Data(unloadReport.utf8)) as! [String: Any]
        try check((memory["mlx_active_bytes"] as? UInt64 ?? .max) < 128 * 1024 * 1024, "Unload retained model weights")
        try check(!store.canUnload && store.loadedPath == nil, "Unload state stale")
        checks.append("unload_releases_allocator_weights")
        print("Comparing App requests with direct engine outputs…")
        let engine = try await NativeEngine.open(modelURL: model, modelID: request.model)
        for source in [original, transformed, edited] {
            var direct = source.request
            direct.output = root.appendingPathComponent("outputs/direct-\(URL(fileURLWithPath: source.request.output).lastPathComponent)").path
            _ = try await engine.generate(direct) { _ in }
            try check(try Data(contentsOf: URL(fileURLWithPath: source.request.output)) == Data(contentsOf: URL(fileURLWithPath: direct.output)), "App/direct PNG parity failed for \(source.request.operation!)")
        }
        _ = try await engine.unload()
        checks.append("app_direct_png_parity_three_operations")
        print("Checking generation after unload and cancellation recovery…")
        var cancelledRequest = request; cancelledRequest.output = root.appendingPathComponent("outputs/cancelled.png").path
        let cancellation = Task { try await store.generate(modelURL: model, request: cancelledRequest) }
        while !store.busy { await Task.yield() }
        store.cancel()
        do { _ = try await cancellation.value; throw NativeFailure(message: "Cancelled job succeeded before execution") }
        catch is CancellationError {}
        try check(!FileManager.default.fileExists(atPath: cancelledRequest.output), "Cancelled output was committed")
        var reloaded = request; reloaded.output = root.appendingPathComponent("outputs/reloaded.png").path
        let final = try await store.generate(modelURL: model, request: reloaded)
        try check(try Data(contentsOf: URL(fileURLWithPath: final.request.output)) == Data(contentsOf: URL(fileURLWithPath: original.request.output)), "Reload changed deterministic result")
        try await store.unload()
        checks += ["cancel_before_generation", "reload_after_unload_same_output"]
        let report: [String: Any] = ["passed": true, "checks": checks, "load": try JSONSerialization.jsonObject(with: Data(loadReport.utf8)), "unload": memory, "text": try JSONSerialization.jsonObject(with: Data(original.resultJSON!.utf8)), "transform": transformResult, "edit": editResult]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print("PASS: \(checks.joined(separator: ", "))")
    }
}
