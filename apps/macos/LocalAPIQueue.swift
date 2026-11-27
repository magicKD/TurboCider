import Foundation

/// The schema-1 loopback API, backed by the App's persistent generation queue.
@MainActor struct LocalAPIQueue {
    let store: NativeJobStore

    private func object(_ data: Data) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NativeFailure(message: "Expected a native JSON object.")
        }
        return value
    }

    private func record(_ job: NativeJob) throws -> [String: Any] {
        guard let payload = job.automationRequestJSON else { throw NativeFailure(message: "Unknown API job.") }
        let state = ["preparing", "finalizing"].contains(job.state) ? "running"
            : job.state == "cleanup_pending" ? "cancelling" : job.state
        var result: [String: Any] = ["schema_version": 1, "id": job.id.uuidString, "state": state,
            "created_at": job.createdAt.timeIntervalSince1970, "request": try object(Data(payload.utf8)),
            "progress": ["phase": job.phase, "completed": job.completed, "total": job.total, "elapsed": job.elapsed]]
        result["error"] = job.error
        result["model_path"] = job.modelPath
        if job.hasOutput, let value = job.resultJSON { result["result"] = try object(Data(value.utf8)) }
        return result
    }

    func call(_ request: [String: Any]) async throws -> [String: Any] {
        do { return try await perform(request) }
        catch is CancellationError { throw CancellationError() }
        catch { throw LocalHTTPGateway.Failure(status: 400, message: error.localizedDescription) }
    }

    private func perform(_ request: [String: Any]) async throws -> [String: Any] {
        switch request["action"] as? String {
        case "models": return try object(Data(NativeEngine.models().utf8))
        case "installations": return try object(JSONEncoder().encode(LibraryInventory.read()))
        case "workflows": return try object(NativeEngine.workflows())
        case "service_status":
            return ["pid": Int(getpid()), "queue": "app", "active_job": store.activeJob?.id.uuidString ?? "",
                    "history_count": apiJobs.count, "session_model": store.loadedModelID ?? "",
                    "session_open": store.loadedModelID != nil, "external_worker_active": store.activeJob?.supervisedWorker != nil,
                    "closing": false]
        case "jobs":
            let jobs = apiJobs
            let offset = min(jobs.count, max(0, request["offset"] as? Int ?? 0))
            let end = min(jobs.count, offset + min(100, max(1, request["limit"] as? Int ?? 20)))
            return ["jobs": try jobs[offset..<end].map(record), "total": jobs.count,
                    "next_offset": end < jobs.count ? end as Any : NSNull()]
        case "status", "cancel":
            guard let text = request["id"] as? String, let id = UUID(uuidString: text),
                  let job = apiJobs.first(where: { $0.id == id }) else { throw NativeFailure(message: "Unknown API job.") }
            if request["action"] as? String == "cancel" { try store.cancelGeneration(id) }
            return try record(store.jobs.first(where: { $0.id == id }) ?? job)
        case "plan", "submit":
            guard let value = request["request"] as? [String: Any] else {
                throw NativeFailure(message: "Expected a native generation request.")
            }
            let payload = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
            let model = (request["model_path"] as? String).map { URL(fileURLWithPath: $0) }
            let plan = try await Task.detached { try NativeEngine.planSchema1(payload, installation: model) }.value
            try Task.checkCancellation()
            let info = try object(plan)
            guard request["action"] as? String == "submit" else { return info }
            guard let model, let output = value["output"] as? String,
                  let modelID = info["model"] as? String, let operation = info["operation"] as? String,
                  let shape = info["requested_shape"] as? [Int], shape.count == 3,
                  let descriptor = StudioModel.catalog().first(where: { $0.id == modelID && $0.executor }) else {
                throw NativeFailure(message: "Native plan is missing executable request metadata.")
            }
            // Persist the original schema for execution. The typed summary only
            // describes the job to desktop history and output validation.
            let defaults: [String: Any] = [
                "schema_version": 1, "model": modelID, "prompt": "", "output": output,
                "operation": operation, "width": shape[0], "height": shape[1], "frames": shape[2],
                "steps": descriptor.default_steps, "seed": 42, "execution": "gpu", "dynamic_text": true,
                "fps": descriptor.default_fps ?? (modelID == "relayvsr-2x" ? 0 : 24),
                "audio": descriptor.default_audio ?? false
            ]
            var summary = value
            for (key, fallback) in defaults where summary[key] == nil { summary[key] = fallback }
            summary["residency"] = info["residency"]
            if let budget = info["memory_budget_bytes"] as? NSNumber { summary["memory_budget_bytes"] = budget }
            if let inputs = value["inputs"] as? [[String: Any]] {
                if let prompt = inputs.first(where: { $0["kind"] as? String == "text" })?["text"] as? String {
                    summary["prompt"] = prompt
                }
                summary["inputs"] = inputs.filter { $0["kind"] as? String != "text" }
            }
            if let loras = value["loras"] as? [[String: Any]] {
                summary["loras"] = loras.map { adapter in
                    var item = adapter
                    if item["strength"] == nil { item["strength"] = 1.0 }
                    if item["role"] == nil { item["role"] = "transformer" }
                    return item
                }
            }
            let native = try JSONDecoder().decode(NativeRequest.self, from: JSONSerialization.data(withJSONObject: summary))
            let ltx = modelID == "ltx-2.5-distilled"
            guard info["executable"] as? Bool == true || ltx else { throw NativeFailure(message: "Model executor unavailable.") }
            let job = NativeJob(id: UUID(), createdAt: Date(), request: native, state: "queued", phase: "queued",
                completed: 0, total: native.steps, elapsed: 0, modelPath: model.path,
                automationRequestJSON: String(decoding: payload, as: UTF8.self))
            try store.enqueueGenerationJobs([job])
            return ["id": job.id.uuidString, "state": "queued"]
        default: throw NativeFailure(message: "Unknown HTTP API action.")
        }
    }

    private var apiJobs: [NativeJob] { store.jobs.filter { $0.automationRequestJSON != nil } }
}
