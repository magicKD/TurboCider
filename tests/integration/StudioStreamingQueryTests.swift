import Foundation

private actor ControlledStreamingQueries {
    private var pending: [String: CheckedContinuation<NativeStreamingOptions, Never>] = [:]
    private var cancelled: Set<String> = []
    func query(_ request: NativeRequestV2) async -> NativeStreamingOptions {
        let key = request.inputs.first?.text ?? "missing"
        return await withTaskCancellationHandler {
            await withCheckedContinuation { pending[key] = $0 }
        } onCancel: { Task { await self.markCancelled(key) } }
    }
    private func markCancelled(_ key: String) { cancelled.insert(key) }
    func waiting(_ key: String) -> Bool { pending[key] != nil }
    func wasCancelled(_ key: String) -> Bool { cancelled.contains(key) }
    func finish(_ key: String) {
        let value = NativeStreamingOptions(schema_version: 2, catalog_revision: key, query_status: "artifact_checked",
            execution_container: "cli_worker", device: NativeStreamingDevice(gpu: "fixture", physical_memory_bytes: 16 << 30, device_class: "fixture"),
            targets: [NativeStreamingTargetOption(target_request_memory_bytes: 10 << 30, status: "available", reason_code: nil,
                preset_id: "fixture", preset_revision: 1, calibrated_request_bytes: 8 << 30, memory_scope: "request", release_channel: "public")])
        pending.removeValue(forKey: key)!.resume(returning: value)
    }
}

enum StudioStreamingQueryTests {
    @MainActor static func run(root: URL) async throws {
        let control = ControlledStreamingQueries()
        let studio = StudioState(directory: root, streamingOptionsProvider: { request, _ in await control.query(request) })
        studio.draft.modelID = "z-image-turbo"; studio.draft.steps = 9; studio.draft.audio = false
        studio.draft.profilePath = ""; studio.draft.acceleration = StudioAcceleration(policy: "gpu")
        studio.draft.ltxBackend = "c_metal"; studio.draft.ltxAccelerationMode = "quality"
        func wait(_ key: String, cancellation: Bool = false) async throws {
            for _ in 0..<500 {
                let value = cancellation ? await control.wasCancelled(key) : await control.waiting(key)
                if value { return }
                try await Task.sleep(for: .milliseconds(10))
            }
            throw NativeFailure(message: "Query test did not reach \(key)")
        }
        studio.draft.prompt = "old"
        let old = Task { await studio.refreshStreamingOptions() }
        try await wait("old")
        studio.draft.prompt = "new"
        let new = Task { await studio.refreshStreamingOptions() }
        try await wait("new"); try await wait("old", cancellation: true)
        await control.finish("new"); await new.value
        precondition(studio.streamingOptions?.catalog_revision == "new" && studio.draft.streaming.selection == .off)
        await control.finish("old"); await old.value
        precondition(studio.streamingOptions?.catalog_revision == "new" && !studio.streamingOptionsLoading)
        studio.draft.prompt = "cancel"
        let cancel = Task { await studio.refreshStreamingOptions() }
        try await wait("cancel"); cancel.cancel(); try await wait("cancel", cancellation: true)
        await control.finish("cancel"); await cancel.value
        precondition(studio.streamingOptions == nil && studio.streamingOptionsError == nil && !studio.streamingOptionsLoading)
        studio.draft.prompt = "changed-key"
        let changed = Task { await studio.refreshStreamingOptions() }
        try await wait("changed-key"); studio.draft.width += 64
        await control.finish("changed-key"); await changed.value
        precondition(studio.streamingOptions == nil)
        studio.draft.prompt = "installation"
        let oldKey = studio.streamingQueryKey
        let installation = Task { await studio.refreshStreamingOptions() }
        try await wait("installation")
        studio.invalidateStreamingInstallation()
        precondition(studio.streamingQueryKey != oldKey)
        try await wait("installation", cancellation: true)
        await control.finish("installation"); await installation.value
        precondition(studio.streamingOptions == nil && !studio.streamingOptionsLoading)
        studio.draft.prompt = "unsupported"
        let active = Task { await studio.refreshStreamingOptions() }
        try await wait("unsupported")
        studio.draft.modelID = "unsupported-model"
        await studio.refreshStreamingOptions()
        try await wait("unsupported", cancellation: true)
        await control.finish("unsupported"); await active.value
        precondition(studio.streamingOptions == nil && studio.streamingOptionsError == nil && !studio.streamingOptionsLoading)
        // Default production route: no candidates means no missing-model open.
        let empty = StudioState(directory: root.appendingPathComponent("production"))
        empty.draft.modelID = "flux2-klein-4b"; empty.draft.audio = false; empty.draft.steps = 4
        empty.draft.acceleration = StudioAcceleration(policy: "gpu"); empty.draft.profilePath = ""
        empty.draft.modelPaths["flux2-klein-4b"] = "/missing/must-not-open"
        await empty.refreshStreamingOptions()
        precondition(empty.streamingOptions?.execution_container == "cli_worker" && empty.streamingOptions?.query_status == "catalog_empty")
        precondition(empty.draft.streaming.selection == .off)
        print("PASS: Studio worker discovery, superseded/cancelled query propagation, stale-key/installation generation rejection and default Off")
    }
}
