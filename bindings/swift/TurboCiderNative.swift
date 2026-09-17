import Foundation
import CTurboCider

public struct NativeInput: Codable, Sendable, Identifiable {
    public var id: String { path + role }
    public var kind: String
    public var role: String
    public var path: String
    public var strength: Double?
    public init(kind: String, role: String, path: String, strength: Double? = nil) {
        self.kind = kind; self.role = role; self.path = path; self.strength = strength
    }
}
public struct NativeLoRA: Codable, Sendable, Identifiable {
    public var id: String { path + role }
    public var path: String
    public var strength: Double
    public var role: String
    public init(path: String, strength: Double = 1.0, role: String = "transformer") {
        self.path = path; self.strength = strength; self.role = role
    }
}
public struct NativeRequest: Codable, Sendable {
    public var schema_version = 1
    public var model = "flux2-klein-4b"
    public var prompt: String
    public var output: String
    public var width = 512
    public var height = 512
    public var steps = 4
    public var seed = 42
    public var frames = 1
    public var execution = "gpu"
    public var dynamic_text = true
    public var noise_path: String?
    public var compile_gpu: Bool?
    public var ane_manifest: String?
    public var allow_approximation: Bool?
    public var dump_tensors: String?
    public var operation: String?
    public var inputs: [NativeInput]?
    public var profile: String?
    public var residency: String?
    public var memory_budget_bytes: UInt64?
    public var quantized_cache: String?
    public var fps: Int?
    public var audio: Bool?
    public var ltx_backend: String?
    public var ltx_fast_av: Bool?
    public var ltx_video_attention_batch: Bool?
    public var ltx_sol_stage1: Bool?
    public var ltx_sol_stage2: Bool?
    public var ltx_sol_tau: Double?
    public var ltx_sol_dense_edge_blocks: Int?
    public var ltx_sol_dense_edge_steps: Int?
    public var ltx_stage2_text_rows: Int?
    public var loras: [NativeLoRA]?
    public var lora_strategy: String?
    public init(prompt: String, output: String) { self.prompt = prompt; self.output = output }
}
public struct NativeStreamingSelectorV2: Codable, Sendable, Equatable {
    public var schema_version = 2
    public var enabled = true
    public var selection = "memory_tier"
    public var retention = "request"
    public var target_request_memory_bytes: UInt64
    public var preset_id: String?
    public var preset_revision: UInt32?
    public var catalog_revision: String?
    public var expected_resolution_digest: String?
    public init(targetBytes: UInt64) { target_request_memory_bytes = targetBytes }
}
public struct NativeInputV2: Codable, Sendable {
    public var kind: String
    public var role: String
    public var path: String?
    public var text: String?
    public var strength: Double?
}
public struct NativeOutputV2: Codable, Sendable {
    public var kind: String
    public var path: String
    public var width: Int
    public var height: Int
    public var frames: Int
    public var fps: Int
    public var audio: Bool
}
public struct NativeSamplingV2: Codable, Sendable {
    public var seed: Int
    public var steps: Int
}
public struct NativeExecutionV2: Codable, Sendable {
    public var policy: String
    public var profile: String?
    public var ane_manifest: String?
    public var encoder_ane_manifest: String?
    public var allow_approximation: Bool?
    public var quantized_cache: String?
    public var warmup_iterations: Int?
    public var ltx_backend: String?
    public var ltx_fast_av: Bool?
    public var ltx_video_attention_batch: Bool?
    public var ltx_sol_stage1: Bool?
    public var ltx_sol_stage2: Bool?
    public var ltx_sol_tau: Double?
    public var ltx_sol_dense_edge_blocks: Int?
    public var ltx_sol_dense_edge_steps: Int?
    public var ltx_stage2_text_rows: Int?
    public var streaming: NativeStreamingSelectorV2
}
public struct NativeParametersV2: Codable, Sendable {
    public var dynamic_text: Bool
    public var compile_gpu: Bool?
    public var noise_path: String?
}
public struct NativeRequestV2: Codable, Sendable {
    public var schema_version = 2
    public var model: String
    public var operation: String
    public var inputs: [NativeInputV2]
    public var outputs: [NativeOutputV2]
    public var sampling: NativeSamplingV2
    public var execution: NativeExecutionV2
    public var parameters: NativeParametersV2
    public var dump_tensors: String?
    public var loras: [NativeLoRA]?
    public var lora_strategy: String?

    public init(legacy request: NativeRequest, targetBytes: UInt64) {
        model = request.model
        operation = request.operation ?? (request.frames > 1 ? "video.generate" : "image.generate")
        inputs = [NativeInputV2(kind: "text", role: "prompt", path: nil,
                                text: request.prompt, strength: nil)]
        inputs += (request.inputs ?? []).map {
            NativeInputV2(kind: $0.kind, role: $0.role, path: $0.path,
                          text: nil, strength: $0.strength)
        }
        let outputKind = operation.hasPrefix("video.") ? "video" : "image"
        outputs = [NativeOutputV2(kind: outputKind, path: request.output,
                                  width: request.width, height: request.height,
                                  frames: request.frames, fps: request.fps ?? 24,
                                  audio: request.audio ?? false)]
        sampling = NativeSamplingV2(seed: request.seed, steps: request.steps)
        execution = NativeExecutionV2(
            policy: request.execution, profile: request.profile,
            ane_manifest: request.ane_manifest, encoder_ane_manifest: nil,
            allow_approximation: request.allow_approximation,
            quantized_cache: request.quantized_cache, warmup_iterations: nil,
            ltx_backend: request.ltx_backend,
            ltx_fast_av: request.ltx_fast_av,
            ltx_video_attention_batch: request.ltx_video_attention_batch,
            ltx_sol_stage1: request.ltx_sol_stage1,
            ltx_sol_stage2: request.ltx_sol_stage2,
            ltx_sol_tau: request.ltx_sol_tau,
            ltx_sol_dense_edge_blocks: request.ltx_sol_dense_edge_blocks,
            ltx_sol_dense_edge_steps: request.ltx_sol_dense_edge_steps,
            ltx_stage2_text_rows: request.ltx_stage2_text_rows,
            streaming: NativeStreamingSelectorV2(targetBytes: targetBytes))
        parameters = NativeParametersV2(
            dynamic_text: request.dynamic_text, compile_gpu: request.compile_gpu,
            noise_path: request.noise_path)
        dump_tensors = request.dump_tensors
        loras = request.loras
        lora_strategy = request.lora_strategy
    }
}
public struct NativeStreamingTargetOption: Codable, Sendable, Identifiable {
    public var id: UInt64 { target_request_memory_bytes }
    public let target_request_memory_bytes: UInt64
    public let status: String
    public let reason_code: String?
    public let preset_id: String?
    public let preset_revision: UInt32?
    public let calibrated_request_bytes: UInt64?
    public let memory_scope: String?
    public let release_channel: String?
}
public struct NativeStreamingDevice: Codable, Sendable {
    public let gpu: String
    public let physical_memory_bytes: UInt64
    public let device_class: String
}
public struct NativeStreamingOptions: Codable, Sendable {
    public let schema_version: Int
    public let catalog_revision: String
    public let query_status: String
    public let execution_container: String
    public let device: NativeStreamingDevice
    public let targets: [NativeStreamingTargetOption]
}
public struct NativeStreamingResolutionSelection: Codable, Sendable {
    public let preset_id: String
    public let preset_revision: UInt32
    public let record_digest: String
    public let release_channel: String
    public let target_request_memory_bytes: UInt64
    public let calibrated_request_bytes: UInt64
    public let memory_scope: String
    public let layout_digest: String
    public let component_policy_revision: String
    public let execution_container: String
}
public struct NativeStreamingResolutionIdentity: Codable, Sendable {
    public let source_digest: String
    public let runtime_digest: String
    public let device_digest: String
}
public struct NativeStreamingResolution: Codable, Sendable {
    public let schema_version: Int
    public let status: String
    public let request_digest: String
    public let resolution_digest: String
    public let catalog_revision: String
    public let requested_selector: NativeStreamingSelectorV2
    public let exact_selector: NativeStreamingSelectorV2
    public let selection: NativeStreamingResolutionSelection
    public let identity: NativeStreamingResolutionIdentity
}
public struct NativeEvent: Codable, Sendable {
    public let sequence: Int
    public let phase: String
    public let completed: Int
    public let total: Int
    public let elapsed_seconds: Double
}
public struct NativeAPIErrorEnvelope: Codable, Sendable {
    public let schema_version: Int
    public let code: String
    public let message: String
    public let retryable: Bool
    public let action: String?
}
public struct NativeFailure: Error, LocalizedError, Sendable {
    public let message: String
    public let code: String?
    public let retryable: Bool
    public let action: String?
    public init(message: String, code: String? = nil,
                retryable: Bool = false, action: String? = nil) {
        self.message = message
        self.code = code
        self.retryable = retryable
        self.action = action
    }
    public var errorDescription: String? { message }
}
private func nativeFailure(_ message: String) -> NativeFailure {
    if let data = message.data(using: .utf8),
       let envelope = try? JSONDecoder().decode(
           NativeAPIErrorEnvelope.self, from: data) {
        return NativeFailure(
            message: envelope.message, code: envelope.code,
            retryable: envelope.retryable, action: envelope.action)
    }
    return NativeFailure(message: message)
}
private func consume(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
    guard let pointer else { return "" }
    defer { tc_string_free(pointer) }
    return String(cString: pointer)
}
private final class EventBox: @unchecked Sendable {
    let receive: @Sendable (NativeEvent) -> Void
    init(_ receive: @escaping @Sendable (NativeEvent) -> Void) { self.receive = receive }
}
private let eventCallback: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { text, context in
    guard let text, let context else { return }
    let box = Unmanaged<EventBox>.fromOpaque(context).takeUnretainedValue()
    if let event = try? JSONDecoder().decode(NativeEvent.self, from: Data(String(cString: text).utf8)) { box.receive(event) }
}

/// Native embedded session. Generation runs on a serial queue. C strings never escape.
public final class NativeEngine: @unchecked Sendable {
    private let handle: OpaquePointer
    private let queue = DispatchQueue(label: "org.turbocider.native.inference", qos: .userInitiated)
    public init(modelURL: URL, modelID: String = "flux2-klein-4b") throws {
        var handle: OpaquePointer?
        var error: UnsafeMutablePointer<CChar>?
        let status = modelID.withCString { id in modelURL.path.withCString { tc_engine_create_model(id, $0, &handle, &error) } }
        let message = consume(error)
        guard status == 0, let handle else { throw NativeFailure(message: message) }
        self.handle = handle
    }
    deinit { tc_engine_free(handle) }
    public func cancel() { tc_engine_cancel(handle) }
    public static func open(modelURL: URL, modelID: String) async throws -> NativeEngine {
        try await Task.detached(priority: .userInitiated) {
            try NativeEngine(modelURL: modelURL, modelID: modelID)
        }.value
    }
    public func load(onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await resources(load: true, onEvent: onEvent)
    }
    public func unload() async throws -> Data {
        try await resources(load: false, onEvent: { _ in })
    }
    private func resources(load: Bool, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                let context = Unmanaged.passRetained(EventBox(onEvent)).toOpaque()
                defer { Unmanaged<EventBox>.fromOpaque(context).release() }
                var result: UnsafeMutablePointer<CChar>?
                var error: UnsafeMutablePointer<CChar>?
                let status = load ? tc_engine_load(handle, eventCallback, context, &result, &error)
                                  : tc_engine_unload(handle, &result, &error)
                let message = consume(error), output = consume(result)
                if status == 0 { continuation.resume(returning: Data(output.utf8)) }
                else if status == 2 { continuation.resume(throwing: CancellationError()) }
                else { continuation.resume(throwing: NativeFailure(message: message)) }
            }
        }
    }
    public func prepare(_ request: NativeRequest, warmup: Bool, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await preparation(payload: JSONEncoder().encode(request), warmup: warmup, cache: false, onEvent: onEvent)
    }
    public func cache(action: String, directory: URL, source: URL? = nil, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        var value = ["action": action, "cache": directory.path]
        if let source { value["source"] = source.path }
        return try await preparation(payload: JSONEncoder().encode(value), warmup: false, cache: true, onEvent: onEvent)
    }
    private func preparation(payload: Data, warmup: Bool, cache: Bool, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                let context = Unmanaged.passRetained(EventBox(onEvent)).toOpaque()
                defer { Unmanaged<EventBox>.fromOpaque(context).release() }
                var result: UnsafeMutablePointer<CChar>?, error: UnsafeMutablePointer<CChar>?
                let status = String(decoding: payload, as: UTF8.self).withCString {
                    cache ? tc_engine_cache(handle, $0, eventCallback, context, &result, &error)
                          : tc_engine_prepare(handle, $0, warmup ? 1 : 0, eventCallback, context, &result, &error)
                }
                let message = consume(error), output = consume(result)
                if status == 0 { continuation.resume(returning: Data(output.utf8)) }
                else if status == 2 { continuation.resume(throwing: CancellationError()) }
                else { continuation.resume(throwing: NativeFailure(message: message)) }
            }
        }
    }
    public static func cancelCoreMLResources() { tc_coreml_resources_cancel() }
    public static func coreMLResources(_ payload: Data, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await Task.detached(priority: .userInitiated) {
            let context = Unmanaged.passRetained(EventBox(onEvent)).toOpaque()
            defer { Unmanaged<EventBox>.fromOpaque(context).release() }
            var result: UnsafeMutablePointer<CChar>?, error: UnsafeMutablePointer<CChar>?
            let status = String(decoding: payload, as: UTF8.self).withCString { tc_coreml_resources_json($0, eventCallback, context, &result, &error) }
            let message = consume(error), output = consume(result)
            if status == 2 { throw CancellationError() }
            guard status == 0 else { throw NativeFailure(message: message) }
            return Data(output.utf8)
        }.value
    }
    public static func zImageTokenCount(modelPath: String, prompt: String) throws -> Int {
        var result: UnsafeMutablePointer<CChar>?, error: UnsafeMutablePointer<CChar>?
        let status = modelPath.withCString { path in
            prompt.withCString { tc_z_image_tokenize_json(path, $0, &result, &error) }
        }
        let message = consume(error), output = consume(result)
        guard status == 0 else { throw NativeFailure(message: message) }
        guard let value = try JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any],
              let count = value["valid"] as? Int else { throw NativeFailure(message: "文本 token 计数结果无效。") }
        return count
    }
    public static func system() -> String { consume(tc_system_json()) }
    public static func models() -> String { consume(tc_models_json()) }
    public static func plan(_ request: NativeRequest) throws -> Data {
        let data = try JSONEncoder().encode(request)
        var result: UnsafeMutablePointer<CChar>?
        var error: UnsafeMutablePointer<CChar>?
        let status = String(decoding: data, as: UTF8.self).withCString { tc_plan_json($0, &result, &error) }
        let message = consume(error), output = consume(result)
        guard status == 0 else { throw NativeFailure(message: message) }
        return Data(output.utf8)
    }
    public static func plan(_ request: NativeRequestV2) throws -> Data {
        let data = try JSONEncoder().encode(request)
        var result: UnsafeMutablePointer<CChar>?, error: UnsafeMutablePointer<CChar>?
        let status = String(decoding: data, as: UTF8.self).withCString {
            tc_plan_json($0, &result, &error)
        }
        let message = consume(error), output = consume(result)
        guard status == 0 else { throw NativeFailure(message: message) }
        return Data(output.utf8)
    }
    public static func streamingOptions(_ request: NativeRequestV2) throws -> NativeStreamingOptions {
        let data = try JSONEncoder().encode(request)
        var result: UnsafeMutablePointer<CChar>?, error: UnsafeMutablePointer<CChar>?
        let status = String(decoding: data, as: UTF8.self).withCString {
            tc_streaming_options_json($0, &result, &error)
        }
        let message = consume(error), output = consume(result)
        guard status == 0 else { throw NativeFailure(message: message) }
        return try JSONDecoder().decode(NativeStreamingOptions.self, from: Data(output.utf8))
    }
    public func resolveStreaming(_ request: NativeRequestV2) async throws -> NativeStreamingResolution {
        let payload = try JSONEncoder().encode(request)
        return try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                var result: UnsafeMutablePointer<CChar>?
                var error: UnsafeMutablePointer<CChar>?
                let status = String(decoding: payload, as: UTF8.self).withCString {
                    tc_engine_resolve_streaming_json(
                        handle, $0, &result, &error)
                }
                let message = consume(error), output = consume(result)
                guard status == 0 else {
                    continuation.resume(throwing: nativeFailure(message))
                    return
                }
                do {
                    continuation.resume(returning: try JSONDecoder().decode(
                        NativeStreamingResolution.self,
                        from: Data(output.utf8)))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    public func generate(_ request: NativeRequest, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await generate(payload: JSONEncoder().encode(request), onEvent: onEvent)
    }
    public func generate(_ request: NativeRequestV2, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await generate(payload: JSONEncoder().encode(request), onEvent: onEvent)
    }
    private func generate(payload data: Data, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                let box = EventBox(onEvent)
                let context = Unmanaged.passRetained(box).toOpaque()
                defer { Unmanaged<EventBox>.fromOpaque(context).release() }
                var result: UnsafeMutablePointer<CChar>?
                var error: UnsafeMutablePointer<CChar>?
                let status = String(decoding: data, as: UTF8.self).withCString {
                    tc_engine_generate(handle, $0, eventCallback, context, &result, &error)
                }
                let message = consume(error), output = consume(result)
                if status == 0 { continuation.resume(returning: Data(output.utf8)) }
                else if status == 2 { continuation.resume(throwing: CancellationError()) }
                else { continuation.resume(throwing: NativeFailure(message: message)) }
            }
        }
    }
}
