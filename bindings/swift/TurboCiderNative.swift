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
    public var compile_gpu: Bool?
    public var ane_manifest: String?
    public var allow_approximation: Bool?
    public var dump_tensors: String?
    public var operation: String?
    public var inputs: [NativeInput]?
    public var profile: String?
    public var residency: String?
    public var fps: Int?
    public var audio: Bool?
    public var loras: [NativeLoRA]?
    public init(prompt: String, output: String) { self.prompt = prompt; self.output = output }
}
public struct NativeEvent: Codable, Sendable {
    public let sequence: Int
    public let phase: String
    public let completed: Int
    public let total: Int
    public let elapsed_seconds: Double
}
public struct NativeFailure: Error, LocalizedError, Sendable {
    public let message: String
    public var errorDescription: String? { message }
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
    public func generate(_ request: NativeRequest, onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        let data = try JSONEncoder().encode(request)
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
