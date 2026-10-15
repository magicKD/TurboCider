import Foundation
import Network

/// App-owned loopback HTTP transport. Uses the same schema-1 contract as the
/// standalone adapter, and delegates all execution decisions to native plan.
@MainActor final class LocalHTTPGateway {
    typealias RPC = ([String: Any]) async throws -> [String: Any]
    struct Failure: LocalizedError {
        let status: Int
        let message: String
        var errorDescription: String? { message }
    }
    @MainActor private final class Connection {
        let network: NWConnection
        var buffer = Data()
        var timeout: Task<Void, Never>?
        var request: Task<Void, Never>?
        init(_ network: NWConnection) { self.network = network }
    }
    private let root: URL
    private let rpc: RPC
    private let onFailure: (String) -> Void
    private var listener: NWListener?
    private var ready: CheckedContinuation<Int, Error>?
    private var startupTimeout: Task<Void, Never>?
    private var connections: [UUID: Connection] = [:]
    private let queue = DispatchQueue(label: "org.turbocider.http")
    private(set) var port = 0
    private let maxUpload = 64 * 1024 * 1024

    init(root: URL, rpc: @escaping RPC, onFailure: @escaping (String) -> Void = { _ in }) {
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        self.rpc = rpc; self.onFailure = onFailure
    }
    func start(port: Int) async throws -> Int {
        guard listener == nil, (0...65535).contains(port) else { throw fail("Invalid HTTP port") }
        for name in ["inputs", "outputs"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        }
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: UInt16(port))!)
        let server = try NWListener(using: parameters)
        listener = server
        server.newConnectionHandler = { [weak self] connection in
            Task { @MainActor in
                guard let self, self.listener === server, self.port != 0 else { connection.cancel(); return }
                self.accept(connection)
            }
        }
        return try await withCheckedThrowingContinuation { continuation in
            ready = continuation
            server.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    guard let self, self.listener === server else { return }
                    switch state {
                    case .ready:
                        guard let bound = server.port else { self.failed("HTTP gateway has no listening port"); return }
                        self.port = Int(bound.rawValue)
                        let pending = self.ready; self.ready = nil
                        self.startupTimeout?.cancel(); self.startupTimeout = nil
                        pending?.resume(returning: self.port)
                    case .failed(let error): self.failed("HTTP 网关启动或监听失败（端口 \(port)）：\(error.localizedDescription)。请检查端口是否被占用。")
                    default: break
                    }
                }
            }
            startupTimeout = Task { [weak self] in
                do { try await Task.sleep(for: .seconds(5)) } catch { return }
                self?.failed("HTTP 网关启动超时。")
            }
            server.start(queue: queue)
        }
    }
    private func failed(_ message: String) {
        let wasReady = ready == nil && port != 0
        let pending = ready; ready = nil
        stop()
        pending?.resume(throwing: fail(message, 503))
        if wasReady { onFailure(message) }
    }
    func stop() {
        listener?.stateUpdateHandler = nil; listener?.newConnectionHandler = nil
        listener?.cancel(); listener = nil; port = 0
        startupTimeout?.cancel(); startupTimeout = nil
        let pending = ready; ready = nil; pending?.resume(throwing: CancellationError())
        for id in Array(connections.keys) { close(id) }
    }
    private func accept(_ network: NWConnection) {
        guard connections.count < 64 else { network.cancel(); return }
        let id = UUID(), connection = Connection(network)
        connections[id] = connection
        connection.timeout = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(30)) } catch { return }
            self?.close(id)
        }
        network.stateUpdateHandler = { [weak self] state in
            if case .failed = state { Task { @MainActor in self?.close(id) } }
        }
        network.start(queue: queue)
        receive(id)
    }
    private func close(_ id: UUID) {
        guard let connection = connections.removeValue(forKey: id) else { return }
        connection.timeout?.cancel(); connection.request?.cancel(); connection.network.cancel()
    }
    private func receive(_ id: UUID) {
        guard let connection = connections[id] else { return }
        connection.network.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, complete, error in
            Task { @MainActor in
                guard let self, self.connections[id] === connection else { return }
                if let data { connection.buffer.append(data) }
                do {
                    if let request = try self.parse(connection.buffer) {
                        connection.timeout?.cancel()
                        connection.timeout = Task { [weak self] in
                            do { try await Task.sleep(for: .seconds(180)) } catch { return }
                            self?.close(id)
                        }
                        connection.buffer = Data()
                        connection.request = Task { [weak self] in
                            guard let self else { return }
                            do {
                                let response = try await self.route(request)
                                try Task.checkCancellation()
                                await self.send(id, response: response)
                            } catch is CancellationError { self.close(id) }
                            catch { await self.sendError(id, error) }
                        }
                    } else if complete || error != nil { self.close(id) }
                    else { self.receive(id) }
                } catch { await self.sendError(id, error) }
            }
        }
    }
    struct Request {
        let method: String
        let path: String
        let headers: [String: String]
        let body: Data
    }
    struct Response {
        let status: Int
        let data: Data
        var mime = "application/json"
        var file: URL? = nil
    }
    private func fail(_ message: String, _ status: Int = 400) -> Failure { Failure(status: status, message: message) }
    private func parse(_ data: Data) throws -> Request? {
        guard let end = data.range(of: Data("\r\n\r\n".utf8)) else {
            if data.count > 16384 { throw fail("HTTP headers too large", 413) }
            return nil
        }
        guard end.upperBound <= 16384, let text = String(data: data[..<end.lowerBound], encoding: .utf8) else { throw fail("Invalid HTTP headers") }
        let lines = text.components(separatedBy: "\r\n")
        let line = lines[0].split(separator: " ")
        guard line.count == 3, line[2] == "HTTP/1.1" || line[2] == "HTTP/1.0" else { throw fail("Invalid request line") }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":"), !line.hasPrefix(" "), !line.hasPrefix("\t") else { throw fail("Invalid header") }
            let name = String(line[..<colon]).lowercased()
            guard !name.isEmpty, !name.contains(where: { $0.isWhitespace }), headers[name] == nil else { throw fail("Duplicate or invalid header") }
            headers[name] = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
        }
        guard headers["origin"] == nil else { throw fail("Browser-origin requests are disabled; use backend clients", 403) }
        let host = headers["host"]?.lowercased() ?? ""
        guard host == "127.0.0.1" || host == "localhost" || host == "127.0.0.1:\(port)" || host == "localhost:\(port)" else {
            throw fail("Only loopback Host headers are accepted", 403)
        }
        guard headers["transfer-encoding"] == nil else { throw fail("Chunked requests are unsupported") }
        let method = String(line[0]), path = String(line[1])
        guard method == "GET" || method == "POST" else { throw fail("Method not allowed", 405) }
        let rawLength = headers["content-length"] ?? (method == "GET" ? "0" : "")
        guard !rawLength.isEmpty, rawLength.allSatisfy({ $0.isASCII && $0.isNumber }), let length = Int(rawLength) else { throw fail("Missing or invalid Content-Length") }
        guard length <= (path == "/v1/assets" ? maxUpload : 1048576) else { throw fail("Request too large", 413) }
        guard method != "GET" || length == 0 else { throw fail("GET body is unsupported") }
        if method == "POST" {
            let expected = path == "/v1/assets" ? "image/png" : "application/json"
            guard headers["content-type"]?.split(separator: ";").first?.trimmingCharacters(in: .whitespaces).lowercased() == expected else { throw fail("Content-Type must be \(expected)") }
        }
        guard data.count >= end.upperBound + length else { return nil }
        guard data.count == end.upperBound + length else { throw fail("HTTP pipelining is unsupported") }
        return Request(method: method, path: path, headers: headers, body: Data(data[end.upperBound...]))
    }
    private func json(_ value: Any, status: Int = 200) throws -> Response {
        Response(status: status, data: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]))
    }
    private func object(_ data: Data) throws -> [String: Any] {
        // Check duplicate object keys before Foundation collapses them. JSON
        // decoding below remains authoritative for numbers, escapes and syntax.
        do {
            try UniqueJSONKeys.validate(data)
            guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw fail("Expected a JSON object") }
            return value
        } catch { throw fail("Invalid JSON object: \(error.localizedDescription)") }
    }
    private func catalog() async throws -> [String: Any] {
        let models = try await rpc(["action": "models"])["models"] as? [[String: Any]] ?? []
        let inventory = try await rpc(["action": "installations"])["index"] as? [String: Any] ?? [:]
        let executable = Set(models.filter { $0["executor"] as? Bool == true }.compactMap { $0["id"] as? String })
        let installations = (inventory["installations"] as? [[String: Any]] ?? []).filter { executable.contains($0["modelID"] as? String ?? "") }
        return ["protocol": "turbocider.http", "version": 1, "models": models, "installations": installations,
                "loras": inventory["loras"] ?? [], "workflows": try await rpc(["action": "workflows"]), "files_verified": false]
    }
    private func identifier(_ value: Any?) throws -> String {
        guard let id = value as? String, id.range(of: "^[A-Za-z0-9_-]{1,80}$", options: .regularExpression) != nil else { throw fail("Invalid ID") }
        return id
    }
    private func prepare(_ payload: [String: Any]) async throws -> (String, [String: Any]) {
        guard Set(payload.keys) == ["installation_id", "request"], var request = payload["request"] as? [String: Any] else { throw fail("Expected installation_id and request") }
        let inventory = try await catalog()
        guard let installation = (inventory["installations"] as? [[String: Any]])?.first(where: { $0["id"] as? String == payload["installation_id"] as? String }),
              let path = installation["path"] as? String, let modelID = installation["modelID"] as? String else { throw fail("Unknown installation; refresh catalog") }
        guard FileManager.default.fileExists(atPath: path) else { throw fail("Registered model path no longer exists") }
        if let schema = request["schema_version"] {
            guard let number = schema as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue == 1 else { throw fail("HTTP generation uses native schema 1") }
        }
        guard !["output", "outputs", "model_path"].contains(where: { request[$0] != nil }) else { throw fail("Output paths are allocated by the gateway") }
        guard request["model"] == nil || request["model"] as? String == modelID else { throw fail("Request model does not match installation") }
        request["model"] = modelID
        let model = (inventory["models"] as? [[String: Any]])?.first { $0["id"] as? String == modelID }
        let ext = model?["output"] as? String == "video" ? "mp4" : "png"
        request["output"] = root.appendingPathComponent("outputs/\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()).\(ext)").path
        if let values = request["inputs"] {
            guard var inputs = values as? [[String: Any]] else { throw fail("inputs must be a list of objects") }
            for index in inputs.indices where inputs[index]["asset_id"] != nil {
                let id = try identifier(inputs[index].removeValue(forKey: "asset_id"))
                let input = root.appendingPathComponent("inputs/\(id).png")
                guard inputs[index]["path"] == nil, FileManager.default.fileExists(atPath: input.path) else { throw fail("Unknown uploaded image or conflicting path") }
                inputs[index]["path"] = input.path
            }
            request["inputs"] = inputs
        }
        if let values = request["loras"] {
            guard var loras = values as? [[String: Any]] else { throw fail("loras must be a list of objects") }
            for index in loras.indices where loras[index]["id"] != nil {
                let id = try identifier(loras[index].removeValue(forKey: "id"))
                guard let adapter = (inventory["loras"] as? [[String: Any]])?.first(where: { $0["id"] as? String == id }),
                      adapter["modelID"] as? String == modelID, loras[index]["path"] == nil,
                      let path = adapter["path"] as? String, FileManager.default.fileExists(atPath: path) else { throw fail("Unknown LoRA, stale path or different model family") }
                loras[index]["path"] = path
            }
            request["loras"] = loras
        }
        return (path, request)
    }
    private func job(_ id: String) async throws -> [String: Any] {
        let value = try await rpc(["action": "status", "id": id])
        guard let request = value["request"] as? [String: Any], let path = request["output"] as? String else { throw fail("Unknown gateway job") }
        let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        guard url.deletingLastPathComponent().path == root.appendingPathComponent("outputs").path,
              url.lastPathComponent.range(of: "^[0-9a-f]{32}\\.(png|mp4)$", options: .regularExpression) != nil else { throw fail("Unknown gateway job") }
        return value
    }
    private func route(_ request: Request) async throws -> Response {
        switch (request.method, request.path) {
        case ("GET", "/v1/health"): return try json(await rpc(["action": "service_status"]))
        case ("GET", "/v1/catalog"): return try json(await catalog())
        case ("GET", "/v1/jobs"): return try json(await rpc(["action": "jobs", "limit": 100]))
        case ("POST", "/v1/plan"), ("POST", "/v1/jobs"):
            let (path, value) = try await prepare(object(request.body))
            if request.path == "/v1/plan" { return try json(await rpc(["action": "plan", "request": value])) }
            return try json(await rpc(["action": "submit", "model_path": path, "request": value]), status: 202)
        case ("POST", "/v1/assets"):
            let data = request.body
            guard data.count >= 33, data.prefix(8) == Data([137, 80, 78, 71, 13, 10, 26, 10]), data[12..<16] == Data("IHDR".utf8) else { throw fail("Upload must be a PNG image") }
            func integer(_ offset: Int) -> UInt64 { data[offset..<offset+4].reduce(0) { ($0 << 8) | UInt64($1) } }
            let width = integer(16), height = integer(20)
            guard width > 0, height > 0, width <= 80000000, height <= 80000000, width * height <= 80000000 else { throw fail("Image exceeds 80 million pixels or has zero dimension") }
            let id = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
            try data.write(to: root.appendingPathComponent("inputs/\(id).png"), options: .withoutOverwriting)
            return try json(["id": id], status: 201)
        default:
            let parts = request.path.split(separator: "/", omittingEmptySubsequences: false)
            guard parts.count == 4 || parts.count == 5, parts[0].isEmpty, parts[1] == "v1", parts[2] == "jobs" else { throw fail("Unknown endpoint", 404) }
            let id = try identifier(String(parts[3]))
            let suffix = parts.count == 5 ? String(parts[4]) : ""
            if request.method == "GET", suffix.isEmpty { return try json(await job(id)) }
            if request.method == "POST", suffix == "cancel" {
                guard try object(request.body).isEmpty else { throw fail("Cancel body must be {}") }
                _ = try await job(id)
                return try json(await rpc(["action": "cancel", "id": id]))
            }
            if request.method == "GET", suffix == "artifact" {
                let value = try await job(id)
                guard value["state"] as? String == "succeeded" else { throw fail("Artifact is available only after job success") }
                let path = (value["request"] as! [String: Any])["output"] as! String
                guard (value["result"] as? [String: Any])?["output"] as? String == path else { throw fail("Result artifact does not match submitted output") }
                let file = URL(fileURLWithPath: path)
                guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                      FileManager.default.isReadableFile(atPath: file.path) else { throw fail("Result artifact is missing or unreadable", 410) }
                return Response(status: 200, data: Data(), mime: file.pathExtension == "png" ? "image/png" : "video/mp4", file: file)
            }
            throw fail("Unknown endpoint or method", 404)
        }
    }
    private func sendError(_ id: UUID, _ error: Error) async {
        let failure = error as? Failure
        // Once an RPC fails, a submit may already have reached the native
        // queue. Never retry mutations; expose ambiguity to all HTTP clients.
        let value: [String: Any] = ["error": error.localizedDescription, "submission_may_have_succeeded": failure == nil]
        if let response = try? json(value, status: failure?.status ?? 503) { await send(id, response: response) }
        else { close(id) }
    }
    private func send(_ id: UUID, response: Response) async {
        guard let connection = connections[id] else { return }
        do {
            let file = try response.file.map { try FileHandle(forReadingFrom: $0) }
            defer { try? file?.close() }
            let size = try file?.seekToEnd() ?? UInt64(response.data.count)
            try file?.seek(toOffset: 0)
            let header = "HTTP/1.1 \(response.status) \(HTTPURLResponse.localizedString(forStatusCode: response.status))\r\nContent-Type: \(response.mime)\r\nContent-Length: \(size)\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n"
            try await write(connection.network, Data(header.utf8))
            if let file {
                while let data = try file.read(upToCount: 1024 * 1024), !data.isEmpty {
                    try Task.checkCancellation(); try await write(connection.network, data)
                }
            } else { try await write(connection.network, response.data) }
        } catch { /* Peer disconnects do not cancel native jobs. */ }
        close(id)
    }
    private func write(_ network: NWConnection, _ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            network.send(content: data, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            })
        }
    }
}

/// A small preflight scanner, not a JSON decoder: only object key uniqueness
/// and nesting limits are checked here. Foundation validates the complete JSON.
private enum UniqueJSONKeys {
    static func validate(_ data: Data) throws {
        let bytes = [UInt8](data)
        var index = 0
        func skip() { while index < bytes.count, [9, 10, 13, 32].contains(bytes[index]) { index += 1 } }
        func invalid() -> LocalHTTPGateway.Failure { .init(status: 400, message: "Invalid JSON or excessive nesting") }
        func string() throws -> String {
            let start = index; index += 1
            while index < bytes.count {
                let byte = bytes[index]; index += 1
                if byte == 92 { index += 1 }
                else if byte == 34 {
                    return try JSONSerialization.jsonObject(with: Data(bytes[start..<index]), options: .fragmentsAllowed) as! String
                }
            }
            throw invalid()
        }
        func value(_ depth: Int) throws {
            skip(); guard depth < 128, index < bytes.count else { throw invalid() }
            if bytes[index] == 34 { _ = try string(); return }
            if bytes[index] == 123 || bytes[index] == 91 {
                let object = bytes[index] == 123, end: UInt8 = object ? 125 : 93
                index += 1; skip()
                if index < bytes.count, bytes[index] == end { index += 1; return }
                var keys = Set<String>()
                while index < bytes.count {
                    if object {
                        skip(); guard index < bytes.count, bytes[index] == 34 else { throw invalid() }
                        let key = try string()
                        guard keys.insert(key).inserted else { throw LocalHTTPGateway.Failure(status: 400, message: "Duplicate JSON field: \(key)") }
                        skip(); guard index < bytes.count, bytes[index] == 58 else { throw invalid() }; index += 1
                    }
                    try value(depth + 1); skip()
                    guard index < bytes.count else { throw invalid() }
                    if bytes[index] == end { index += 1; return }
                    guard bytes[index] == 44 else { throw invalid() }; index += 1
                }
                throw invalid()
            }
            let start = index
            while index < bytes.count, ![9, 10, 13, 32, 44, 93, 125].contains(bytes[index]) { index += 1 }
            guard index > start else { throw invalid() }
        }
        try value(0); skip(); guard index == bytes.count else { throw invalid() }
    }
}
