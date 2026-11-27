import Foundation
import Network

/// Bounded HTTP transport for the loopback API and the separate browser host.
@MainActor final class LocalHTTPGateway {
    typealias RPC = ([String: Any]) async throws -> [String: Any]
    typealias WebRoute = (Request) async throws -> Response
    struct Failure: LocalizedError {
        let status: Int
        let message: String
        var errorDescription: String? { message }
    }
    @MainActor private final class Connection {
        let network: NWConnection
        var buffer = Data()
        var head: Head?
        var upload: (url: URL, file: FileHandle, received: Int)?
        var timeout: Task<Void, Never>?
        var request: Task<Void, Never>?
        init(_ network: NWConnection) { self.network = network }
    }
    private struct Head {
        let request: Request
        let length: Int
        let streamedUpload: Bool
    }
    private let root: URL
    private let outputDirectory: URL
    private let rpc: RPC
    private let host: String
    private let webRoute: WebRoute?
    private let authorizeUpload: ((Request) -> Bool)?
    private let uploadDirectory: URL?
    private let onFailure: (String) -> Void
    private var listener: NWListener?
    private var ready: CheckedContinuation<Int, Error>?
    private var startupTimeout: Task<Void, Never>?
    private var connections: [UUID: Connection] = [:]
    private let queue = DispatchQueue(label: "org.turbocider.http")
    private(set) var port = 0
    private let maxUpload = 64 * 1024 * 1024

    init(root: URL, outputDirectory: URL? = nil, rpc: @escaping RPC, onFailure: @escaping (String) -> Void = { _ in }) {
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        self.outputDirectory = (outputDirectory ?? root.appendingPathComponent("outputs")).standardizedFileURL.resolvingSymlinksInPath()
        self.rpc = rpc; self.onFailure = onFailure
        host = "127.0.0.1"; webRoute = nil; authorizeUpload = nil; uploadDirectory = nil
    }

    init(webHost: String, uploadDirectory: URL? = nil, route: @escaping WebRoute, authorizeUpload: @escaping (Request) -> Bool,
         onFailure: @escaping (String) -> Void) {
        root = FileManager.default.temporaryDirectory
        outputDirectory = root
        rpc = { _ in throw Failure(status: 404, message: "Unknown endpoint") }
        host = webHost; webRoute = route; self.authorizeUpload = authorizeUpload; self.onFailure = onFailure
        self.uploadDirectory = uploadDirectory
    }
    func start(port: Int) async throws -> Int {
        guard listener == nil, (0...65535).contains(port) else { throw fail("Invalid HTTP port") }
        for directory in webRoute == nil ? [root.appendingPathComponent("inputs"), outputDirectory] : [] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        }
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: UInt16(port))!)
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
        if let upload = connection.upload {
            try? upload.file.close()
            try? FileManager.default.removeItem(at: upload.url)
        }
    }
    private func receive(_ id: UUID) {
        guard let connection = connections[id] else { return }
        connection.network.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, complete, error in
            Task { @MainActor in
                guard let self, self.connections[id] === connection else { return }
                if let data { connection.buffer.append(data) }
                do {
                    if let request = try self.parse(connection, id: id) {
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
                                await self.send(id, response: response, range: request.headers["range"])
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
        var body: Data
        var bodyFile: URL? = nil
    }
    struct Response {
        let status: Int
        let data: Data
        var mime = "application/json"
        var file: URL? = nil
        var headers: [String: String] = [:]
        var ranges = false
        var sandboxPolicy: String? = nil
    }
    private func fail(_ message: String, _ status: Int = 400) -> Failure { Failure(status: status, message: message) }
    private func parse(_ connection: Connection, id: UUID) throws -> Request? {
        if connection.head == nil {
            guard let (head, headerBytes) = try parseHead(connection.buffer) else { return nil }
            connection.head = head
            // Data slices retain their original indices. Body readers use offsets from zero.
            connection.buffer = Data(connection.buffer.dropFirst(headerBytes))
            if head.streamedUpload {
                guard let uploadDirectory, connections.values.lazy.filter({ $0.upload != nil }).count < 4 else {
                    throw fail("同时上传的文件过多，请稍后重试。", 429)
                }
                try FileManager.default.createDirectory(at: uploadDirectory, withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                let url = uploadDirectory.appendingPathComponent(UUID().uuidString)
                guard FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
                    throw fail("无法建立上传文件。", 503)
                }
                do { connection.upload = (url, try FileHandle(forWritingTo: url), 0) }
                catch { try? FileManager.default.removeItem(at: url); throw error }
                connection.timeout?.cancel()
                connection.timeout = Task { [weak self] in
                    do { try await Task.sleep(for: .seconds(180)) } catch { return }
                    self?.close(id)
                }
            }
        }
        guard let head = connection.head else { return nil }
        var request = head.request
        if var upload = connection.upload {
            upload.received += connection.buffer.count
            guard upload.received <= head.length else { throw fail("HTTP pipelining is unsupported") }
            try upload.file.write(contentsOf: connection.buffer)
            connection.buffer = Data()
            connection.upload = upload
            guard upload.received == head.length else { return nil }
            try upload.file.close()
            request.bodyFile = upload.url
        } else {
            guard connection.buffer.count >= head.length else { return nil }
            guard connection.buffer.count == head.length else { throw fail("HTTP pipelining is unsupported") }
            request.body = connection.buffer
        }
        return request
    }
    private func parseHead(_ data: Data) throws -> (Head, Int)? {
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
        let method = String(line[0]), path = String(line[1])
        guard method == "GET" || method == "POST" else { throw fail("Method not allowed", 405) }
        guard headers["transfer-encoding"] == nil else { throw fail("Chunked requests are unsupported") }
        let requestHost = headers["host"]?.lowercased() ?? ""
        if webRoute != nil {
            let authority = port == 80 ? host : "\(host):\(port)"
            guard requestHost == authority || (port == 80 && requestHost == "\(host):80") else {
                throw fail("Invalid web Host", 403)
            }
            if method == "POST" {
                guard headers["origin"] == "http://\(authority)",
                      headers["sec-fetch-site"].map({ $0 == "same-origin" }) ?? true else {
                    throw fail("Invalid browser origin", 403)
                }
            }
        } else {
            guard headers["origin"] == nil else { throw fail("Browser-origin requests are disabled; use backend clients", 403) }
            guard ["127.0.0.1", "localhost", "127.0.0.1:\(port)", "localhost:\(port)"].contains(requestHost) else {
                throw fail("Only loopback Host headers are accepted", 403)
            }
        }
        let rawLength = headers["content-length"] ?? (method == "GET" ? "0" : "")
        guard !rawLength.isEmpty, rawLength.allSatisfy({ $0.isASCII && $0.isNumber }), let length = Int(rawLength) else { throw fail("Missing or invalid Content-Length") }
        let streamedUpload = webRoute != nil && ["/api/studio/upload", "/api/studio/host/upload"].contains(path)
        let browserUpload = webRoute != nil && (path == "/api/media" || streamedUpload)
        let apiUpload = webRoute == nil && path == "/v1/assets"
        let limit = streamedUpload ? 256 * 1024 * 1024 : browserUpload ? 16 * 1024 * 1024 : apiUpload ? maxUpload : 1024 * 1024
        guard length <= limit else { throw fail("Request too large", 413) }
        guard method != "GET" || length == 0 else { throw fail("GET body is unsupported") }
        if method == "POST" {
            let expected = browserUpload ? "application/octet-stream" : apiUpload ? "image/png" : "application/json"
            guard headers["content-type"]?.split(separator: ";").first?.trimmingCharacters(in: .whitespaces).lowercased() == expected else { throw fail("Content-Type must be \(expected)") }
            if browserUpload, authorizeUpload?(Request(method: method, path: path, headers: headers, body: Data())) != true {
                if path == "/api/studio/host/upload" { throw fail("这台设备没有 Mac 管理权限。", 403) }
                throw fail("请先连接网页工作台。", 401)
            }
        }
        let request = Request(method: method, path: path, headers: headers, body: Data())
        return (Head(request: request, length: length, streamedUpload: streamedUpload), end.upperBound)
    }
    private func json(_ value: Any, status: Int = 200) throws -> Response {
        Response(status: status, data: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]))
    }
    static func object(_ data: Data) throws -> [String: Any] {
        // Check duplicate object keys before Foundation collapses them. JSON
        // decoding below remains authoritative for numbers, escapes and syntax.
        do {
            try UniqueJSONKeys.validate(data)
            guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw Failure(status: 400, message: "Expected a JSON object")
            }
            return value
        } catch { throw Failure(status: 400, message: "Invalid JSON object: \(error.localizedDescription)") }
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
        let video = (request["operation"] as? String)?.hasPrefix("video.") ?? (model?["output"] as? String == "video")
        let ext = video ? "mp4" : "png"
        request["output"] = outputDirectory.appendingPathComponent("\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()).\(ext)").path
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
        guard url.deletingLastPathComponent().path == outputDirectory.path,
              url.lastPathComponent.range(of: "^[0-9a-f]{32}\\.(png|mp4)$", options: .regularExpression) != nil else { throw fail("Unknown gateway job") }
        return value
    }
    private func route(_ request: Request) async throws -> Response {
        if let webRoute { return try await webRoute(request) }
        switch (request.method, request.path) {
        case ("GET", "/v1/health"): return try json(await rpc(["action": "service_status"]))
        case ("GET", "/v1/catalog"): return try json(await catalog())
        case ("GET", "/v1/jobs"): return try json(await rpc(["action": "jobs", "limit": 100]))
        case ("POST", "/v1/plan"), ("POST", "/v1/jobs"):
            let (path, value) = try await prepare(Self.object(request.body))
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
                guard try Self.object(request.body).isEmpty else { throw fail("Cancel body must be {}") }
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
    private func send(_ id: UUID, response: Response, range: String? = nil) async {
        guard let connection = connections[id] else { return }
        do {
            let file = try response.file.map { try FileHandle(forReadingFrom: $0) }
            defer { try? file?.close() }
            let size = try file?.seekToEnd() ?? UInt64(response.data.count)
            var status = response.status, offset: UInt64 = 0, count = size
            var headers = response.headers
            if response.ranges, file != nil {
                headers["Accept-Ranges"] = "bytes"
                if let range, range.hasPrefix("bytes="), !range.contains(",") {
                    if let selected = Self.byteRange(String(range.dropFirst(6)), size: size) {
                        status = 206; offset = selected.lowerBound; count = selected.upperBound - offset + 1
                        headers["Content-Range"] = "bytes \(offset)-\(selected.upperBound)/\(size)"
                    } else {
                        status = 416; count = 0
                        headers["Content-Range"] = "bytes */\(size)"
                    }
                }
            }
            try file?.seek(toOffset: offset)
            var header = "HTTP/1.1 \(status) \(HTTPURLResponse.localizedString(forStatusCode: status))\r\nContent-Type: \(response.mime)\r\nContent-Length: \(count)\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
            if webRoute != nil {
                header += "Referrer-Policy: no-referrer\r\n"
                if let policy = response.sandboxPolicy {
                    header += "Content-Security-Policy: \(policy)\r\n"
                } else {
                    header += "X-Frame-Options: DENY\r\nCross-Origin-Resource-Policy: same-origin\r\nContent-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; media-src 'self' blob:; frame-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'\r\n"
                }
            }
            for (key, value) in headers { header += "\(key): \(value)\r\n" }
            header += "\r\n"
            try await write(connection.network, Data(header.utf8))
            if let file {
                var remaining = count
                while remaining > 0, let data = try file.read(upToCount: Int(min(remaining, 1024 * 1024))), !data.isEmpty {
                    try Task.checkCancellation(); try await write(connection.network, data)
                    remaining -= UInt64(data.count)
                }
            } else { try await write(connection.network, response.data) }
        } catch { /* Peer disconnects do not cancel native jobs. */ }
        close(id)
    }
    private static func byteRange(_ value: String, size: UInt64) -> ClosedRange<UInt64>? {
        let parts = value.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        guard size > 0, parts.count == 2 else { return nil }
        if parts[0].isEmpty {
            guard let count = UInt64(parts[1]), count > 0 else { return nil }
            return (size - min(size, count))...(size - 1)
        }
        guard let start = UInt64(parts[0]), start < size,
              let end = parts[1].isEmpty ? size - 1 : UInt64(parts[1]), start <= end else { return nil }
        return start...min(end, size - 1)
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
