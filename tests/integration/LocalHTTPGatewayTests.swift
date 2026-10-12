import Foundation
import Darwin

@main struct LocalHTTPGatewayTests {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-http-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let adapter = root.appendingPathComponent("adapter.safetensors")
        try Data().write(to: adapter)
        var records: [String: [String: Any]] = [:]
        var submits = 0
        var lastRequest: [String: Any] = [:]
        let rpc: LocalHTTPGateway.RPC = { request in
            switch request["action"] as? String {
            case "service_status": return ["pid": 123, "active_job": ""]
            case "models": return ["models": [["id": "flux", "executor": true, "output": "image"], ["id": "absent", "executor": false]]]
            case "installations": return ["index": ["installations": [
                ["id": "model1", "modelID": "flux", "name": "Flux", "path": root.path],
                ["id": "unsupported", "modelID": "absent", "path": root.path],
                ["id": "stale", "modelID": "flux", "path": root.appendingPathComponent("absent").path]],
                "loras": [["id": "adapter", "modelID": "flux", "path": adapter.path],
                          ["id": "wrong", "modelID": "z-image", "path": adapter.path]]]]
            case "workflows": return [:]
            case "plan": lastRequest = request["request"] as! [String: Any]; return ["executable": true, "request": lastRequest]
            case "submit":
                submits += 1
                let id = "job-\(submits)", value = request["request"] as! [String: Any]
                records[id] = ["id": id, "state": "queued", "request": value]
                return ["id": id]
            case "status":
                guard let record = records[request["id"] as! String] else { throw LocalHTTPGateway.Failure(status: 400, message: "Unknown job") }
                return record
            case "cancel":
                let id = request["id"] as! String
                records[id]?["state"] = "cancelled"
                return records[id]!
            case "jobs": return ["jobs": Array(records.values)]
            default: throw LocalHTTPGateway.Failure(status: 400, message: "Unknown action")
            }
        }
        let gateway = LocalHTTPGateway(root: root.appendingPathComponent("http"), rpc: rpc)
        defer { gateway.stop() }
        let port = try await gateway.start(port: 0)
        let base = "http://127.0.0.1:\(port)"
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw NSError(domain: message, code: 1) }
        }
        func exchange(_ path: String, method: String = "GET", data: Data? = nil,
                      headers: [String: String] = [:], status: Int = 200) async throws -> Data {
            var request = URLRequest(url: URL(string: base + path)!)
            request.timeoutInterval = 5; request.httpMethod = method; request.httpBody = data
            if data != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
            for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
            let (body, response) = try await URLSession.shared.data(for: request)
            try check((response as? HTTPURLResponse)?.statusCode == status, "\(path): wanted \(status), got \((response as? HTTPURLResponse)?.statusCode ?? 0): \(String(decoding: body, as: UTF8.self))")
            return body
        }
        func json(_ path: String, method: String = "GET", value: [String: Any]? = nil, status: Int = 200) async throws -> [String: Any] {
            let data = try value.map { try JSONSerialization.data(withJSONObject: $0) }
            return try JSONSerialization.jsonObject(with: await exchange(path, method: method, data: data, status: status)) as! [String: Any]
        }
        func raw(_ pieces: [String]) async throws -> String {
            try await Task.detached {
                let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
                guard descriptor >= 0 else { throw NSError(domain: "socket", code: Int(errno)) }
                defer { Darwin.close(descriptor) }
                var address = sockaddr_in()
                address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
                address.sin_family = sa_family_t(AF_INET); address.sin_port = UInt16(port).bigEndian
                address.sin_addr.s_addr = inet_addr("127.0.0.1")
                var timeout = timeval(tv_sec: 5, tv_usec: 0)
                setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
                var noSignal: Int32 = 1
                setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &noSignal, socklen_t(MemoryLayout<Int32>.size))
                let connected = withUnsafePointer(to: &address) { pointer in
                    pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
                }
                guard connected == 0 else { throw NSError(domain: "connect", code: Int(errno)) }
                for piece in pieces {
                    let bytes = Array(piece.utf8)
                    try bytes.withUnsafeBytes { buffer in
                        var offset = 0
                        while offset < buffer.count {
                            let count = Darwin.send(descriptor, buffer.baseAddress!.advanced(by: offset), buffer.count - offset, 0)
                            guard count > 0 else { throw NSError(domain: "send", code: Int(errno)) }
                            offset += count
                        }
                    }
                    usleep(10000)
                }
                Darwin.shutdown(descriptor, SHUT_WR)
                var data = Data(), buffer = [UInt8](repeating: 0, count: 4096)
                while true {
                    let count = Darwin.recv(descriptor, &buffer, buffer.count, 0)
                    if count == 0 { break }
                    guard count > 0 else { throw NSError(domain: "recv", code: Int(errno)) }
                    data.append(contentsOf: buffer.prefix(count))
                }
                return String(decoding: data, as: UTF8.self)
            }.value
        }
        let header = "POST /v1/plan HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\nContent-Type: application/json\r\n"
        let body = "{\"installation_id\":\"model1\",\"request\":{}}"
        let fragmented = try await raw([header, "Content-Length: \(body.utf8.count)\r\n\r\n", body])
        try check(fragmented.hasPrefix("HTTP/1.1 200"), "Fragmented body did not parse")
        for invalid in [header + "Content-Length: 0\r\nContent-Length: 0\r\n\r\n", header + "Transfer-Encoding: chunked\r\n\r\n"] {
            let response = try await raw([invalid])
            try check(response.hasPrefix("HTTP/1.1 400"), "Ambiguous framing accepted")
        }
        let oversized = try await raw([header + "Content-Length: 1048577\r\n\r\n"])
        try check(oversized.hasPrefix("HTTP/1.1 413"), "Oversized JSON body admitted")
        let truncated = try await raw([header + "Content-Length: 100\r\n\r\n{}"])
        try check(truncated.isEmpty, "Truncated request was processed")
        let health = try await json("/v1/health")
        try check(health["pid"] as? Int == 123, "Health differs from native service")
        let catalog = try await json("/v1/catalog")
        try check((catalog["installations"] as? [[String: Any]])?.count == 2, "Non-executable registration was advertised")
        _ = try await exchange("/v1/catalog", headers: ["Origin": "https://example.org"], status: 403)
        _ = try await exchange("/v1/catalog", headers: ["Host": "evil.example"], status: 403)
        _ = try await exchange("/unknown", status: 404)
        _ = try await exchange("/v1/health", method: "DELETE", status: 405)
        _ = try await exchange("/v1/jobs", method: "POST", data: Data("{}".utf8), headers: ["Content-Type": "text/plain"], status: 400)
        for text in ["{", "[]", "{\"a\":1,\"a\":2}", "{\"request\":{\"a\":1,\"\\u0061\":2}}", "{\"v\":NaN}"] {
            _ = try await exchange("/v1/plan", method: "POST", data: Data(text.utf8), status: 400)
        }
        var png = Data([137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,0,0,0,0])
        let upload = try await exchange("/v1/assets", method: "POST", data: png, headers: ["Content-Type": "image/png"], status: 201)
        let asset = (try JSONSerialization.jsonObject(with: upload) as! [String: Any])["id"] as! String
        let request: [String: Any] = ["inputs": [["kind": "image", "role": "reference", "asset_id": asset]],
                                     "loras": [["id": "adapter", "role": "transformer", "strength": 0.5]]]
        let payload: [String: Any] = ["installation_id": "model1", "request": request]
        _ = try await json("/v1/plan", method: "POST", value: payload)
        try check((lastRequest["loras"] as? [[String: Any]])?.first?["path"] as? String == adapter.path, "LoRA registration was not resolved")
        let input = (lastRequest["inputs"] as! [[String: Any]])[0]["path"] as! String
        try check(FileManager.default.fileExists(atPath: input), "Reference upload missing")
        let submit = try await json("/v1/jobs", method: "POST", value: payload, status: 202)
        let id = submit["id"] as! String
        _ = try await json("/v1/jobs/\(id)")
        _ = try await exchange("/v1/jobs/\(id)/artifact", status: 400)
        _ = try await json("/v1/jobs/\(id)/cancel", method: "POST", value: [:])
        try check(records[id]?["state"] as? String == "cancelled", "Cancel not forwarded")
        let output = (records[id]!["request"] as! [String: Any])["output"] as! String
        try png.write(to: URL(fileURLWithPath: output))
        records[id]?["state"] = "succeeded"; records[id]?["result"] = ["output": output]
        let download = try await exchange("/v1/jobs/\(id)/artifact")
        try check(download == png, "Artifact data changed")
        records[id]?["result"] = ["output": "/etc/passwd"]
        _ = try await exchange("/v1/jobs/\(id)/artifact", status: 400)
        records[id]?["result"] = ["output": output]
        try FileManager.default.removeItem(atPath: output)
        _ = try await exchange("/v1/jobs/\(id)/artifact", status: 410)
        try png.write(to: URL(fileURLWithPath: output))
        records["outside"] = ["state": "succeeded", "request": ["output": "/etc/passwd"]]
        _ = try await exchange("/v1/jobs/outside", status: 400)
        let invalid: [[String: Any]] = [["output": "/tmp/x.png"], ["schema_version": true], ["schema_version": 2], ["model": "wrong"],
            ["inputs": "bad"], ["inputs": [["asset_id": "../bad"]]], ["inputs": [["asset_id": "absent"]]],
            ["loras": [1]], ["loras": [["id": "wrong"]]], ["loras": [["id": "absent"]]]]
        let before = submits
        for value in invalid { _ = try await json("/v1/jobs", method: "POST", value: ["installation_id": "model1", "request": value], status: 400) }
        _ = try await json("/v1/jobs", method: "POST", value: ["installation_id": "stale", "request": [:]], status: 400)
        try check(submits == before, "Invalid request reached submit")
        png[16] = 255
        _ = try await exchange("/v1/assets", method: "POST", data: png, headers: ["Content-Type": "image/png"], status: 400)
        // Two independent instances must never silently share a port.
        let conflict = LocalHTTPGateway(root: root.appendingPathComponent("other"), rpc: rpc)
        do { _ = try await conflict.start(port: port); throw NSError(domain: "Occupied port unexpectedly opened", code: 1) }
        catch { try check(conflict.port == 0, "Port conflict left listener active") }
        conflict.stop()
        _ = try await json("/v1/health")
        gateway.stop()
        try await Task.sleep(for: .milliseconds(100))
        do { _ = try await exchange("/v1/health"); throw NSError(domain: "Listener survived stop", code: 1) }
        catch let error as URLError { try check(error.code != .timedOut, "Stop left a hanging connection") }
        _ = try await gateway.start(port: port)
        _ = try await json("/v1/health")
        let recovered = try await exchange("/v1/jobs/\(id)/artifact")
        try check(recovered == download, "Restart lost native artifact ownership")
        gateway.stop()
        print("PASS: native HTTP health/catalog, upload/LoRA/plan/submit/status/cancel/artifact, JSON/framing/origin validation, port conflict, stop/restart and durable history")
    }
}
