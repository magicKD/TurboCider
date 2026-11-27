import CryptoKit
import Darwin
import Foundation

@MainActor final class LANWebServer: ObservableObject {
    enum Mode: String, CaseIterable, Identifiable {
        case off = "关闭", local = "仅本机", lan = "局域网"
        var id: Self { self }
    }
    struct Address: Identifiable, Equatable {
        let interface: String
        let ip: String
        var id: String { interface }
    }
    struct Device: Codable, Identifiable {
        let id: UUID
        let name: String
        let expires: Date
        var canManage: Bool? = nil
    }
    struct Approval: Identifiable {
        let device: Device
        let code: String
        var id: UUID { device.id }
    }
    private struct Pending {
        let approval: Approval
        let deadline: Date
    }
    private struct Frame {
        let device: UUID
        let app: String
        let digest: String
        let expires: Date
    }

    @Published private(set) var mode = Mode.off
    @Published private(set) var addresses: [Address] = []
    @Published private(set) var devices: [Device] = []
    @Published private(set) var approvals: [Approval] = []
    @Published private(set) var url: URL?
    @Published private(set) var pairingURL: URL?
    @Published private(set) var pairingExpires: Date?
    @Published private(set) var changing = false
    @Published private(set) var error: String?
    @Published private(set) var allowsDirectAccess = false

    private static let cookie = "tc_web"
    private static let sessionLifetime: TimeInterval = 7 * 24 * 60 * 60
    private let runtime: MiniAppRuntime
    private let studio: WebStudioRuntime
    private let assets: URL?
    private let deviceFile: URL
    private var registered: [String: Device] = [:] // SHA-256 of the browser cookie
    private var guests: [String: Device] = [:]
    private var frames: [String: Frame] = [:]
    private var pending: [String: Pending] = [:]
    private var ticket: String?
    private var gateway: LocalHTTPGateway?
    private var networkTask: Task<Void, Never>?
    private var activeInterface: String?
    private var generation = UUID()

    init(runtime: MiniAppRuntime, assets: URL? = Bundle.main.resourceURL?.appendingPathComponent("WebStudio"),
         library: ModelLibraryController? = nil, api: LocalAPIController? = nil) {
        self.runtime = runtime
        self.assets = assets
        studio = WebStudioRuntime(runtime: runtime, library: library ?? ModelLibraryController(), api: api)
        deviceFile = runtime.store.directory.appendingPathComponent("lan-web/devices.json")
        if let data = try? Data(contentsOf: deviceFile) {
            registered = (try? JSONDecoder().decode([String: Device].self, from: data)) ?? [:]
        }
        expire()
        refreshAddresses()
    }

    func refreshAddresses() {
        let current = Self.interfaceAddresses()
        if current != addresses { addresses = current }
    }

    func start(_ mode: Mode, interface: String? = nil, port: Int = 8766, direct: Bool? = nil) async {
        guard mode != .off else { stop(); return }
        let direct = direct ?? allowsDirectAccess
        stop()
        error = nil
        refreshAddresses()
        guard let assets, FileManager.default.isReadableFile(atPath: assets.appendingPathComponent("index.html").path),
              runtime.package(MiniAppPackage.officialID)?.builtIn == true else {
            error = "网页工作台资源未安装。"; return
        }
        guard !runtime.store.externalServiceActive else {
            error = "请先停止独立原生服务，再启动网页工作台。"; return
        }
        let address = interface.flatMap { name in addresses.first { $0.interface == name } } ?? (interface == nil ? addresses.first : nil)
        guard let host = mode == .local ? "127.0.0.1" : address?.ip else {
            error = "所选网卡没有可用的 IPv4 地址。请连接局域网后重试。"; return
        }
        changing = true
        let run = generation
        let http = LocalHTTPGateway(webHost: host, uploadDirectory: runtime.store.directory.appendingPathComponent("lan-web/incoming"), route: { [weak self] request in
            guard let self, self.generation == run else { throw Self.failure(503, "网页服务已关闭。") }
            return try await self.route(request)
        }, authorizeUpload: { [weak self] request in
            guard let self, self.generation == run else { return false }
            guard let device = self.device(request) else { return false }
            return request.path != "/api/studio/host/upload" || device.canManage == true
        }, onFailure: { [weak self] message in
            guard let self, self.generation == run else { return }
            self.stop(); self.error = message
        })
        gateway = http
        do {
            let bound = try await http.start(port: port)
            guard generation == run else { http.stop(); return }
            self.mode = mode
            activeInterface = mode == .lan ? address?.interface : nil
            url = URL(string: "http://\(host):\(bound)/")
            changing = false
            setDirectAccess(direct)
            networkTask = Task { [weak self] in
                while !Task.isCancelled {
                    do { try await Task.sleep(for: .seconds(5)) } catch { return }
                    guard let self, self.generation == run else { return }
                    self.expire()
                    guard let active = self.activeInterface else { continue }
                    self.refreshAddresses()
                    if self.addresses.first(where: { $0.interface == active })?.ip != host {
                        await self.start(mode, interface: active, port: bound)
                        return
                    }
                }
            }
        } catch {
            guard generation == run else { return }
            stop(); self.error = error.localizedDescription
        }
    }

    func stop() {
        generation = UUID()
        networkTask?.cancel(); networkTask = nil
        gateway?.stop(); gateway = nil
        activeInterface = nil
        mode = .off; url = nil; changing = false
        ticket = nil; pairingURL = nil; pairingExpires = nil
        pending.removeAll(); approvals = []
        guests.removeAll(); allowsDirectAccess = false
        frames.removeAll()
    }

    func setDirectAccess(_ enabled: Bool) {
        allowsDirectAccess = enabled
        if enabled {
            ticket = nil; pairingURL = nil; pairingExpires = nil
        } else {
            let guestIDs = Set(guests.values.map(\.id))
            guests.removeAll()
            frames = frames.filter { !guestIDs.contains($0.value.device) }
            issuePairingLink()
        }
    }

    func issuePairingLink() {
        guard !allowsDirectAccess, let url else { return }
        let token = Self.secret()
        var link = URLComponents(url: url.appendingPathComponent("pair"), resolvingAgainstBaseURL: false)
        link?.fragment = "ticket=\(token)"
        ticket = Self.digest(token)
        pairingURL = link?.url
        pairingExpires = Date().addingTimeInterval(180)
    }

    func approve(_ id: UUID, canManage: Bool = false) {
        expire()
        guard let entry = pending.first(where: { $0.value.approval.id == id }) else { return }
        guard registered.count < 32 else { error = "已配对 32 台设备，请先撤销一台。"; return }
        var updated = registered
        var device = entry.value.approval.device
        device.canManage = canManage
        updated[entry.key] = device
        do {
            try save(updated)
            reject(id)
        } catch { self.error = error.localizedDescription }
    }

    func reject(_ id: UUID) {
        pending = pending.filter { $0.value.approval.id != id }
        publishDevices()
    }

    func revoke(_ id: UUID) {
        do { try save(registered.filter { $0.value.id != id }) }
        catch { self.error = error.localizedDescription }
    }

    func setManagement(_ enabled: Bool, for id: UUID) {
        guard let key = registered.first(where: { $0.value.id == id })?.key else { return }
        var updated = registered
        updated[key]?.canManage = enabled
        do { try save(updated) }
        catch { self.error = error.localizedDescription }
    }

    private func save(_ updated: [String: Device]) throws {
        try FileManager.default.createDirectory(at: deviceFile.deletingLastPathComponent(), withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(updated).write(to: deviceFile, options: .atomic)
        registered = updated
        publishDevices()
        error = nil
    }

    private func expire() {
        let now = Date()
        if let deadline = pairingExpires, deadline <= now {
            ticket = nil; pairingURL = nil; pairingExpires = nil
        }
        if pending.values.contains(where: { $0.deadline <= now }) || registered.values.contains(where: { $0.expires <= now }) {
            pending = pending.filter { $0.value.deadline > now }
            registered = registered.filter { $0.value.expires > now }
        }
        guests = guests.filter { $0.value.expires > now }
        if frames.values.contains(where: { $0.expires <= now }) {
            frames = frames.filter { $0.value.expires > now }
        }
        if devices.count != registered.count || approvals.count != pending.count { publishDevices() }
    }

    private func publishDevices() {
        devices = registered.values.sorted { $0.expires > $1.expires }
        approvals = pending.values.sorted { $0.deadline < $1.deadline }.map(\.approval)
    }

    private static func interfaceAddresses() -> [Address] {
        var first: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&first) == 0, let first else { return [] }
        defer { freeifaddrs(first) }
        var rows: [String: Address] = [:]
        for item in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let net = item.pointee
            guard let address = net.ifa_addr, address.pointee.sa_family == AF_INET,
                  net.ifa_flags & UInt32(IFF_UP | IFF_RUNNING | IFF_BROADCAST) == UInt32(IFF_UP | IFF_RUNNING | IFF_BROADCAST),
                  net.ifa_flags & UInt32(IFF_LOOPBACK | IFF_POINTOPOINT) == 0 else { continue }
            var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(address, socklen_t(address.pointee.sa_len), &buffer, socklen_t(buffer.count),
                              nil, 0, NI_NUMERICHOST) == 0 else { continue }
            let name = String(cString: net.ifa_name), ip = String(cString: buffer)
            guard ip != "0.0.0.0", !ip.hasPrefix("169.254.") else { continue }
            if rows[name] == nil { rows[name] = Address(interface: name, ip: ip) }
        }
        return rows.values.sorted {
            if $0.interface.hasPrefix("en") != $1.interface.hasPrefix("en") { return $0.interface.hasPrefix("en") }
            return $0.interface.localizedStandardCompare($1.interface) == .orderedAscending
        }
    }

    private static func secret() -> String {
        (UUID().uuidString + UUID().uuidString).replacingOccurrences(of: "-", with: "").lowercased()
    }

    private static func digest(_ token: String) -> String { SHA256.hash(data: Data(token.utf8)).description }

    private func credential(_ request: LocalHTTPGateway.Request) -> String? {
        let cookies = request.headers["cookie"]?.split(separator: ";") ?? []
        for cookie in cookies {
            let pair = cookie.split(separator: "=", maxSplits: 1)
            if pair.count == 2, pair[0].trimmingCharacters(in: .whitespaces) == Self.cookie, pair[1].utf8.count == 64 {
                return Self.digest(String(pair[1]))
            }
        }
        return nil
    }

    private func device(_ request: LocalHTTPGateway.Request) -> Device? {
        guard let key = credential(request),
              let device = registered[key] ?? (allowsDirectAccess ? guests[key] : nil), device.expires > Date() else { return nil }
        return device
    }

    private func cookie(_ secret: String, lifetime: TimeInterval? = nil) -> [String: String] {
        ["Set-Cookie": "\(Self.cookie)=\(secret); Path=/; Max-Age=\(Int(lifetime ?? Self.sessionLifetime)); HttpOnly; SameSite=Strict"]
    }

    private func session(_ device: Device, key: String?) -> [String: Any] {
        ["state": "ready", "name": device.name, "direct": key.map { guests[$0] != nil } ?? false,
         "canManage": device.canManage == true, "workspace": device.id.uuidString]
    }

    private func connect(_ request: LocalHTTPGateway.Request) throws -> LocalHTTPGateway.Response {
        expire()
        if let device = device(request) { return try json(session(device, key: credential(request))) }
        guard allowsDirectAccess else { throw Self.failure(403, "此网页服务需要在 Mac 上配对。") }
        guard guests.count < 128 else { throw Self.failure(429, "直连设备已达上限。") }
        let secret = Self.secret(), key = Self.digest(secret)
        let device = Device(id: UUID(), name: "直连设备", expires: Date().addingTimeInterval(Self.sessionLifetime))
        guests[key] = device
        return try json(session(device, key: key), status: 201, headers: cookie(secret))
    }

    private static func failure(_ status: Int, _ message: String) -> LocalHTTPGateway.Failure { .init(status: status, message: message) }

    private func json(_ value: Any, status: Int = 200, headers: [String: String] = [:]) throws -> LocalHTTPGateway.Response {
        .init(status: status, data: try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]), headers: headers)
    }

    private func pair(_ request: LocalHTTPGateway.Request) throws -> LocalHTTPGateway.Response {
        expire()
        let body = try LocalHTTPGateway.object(request.body)
        guard let token = body["ticket"] as? String, token.utf8.count == 64, Self.digest(token) == ticket else {
            throw Self.failure(403, "配对链接已过期或已使用。请在 Mac 上刷新配对码。")
        }
        guard pending.count < 8, registered.count < 32 else { throw Self.failure(429, "待确认或已配对的设备已达上限。") }
        let source = (body["name"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let name = source.isEmpty ? "网页设备" : String(source.prefix(60))
        let secret = Self.secret()
        let device = Device(id: UUID(), name: name, expires: Date().addingTimeInterval(Self.sessionLifetime))
        let approval = Approval(device: device, code: String(format: "%06d", Int.random(in: 0...999999)))
        pending[Self.digest(secret)] = Pending(approval: approval, deadline: Date().addingTimeInterval(120))
        publishDevices()
        issuePairingLink()
        return try json(["state": "pending", "code": approval.code], status: 201, headers: cookie(secret))
    }

    private func siteAsset(_ name: String, mimeType: String? = nil) throws -> LocalHTTPGateway.Response {
        guard let file = assets?.appendingPathComponent(name),
              let mime = mimeType ?? MiniAppPackage.mimeTypes[file.pathExtension] else { throw Self.failure(404, "找不到网页资源。") }
        return .init(status: 200, data: try Data(contentsOf: file), mime: mime.hasPrefix("text/") ? "\(mime); charset=utf-8" : mime)
    }

    private func route(_ request: LocalHTTPGateway.Request) async throws -> LocalHTTPGateway.Response {
        let path = String(request.path.prefix(while: { $0 != "?" }))
        switch (request.method, path) {
        case ("GET", "/"), ("GET", "/pair"): return try siteAsset("index.html")
        case ("GET", "/shell.js"), ("GET", "/workspace.js"), ("GET", "/shell.css"), ("GET", "/community.js"),
             ("GET", "/studio-mark.webp"), ("GET", "/amber-study.webp"), ("GET", "/look-outfit.webp"),
             ("GET", "/look-pose.webp"), ("GET", "/template-cutout.webp"):
            return try siteAsset(String(path.dropFirst()))
        case ("GET", "/studio-icons.svg"): return try siteAsset("studio-icons.svg", mimeType: "image/svg+xml")
        case ("GET", "/cover.webp"):
            guard let data = runtime.package(MiniAppPackage.officialID)?.files["samples/demo-person.webp"] else {
                throw Self.failure(404, "封面资源不可用。")
            }
            return .init(status: 200, data: data, mime: "image/webp")
        case ("POST", "/api/pair"): return try pair(request)
        case ("POST", "/api/connect"): return try connect(request)
        case ("GET", "/api/session"):
            expire()
            if let device = device(request) { return try json(session(device, key: credential(request))) }
            if let key = credential(request), let record = pending[key] {
                return try json(["state": "pending", "code": record.approval.code])
            }
            return try json(["state": "unpaired", "directAvailable": allowsDirectAccess])
        case ("POST", "/api/logout"):
            if let key = credential(request) {
                guests.removeValue(forKey: key)
                if registered[key] != nil { try save(registered.filter { $0.key != key }) }
            }
            return try json(["state": "unpaired"], headers: cookie("", lifetime: 0))
        default: break
        }
        if request.method == "GET", path.hasPrefix("/_frame/") { return try frameAsset(path) }
        guard path.hasPrefix("/api/") || path.hasPrefix("/apps/") || path.hasPrefix("/_materials/") else {
            throw Self.failure(404, "找不到网页。")
        }
        guard let device = device(request) else {
            if request.method == "GET", path.hasPrefix("/apps/"), path.hasSuffix("/") {
                return .init(status: 303, data: Data(), mime: "text/plain", headers: ["Location": "/"])
            }
            throw Self.failure(401, "请在 Mac 上配对此设备。")
        }
        do {
            if path.hasPrefix("/api/studio/") {
                return try await studio.route(request, path: path, device: device)
            }
            if request.method == "GET", path == "/api/apps" {
                return try json(runtime.packages.map { app in [
                    "id": app.id, "name": app.manifest.name, "version": app.manifest.version, "official": app.builtIn,
                    "description": app.manifest.description ?? "", "url": "/apps/\(app.id)/",
                    "permissions": app.manifest.permissions.map(\.title)
                ] as [String: Any] })
            }
            if request.method == "POST", path == "/api/frame" { return try openFrame(request, device: device.id) }
            guard let app = runtime.package(MiniAppPackage.officialID), app.builtIn else {
                throw Self.failure(404, "应用未安装。")
            }
            let prefix = "/apps/\(app.id)/"
            if request.method == "GET", path.hasPrefix("/apps/"), !path.hasPrefix(prefix) {
                let parts = path.split(separator: "/")
                guard parts.count >= 2, let community = runtime.package(String(parts[1])), !community.builtIn else {
                    throw Self.failure(404, "应用未安装。")
                }
                if path == "/apps/\(community.id)/" { return try siteAsset("community.html") }
                return try await appRoute(request, path: path, app: community, device: device.id, prefix: "/apps/\(community.id)/")
            }
            if request.method == "POST", ["/api/call", "/api/media"].contains(path) {
                let id = path == "/api/call" ? try LocalHTTPGateway.object(request.body)["app"] as? String
                    : request.headers["x-turbocider-app"]
                guard let id, let selected = runtime.package(id) else { throw Self.failure(404, "应用未安装。") }
                return try await appRoute(request, path: path, app: selected, device: device.id, prefix: "/apps/\(selected.id)/",
                                          managesHost: device.canManage == true && selected.builtIn)
            }
            return try await appRoute(request, path: path, app: app, device: device.id, prefix: prefix, managesHost: device.canManage == true)
        } catch let error as MiniAppRuntime.Failure {
            let status = ["mediaNotFound", "jobNotFound"].contains(error.code) ? 404
                : error.code == "permissionDenied" ? 403 : 400
            return try json(["error": MiniAppRuntime.redact(error.message), "code": error.code], status: status)
        } catch let error as LocalHTTPGateway.Failure {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Self.failure(503, MiniAppRuntime.redact(error.localizedDescription))
        }
    }

    private func openFrame(_ request: LocalHTTPGateway.Request, device: UUID) throws -> LocalHTTPGateway.Response {
        let body = try LocalHTTPGateway.object(request.body)
        guard let id = body["app"] as? String, let app = runtime.package(id), !app.builtIn else {
            throw Self.failure(404, "社区应用未安装。")
        }
        expire()
        let recentlyOpened = frames.filter { $0.value.device == device && $0.value.app == id }
        if recentlyOpened.count >= 4, let oldest = recentlyOpened.min(by: { $0.value.expires < $1.value.expires })?.key {
            frames.removeValue(forKey: oldest)
        }
        guard frames.count < 256 else { throw Self.failure(429, "打开的社区应用太多，请稍后重试。") }
        let token = Self.secret()
        frames[token] = Frame(device: device, app: app.id, digest: app.digest, expires: Date().addingTimeInterval(8 * 60 * 60))
        return try json(["src": "/_frame/\(token)/\(app.manifest.entry)", "token": token, "name": app.manifest.name,
                         "export": app.manifest.permissions.contains(.exportMedia)])
    }

    private func frameAsset(_ path: String) throws -> LocalHTTPGateway.Response {
        let pieces = path.dropFirst("/_frame/".count).split(separator: "/", maxSplits: 1)
        guard pieces.count == 2, let frame = frames[String(pieces[0])], frame.expires > Date(),
              let app = runtime.package(frame.app), app.digest == frame.digest,
              (registered.values.contains { $0.id == frame.device && $0.expires > Date() } ||
               (allowsDirectAccess && guests.values.contains { $0.id == frame.device && $0.expires > Date() })),
              let url, let host = url.host else { throw Self.failure(404, "应用会话已失效，请重新打开。") }
        let name = String(pieces[1]).removingPercentEncoding ?? ""
        let origin = url.port == 80 ? "http://\(host)" : String(url.absoluteString.dropLast())
        let base = "\(origin)/_frame/\(pieces[0])/"
        let policy = ["sandbox allow-scripts", "default-src 'none'", "script-src \(base)", "style-src 'unsafe-inline' \(base)",
                      "img-src data: blob: \(base)", "font-src \(base)", "connect-src \(base)", "media-src blob: \(base)",
                      "base-uri 'none'", "frame-ancestors \(origin)", "form-action 'none'"].joined(separator: "; ")
        // Opaque-origin frames can read their own static package only.
        let headers = ["Access-Control-Allow-Origin": "null"]
        if name == "_host.js" {
            let config: [String: Any] = ["token": String(pieces[0]), "origin": origin,
                "settings": runtime.storageSnapshot(app, device: frame.device)]
            var script = Data("window.TurboCiderFrame = ".utf8)
            script.append(try JSONSerialization.data(withJSONObject: config))
            script.append(Data(";\n".utf8))
            script.append(try siteAsset("frame-sdk.js").data)
            return .init(status: 200, data: script, mime: "text/javascript; charset=utf-8", headers: headers, sandboxPolicy: policy)
        }
        guard MiniAppPackage.validPath(name), let bytes = app.files[name],
              let mime = MiniAppPackage.mimeTypes[(name as NSString).pathExtension.lowercased()] else {
            throw Self.failure(404, "找不到社区应用资源。")
        }
        if mime == "text/html", var html = String(data: bytes, encoding: .utf8) {
            let script = "<script src=\"\(base)_host.js\"></script>"
            if let head = html.range(of: #"<head\b[^>]*>"#, options: [.caseInsensitive, .regularExpression]) {
                html.insert(contentsOf: script, at: head.upperBound)
            } else if let doctype = html.range(of: #"<!doctype[^>]*>"#, options: [.caseInsensitive, .regularExpression]) {
                html.insert(contentsOf: script, at: doctype.upperBound)
            } else { html = script + html }
            return .init(status: 200, data: Data(html.utf8), mime: "text/html; charset=utf-8", headers: headers, sandboxPolicy: policy)
        }
        return .init(status: 200, data: bytes, mime: mime, headers: headers, sandboxPolicy: policy)
    }

    private func appRoute(_ request: LocalHTTPGateway.Request, path: String, app: MiniAppPackage,
                          device: UUID, prefix: String, managesHost: Bool = false) async throws -> LocalHTTPGateway.Response {
        switch (request.method, path) {
        case ("GET", "/api/jobs"):
            return try json(await runtime.call("jobs.list", payload: [:], app: app, device: device))
        case ("POST", "/api/call"):
            let body = try LocalHTTPGateway.object(request.body)
            guard Set(body.keys) == ["app", "method", "payload"], body["app"] as? String == app.id,
                  let method = body["method"] as? String, let payload = body["payload"] as? [String: Any] else {
                throw Self.failure(400, "无效的应用请求。")
            }
            return try json(await runtime.call(method, payload: payload, app: app, device: device, managesHost: managesHost))
        case ("POST", "/api/media"):
            guard request.headers["x-turbocider-app"] == app.id else { throw Self.failure(400, "无效的应用请求。") }
            let name = request.headers["x-turbocider-filename"]?.removingPercentEncoding
            return try json(await runtime.importBrowserImage(request.body, name: name, app: app, device: device), status: 201)
        default: break
        }
        guard request.method == "GET" else { throw Self.failure(404, "找不到网页接口。") }
        if path.hasPrefix("/_materials/") {
            let resource = try await runtime.materialResource(String(path.dropFirst("/_materials/".count)), app: app)
            return .init(status: 200, data: resource.data, mime: resource.mime)
        }
        guard path.hasPrefix(prefix) else { throw Self.failure(404, "找不到应用资源。") }
        let name = String(path.dropFirst(prefix.count))
        if name.hasPrefix("_media/") {
            guard let id = UUID(uuidString: String(name.dropFirst("_media/".count))) else { throw Self.failure(404, "找不到图片。") }
            let download = request.path == path + "?download=1"
            let (file, mime) = try runtime.mediaFile(id, package: app, device: device, exporting: download)
            let headers = download ? ["Content-Disposition": "attachment; filename=\"TurboCider-\(id.uuidString).\(file.pathExtension)\""] : [:]
            return .init(status: 200, data: Data(), mime: mime, file: file, headers: headers)
        }
        guard app.builtIn else { throw Self.failure(404, "社区资源仅在应用沙盒内可用。") }
        if name == "_host.js" {
            let config: [String: Any] = ["app": app.id, "base": prefix, "settings": runtime.storageSnapshot(app, device: device),
                                         "canManage": managesHost]
            let data = try JSONSerialization.data(withJSONObject: config)
            var script = Data("window.TurboCiderWebConfig = ".utf8)
            script.append(data)
            script.append(Data(";\n".utf8))
            script.append(try siteAsset("sdk.js").data)
            return .init(status: 200, data: script, mime: "text/javascript; charset=utf-8")
        }
        let file = name.isEmpty ? app.manifest.entry : (name.removingPercentEncoding ?? "")
        guard MiniAppPackage.validPath(file), let data = app.files[file],
              let mime = MiniAppPackage.mimeTypes[(file as NSString).pathExtension.lowercased()] else {
            throw Self.failure(404, "找不到应用资源。")
        }
        if file == app.manifest.entry {
            let marker = "<!-- turbocider-web-host -->"
            guard var html = String(data: data, encoding: .utf8), let target = html.range(of: marker) else {
                throw Self.failure(503, "应用缺少网页宿主入口。")
            }
            html.replaceSubrange(target, with: "<script src=\"_host.js\"></script>")
            return .init(status: 200, data: Data(html.utf8), mime: "text/html; charset=utf-8")
        }
        return .init(status: 200, data: data, mime: mime)
    }
}
