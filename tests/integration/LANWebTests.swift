import Foundation
import AVFoundation
import ImageIO
import UniformTypeIdentifiers

@main @MainActor struct LANWebTests {
    struct Check: LocalizedError {
        let message: String
        var errorDescription: String? { message }
        init(_ message: String) { self.message = message }
    }

    struct Reply {
        let status: Int
        let response: HTTPURLResponse
        let data: Data
        func object() throws -> [String: Any] { try JSONSerialization.jsonObject(with: data) as! [String: Any] }
        func text() -> String { String(decoding: data, as: UTF8.self) }
    }

    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
            exit(1)
        }
    }

    static func check(_ condition: Bool, _ message: String) throws {
        if !condition { throw Check(message) }
    }

    static func sessionCookie(_ reply: Reply) throws -> String {
        guard let raw = reply.response.value(forHTTPHeaderField: "Set-Cookie"), let value = raw.split(separator: ";").first else {
            throw Check("No session cookie (\(reply.status))")
        }
        return String(value)
    }

    static func origin(_ base: URL) -> String {
        "\(base.scheme!)://\(base.host!):\(base.port!)"
    }

    static func fetch(_ base: URL, _ path: String, json: [String: Any]? = nil, cookie: String? = nil,
                      headers: [String: String] = [:], bytes: Data? = nil, sendOrigin: Bool = true) async throws -> Reply {
        let config = URLSessionConfiguration.ephemeral
        config.httpShouldSetCookies = false
        config.timeoutIntervalForRequest = 15
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }
        var request = URLRequest(url: URL(string: origin(base) + path)!)
        if let cookie { request.setValue(cookie, forHTTPHeaderField: "Cookie") }
        if json != nil || bytes != nil {
            request.httpMethod = "POST"
            request.httpBody = try bytes ?? JSONSerialization.data(withJSONObject: json!)
            request.setValue(bytes == nil ? "application/json" : "application/octet-stream", forHTTPHeaderField: "Content-Type")
            if sendOrigin { request.setValue(origin(base), forHTTPHeaderField: "Origin") }
        }
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        let (data, response) = try await session.data(for: request)
        let http = response as! HTTPURLResponse
        return Reply(status: http.statusCode, response: http, data: data)
    }

    static func call(_ url: URL, _ method: String, _ payload: [String: Any], cookie: String) async throws -> Reply {
        try await fetch(url, "/api/call", json: ["app": MiniAppPackage.officialID, "method": method, "payload": payload], cookie: cookie)
    }

    static func pair(_ web: LANWebServer, at url: URL, name: String) async throws -> (String, UUID) {
        let ticket = web.pairingURL!.fragment!.replacingOccurrences(of: "ticket=", with: "")
        let payload = ["ticket": ticket, "name": name]
        let badOrigin = try await fetch(url, "/api/pair", json: payload, headers: ["Origin": "http://other.invalid"])
        try check(badOrigin.status == 403, "Wrong origin consumed the pairing link")
        let reply = try await fetch(url, "/api/pair", json: payload)
        guard reply.status == 201, let approval = web.approvals.first(where: { $0.device.name == name }),
              let rawCookie = reply.response.value(forHTTPHeaderField: "Set-Cookie"), let first = rawCookie.split(separator: ";").first else {
            throw Check("Pairing did not return a confirmation and a cookie (\(reply.status))")
        }
        try check(rawCookie.contains("HttpOnly") && rawCookie.contains("SameSite=Strict"), "Pairing cookie is missing attributes")
        let cookie = String(first)
        let notYet = try await fetch(url, "/api/apps", cookie: cookie)
        try check(notYet.status == 401, "Device accessed the app before approval")
        let replay = try await fetch(url, "/api/pair", json: payload)
        try check(replay.status == 403, "Pairing link can be reused")
        web.approve(approval.id)
        let session = try await fetch(url, "/api/session", cookie: cookie).object()
        try check(session["state"] as? String == "ready", "Approval did not activate the browser session")
        return (cookie, approval.id)
    }

    static func run() async throws {
        let args = CommandLine.arguments
        guard args.count == 3 || (args.count == 7 && args[3] == "--serve") else {
            throw Check("Usage: turbocider-lan-web-tests PEOPLE_STUDIO WEB_STUDIO [--serve STATE MODEL local|lan]")
        }
        let serving = args.count == 7
        let root = serving ? URL(fileURLWithPath: args[4]) : FileManager.default.temporaryDirectory.appendingPathComponent("tc-web-tests-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { if !serving { try? FileManager.default.removeItem(at: root) } }
        setenv("TURBOCIDER_MODEL_LIBRARY", root.appendingPathComponent("library").path, 1)
        let store = NativeJobStore(directory: root)
        let studio = StudioState(directory: root)
        let appRoot = URL(fileURLWithPath: args[1])
        let runtime = MiniAppRuntime(store: store, studio: studio, builtIn: appRoot)
        guard let app = runtime.package(MiniAppPackage.officialID) else { throw Check(runtime.message ?? "Missing official app") }
        let mode: LANWebServer.Mode = serving && args[6] == "lan" ? .lan : .local
        let model = serving ? URL(fileURLWithPath: args[5]) : root.appendingPathComponent("planning-model")
        if !serving {
            for name in ["diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                         "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors",
                         "processor/tokenizer.json"] {
                let file = model.appendingPathComponent(name)
                try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data("plan metadata fixture".utf8).write(to: file)
            }
        }
        let install = try LibraryStore().register(modelID: Qwen21Installation.modelID, path: model)
        studio.draft.modelPaths[Qwen21Installation.modelID] = model.path
        if serving {
            for (key, model) in [("TURBOCIDER_WEB_TEST_LTX", "ltx-2.5-distilled"), ("TURBOCIDER_WEB_TEST_RELAY", "relayvsr-2x")] {
                if let path = ProcessInfo.processInfo.environment[key] {
                    _ = try LibraryStore().register(modelID: model, path: URL(fileURLWithPath: path))
                }
            }
        }
        let api = LocalAPIController(directory: root, httpPort: 0)
        let assets = URL(fileURLWithPath: args[2])
        let web = LANWebServer(runtime: runtime, assets: assets)
        await web.start(mode, port: 0)
        await api.startShared(store: store)
        defer { web.stop(); api.stop() }
        guard let url = web.url, api.running, let apiURL = URL(string: api.httpURL), !store.externalServiceActive else {
            throw Check("Web / loopback API did not coexist: \(web.error ?? api.error ?? "")")
        }
        if serving {
            try await serveBrowserTest(root: root, web: web, apiURL: apiURL, installation: install, store: store, model: model)
            return
        }

        let landing = try await fetch(url, "/")
        try check(landing.status == 200 && landing.text().contains("viewport"), "Responsive application shell missing")
        for path in ["/studio-mark.webp", "/amber-study.webp", "/look-outfit.webp", "/look-pose.webp", "/template-cutout.webp"] {
            let response = try await fetch(url, path)
            try check(response.status == 200 && response.response.mimeType == "image/webp" &&
                      CGImageSourceCreateWithData(response.data as CFData, nil) != nil, "Undecodable studio artwork at \(path)")
        }
        let icons = try await fetch(url, "/studio-icons.svg")
        try check(icons.status == 200 && icons.response.mimeType == "image/svg+xml" && icons.text().contains("<symbol"), "Missing studio icons")
        let anonymous = try await fetch(url, "/api/apps")
        try check(anonymous.status == 401, "Anonymous app access")
        let apiOnLAN = try await fetch(url, "/v1/health")
        try check(apiOnLAN.status == 404, "Programmatic API exposed on Web listener")
        let (phone, phoneID) = try await pair(web, at: url, name: "Phone")
        let (tablet, tabletID) = try await pair(web, at: url, name: "Tablet")
        let appPath = "/apps/\(app.id)/"
        let entry = try await fetch(url, appPath, cookie: phone)
        let isolated = entry.response.value(forHTTPHeaderField: "Cross-Origin-Resource-Policy")
        try check(entry.status == 200 && entry.text().contains("_host.js") && isolated == "same-origin", "Missing protected app bootstrap")
        let asset = try await fetch(url, appPath + "samples/demo-person.webp", cookie: phone)
        try check(asset.status == 200 && asset.response.mimeType == "image/webp", "Missing bundled demo")
        let bad = try await fetch(url, "/api/media", bytes: asset.data)
        try check(bad.status == 401, "Upload endpoint accepted an anonymous request")
        let headers = ["X-TurboCider-App": app.id, "X-TurboCider-Filename": "portrait.webp"]
        let uploaded = try await fetch(url, "/api/media", cookie: phone, headers: headers, bytes: asset.data)
        try check(uploaded.status == 201, "Browser upload failed with \(uploaded.status) \(uploaded.text())")
        let media = try uploaded.object()
        let imageURL = media["url"] as! String
        let photo = try await fetch(url, imageURL + "?download=1", cookie: phone)
        try check(photo.status == 200 && photo.response.value(forHTTPHeaderField: "Content-Disposition")?.hasPrefix("attachment") == true,
                  "Browser download missing")
        let foreignPhoto = try await fetch(url, imageURL, cookie: tablet)
        try check(foreignPhoto.status == 404, "Second device accessed first device's photo")
        let heic = NSMutableData()
        if let source = CGImageSourceCreateWithData(asset.data as CFData, nil),
           let original = CGImageSourceCreateImageAtIndex(source, 0, nil),
           let encoded = CGImageDestinationCreateWithData(heic, UTType.heic.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(encoded, original, [kCGImagePropertyOrientation: 6] as CFDictionary)
            if CGImageDestinationFinalize(encoded) {
                let converted = try await fetch(url, "/api/media", cookie: phone, headers: headers, bytes: heic as Data)
                let metadata = try converted.object()
                let preview = try await fetch(url, metadata["url"] as! String, cookie: phone)
                let scale = min(1.0, 1600.0 / Double(max(original.width, original.height)))
                try check(converted.status == 201 && preview.response.mimeType == "image/png" &&
                          metadata["width"] as? Int == Int((Double(original.height) * scale).rounded()) &&
                          metadata["height"] as? Int == Int((Double(original.width) * scale).rounded()),
                          "Phone HEIC preview was not upright PNG")
                print("Phone HEIC upload and EXIF orientation passed")
            }
        }
        let desktopPicker = try await call(url, "models.chooseTurbo", [:], cookie: phone)
        try check(desktopPicker.status == 403, "LAN can invoke a Mac file picker")
        let reference = try await call(url, "media.bundled", ["path": "samples/demo-person.webp"], cookie: tablet).object()

        let requestID = UUID().uuidString
        func payload(_ id: Any) -> [String: Any] {
            ["requestID": requestID, "workflow": "people.pose", "shots": [[
                "prompt": "A person standing in a studio.", "references": [["mediaID": id, "role": "person"]],
                "referencePixelBudget": 512 * 512
            ]]]
        }
        let firstPlan = try await call(url, "workflow.plan", payload(media["id"]!), cookie: phone)
        let secondPlan = try await call(url, "workflow.plan", payload(reference["id"]!), cookie: tablet)
        try check(firstPlan.status == 200 && secondPlan.status == 200, "Concurrent-device preflight failed")
        let planA = try firstPlan.object()["id"]!
        let planB = try secondPlan.object()["id"]!
        let foreignPlan = try await call(url, "jobs.submit", ["id": planB], cookie: phone)
        try check(foreignPlan.status == 400, "An unowned plan was accepted")
        let phoneSubmit = try await call(url, "jobs.submit", ["id": planA], cookie: phone)
        let secondSubmit = try await call(url, "jobs.submit", ["id": planB], cookie: tablet)
        let retry = try await call(url, "jobs.submit", ["id": planA], cookie: phone)
        let tasks = try JSONSerialization.jsonObject(with: phoneSubmit.data) as! [[String: Any]]
        let retries = try JSONSerialization.jsonObject(with: retry.data) as! [[String: Any]]
        try check(secondSubmit.status == 200 && tasks.map { $0["id"] as! String } == retries.map { $0["id"] as! String },
                  "Device plans invalidated one another or request was duplicated")

        let apiRequest = ["installation_id": install.id, "request": [
            "prompt": "A garden in sunlight", "model_variant": Qwen21Installation.turboID,
            "width": 384, "height": 384, "steps": 8, "residency": "streamed", "allow_approximation": true
        ] as [String: Any]] as [String: Any]
        let apiSubmit = try await fetch(apiURL, "/v1/jobs", json: apiRequest, sendOrigin: false)
        try check(apiSubmit.status == 202, "Shared local API failed to enqueue: \(apiSubmit.text())")
        let id = try apiSubmit.object()["id"] as! String
        try check(store.jobs.contains(where: { $0.id.uuidString == id && $0.automationRequestJSON != nil }) &&
                  store.jobs.filter { $0.miniAppOrigin?.requestID.uuidString == requestID }.count == 2, "Web and API did not write to the same store")
        try check(!store.externalServiceActive, "A browser or API listener reserved the GPU")
        let foreignJob = try await call(url, "jobs.get", ["id": tasks[0]["id"]!], cookie: tablet)
        try check(foreignJob.status == 404, "Second device accessed first device's job")
        await api.refreshJobs(offset: 0)
        try check(api.jobs.first?.id == id, "Shared API history did not load")
        try await testWorkspace(web, url: url, phone: phone, tablet: tablet, tabletID: tabletID,
                                image: asset.data, installation: install, apiJob: id, root: root)
        web.stop()
        let restarted = LANWebServer(runtime: runtime, assets: assets)
        await restarted.start(.local, port: 0)
        defer { restarted.stop() }
        let ready = try await fetch(restarted.url!, "/api/apps", cookie: phone)
        try check(ready.status == 200, "Stored device credential did not survive a listener restart")
        restarted.revoke(phoneID)
        let revoked = try await fetch(restarted.url!, imageURL, cookie: phone)
        try check(revoked.status == 401, "Revoked device still had media access")
        for _ in 0..<100 where store.busy || store.hasQueuedGeneration { try await Task.sleep(for: .milliseconds(30)) }
        try check(!restarted.allowsDirectAccess && restarted.pairingURL != nil, "Direct access remained enabled after restart")
        print("LAN web integration passed: pairing and direct sessions, workspace isolation, image and video uploads, range playback, and shared queue")
    }

    static func testWorkspace(_ web: LANWebServer, url: URL, phone: String, tablet: String,
                              tabletID: UUID, image: Data, installation: LibraryInstallation,
                              apiJob: String, root: URL) async throws {
        let off = try await fetch(url, "/api/connect", json: [:])
        try check(off.status == 403 && !web.allowsDirectAccess, "Pairing was bypassed by default")
        let noCatalog = try await fetch(url, "/api/studio/catalog")
        try check(noCatalog.status == 401, "Unpaired browser obtained model information")
        web.setDirectAccess(true)
        try check(web.pairingURL == nil, "Direct mode displayed a pairing token")
        let invalidOrigin = try await fetch(url, "/api/connect", json: [:], headers: ["Origin": "http://different.invalid"])
        try check(invalidOrigin.status == 403, "Direct connections accepted a foreign origin")
        let joined = try await fetch(url, "/api/connect", json: [:])
        let guest = try sessionCookie(joined)
        let id = try joined.object()["workspace"] as? String
        let joinedAgain = try await fetch(url, "/api/connect", json: [:], cookie: guest)
        let current = try joinedAgain.object()
        try check(joined.status == 201 && current["workspace"] as? String == id && current["direct"] as? Bool == true &&
                  current["canManage"] as? Bool == false, "Direct browser session was not isolated or stable")

        let catalog = try await fetch(url, "/api/studio/catalog", cookie: guest).object()
        let models = catalog["models"] as? [[String: Any]] ?? []
        try check(models.contains(where: { $0["id"] as? String == "real-esrgan-x2plus" }) &&
                  models.contains(where: { $0["id"] as? String == "ltx-2.5-distilled" }) &&
                  !String(describing: models).contains(installation.path), "Catalog omitted executors or leaked Mac paths")
        let imageUpload = try await fetch(url, "/api/studio/upload", cookie: guest,
            headers: ["X-TurboCider-Kind": "image", "X-TurboCider-Filename": "person.webp"], bytes: image)
        let record = try imageUpload.object()
        guard imageUpload.status == 201, let media = record["id"] as? String, let mediaURL = record["url"] as? String else {
            throw Check("Workspace image upload: \(imageUpload.text())")
        }
        let preview = try await fetch(url, mediaURL + "/preview?size=320", cookie: guest)
        let other = try await fetch(url, mediaURL, cookie: phone)
        try check(preview.status == 200 && preview.response.mimeType == "image/png" && other.status == 404,
                  "Workspace did not preserve private thumbnails")
        let scaled = try await fetch(url, "/api/studio/prepare", json: ["id": media, "preset": "fit512"], cookie: guest)
        let resized = try scaled.object()
        try check(scaled.status == 201 && resized["id"] as? String != media && (resized["height"] as? Int ?? .max) <= 512,
                  "Web reference preparation did not make an immutable derivative")
        let request: [String: Any] = ["requestID": UUID().uuidString, "installation": installation.id,
            "operation": "image.edit", "prompt": "A person in studio light", "media": [media], "width": 384, "height": 768,
            "qwenReference": "budget"]
        let foreign = try await fetch(url, "/api/studio/submit", json: request, cookie: phone)
        try check(foreign.status == 404, "Paired browser used a guest's references")
        let submit = try await fetch(url, "/api/studio/submit", json: request, cookie: guest)
        guard submit.status == 201, let job = storeJobPayload(submit.data) else { throw Check("Browser workspace submit: \(submit.text())") }
        let replay = try await fetch(url, "/api/studio/submit", json: request, cookie: guest)
        try check(storeJobPayload(replay.data) == job, "Retried workspace request enqueued twice")
        let own = try await fetch(url, "/api/studio/jobs", cookie: guest).object()
        let ownJobs = own["items"] as? [[String: Any]] ?? []
        try check(ownJobs.first?["id"] as? String == job && !ownJobs.contains { $0["id"] as? String == apiJob },
                  "Guest jobs crossed the browser workspace boundary")

        let forbidden = try await fetch(url, "/api/studio/host", json: ["action": "refresh"], cookie: guest)
        try check(forbidden.status == 403, "A direct session controlled host management")
        web.setManagement(true, for: tabletID)
        let hostJobs = try await fetch(url, "/api/studio/jobs?scope=host", cookie: tablet).object()
        try check((hostJobs["items"] as? [[String: Any]])?.first?["id"] as? String == apiJob, "Manager did not see Mac API history")
        let preventGuestLeak = try await fetch(url, mediaURL, cookie: tablet)
        try check(preventGuestLeak.status == 404, "A manager saw a guest's private materials")
        web.setManagement(false, for: tabletID)
        let revokedManager = try await fetch(url, "/api/studio/host", json: ["action": "refresh"], cookie: tablet)
        try check(revokedManager.status == 403, "Revoking management left an active privilege")

        let clip = root.appendingPathComponent("clip.mp4")
        try await videoFixture(clip)
        let bytes = try Data(contentsOf: clip)
        let video = try await fetch(url, "/api/studio/upload", cookie: guest,
            headers: ["X-TurboCider-Kind": "video", "X-TurboCider-Filename": "clip.mp4"], bytes: bytes)
        let videoURL = try video.object()["url"] as? String
        guard video.status == 201, let videoURL else { throw Check("Workspace video upload: \(video.text())") }
        let range = try await fetch(url, videoURL, cookie: guest, headers: ["Range": "bytes=0-1"])
        let suffix = try await fetch(url, videoURL, cookie: guest, headers: ["Range": "bytes=-4"])
        let badRange = try await fetch(url, videoURL, cookie: guest, headers: ["Range": "bytes=\(bytes.count)-"])
        try check(range.status == 206 && range.data == bytes.prefix(2) && range.response.value(forHTTPHeaderField: "Content-Range") == "bytes 0-1/\(bytes.count)" &&
                  suffix.status == 206 && suffix.data == bytes.suffix(4) && badRange.status == 416, "Video byte-range playback is invalid")

        web.setDirectAccess(false)
        let old = try await fetch(url, mediaURL, cookie: guest)
        let paired = try await fetch(url, "/api/apps", cookie: phone)
        try check(old.status == 401 && paired.status == 200 && web.pairingURL != nil, "Switching off direct access broke guest or paired sessions")
        web.setDirectAccess(true)
        let newer = try await fetch(url, "/api/connect", json: [:], cookie: guest)
        try check(try newer.object()["workspace"] as? String != id, "Revoked guest session was revived")
        web.setDirectAccess(false)
    }

    static func storeJobPayload(_ data: Data) -> String? {
        let result = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        return result?.first?["id"] as? String
    }

    static func videoFixture(_ file: URL) async throws {
        let writer = try AVAssetWriter(outputURL: file, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64])
        writer.add(input)
        try check(writer.startWriting(), "Cannot start video fixture")
        writer.startSession(atSourceTime: .zero)
        guard let pool = adaptor.pixelBufferPool else { throw Check("Missing video fixture buffer pool") }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        guard let buffer else { throw Check("Cannot allocate fixture video frame") }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let address = CVPixelBufferGetBaseAddress(buffer) { memset(address, 160, CVPixelBufferGetDataSize(buffer)) }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        for index in 0..<9 {
            for _ in 0..<100 where !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
            try check(adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(index), timescale: 24)), "Cannot append video fixture")
        }
        input.markAsFinished()
        await writer.finishWriting()
        try check(writer.status == .completed, "Cannot finish video fixture: \(writer.error?.localizedDescription ?? "")")
    }

    /// Starts the same production handlers for browser-driven GPU acceptance.
    /// Browser tests mark requests explicitly so only test devices are approved.
    static func serveBrowserTest(root: URL, web: LANWebServer, apiURL: URL, installation: LibraryInstallation,
                                 store: NativeJobStore, model: URL) async throws {
        var descriptor: [String: String] = ["web": web.url!.absoluteString, "pair": web.pairingURL!.absoluteString,
            "api": apiURL.absoluteString, "install": installation.id]
        let handoff = root.appendingPathComponent("browser-test.json")
        try JSONEncoder().encode(descriptor).write(to: handoff, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: handoff.path)
        let desktop = root.appendingPathComponent("submit-desktop")
        let direct = root.appendingPathComponent("direct-web")
        let clip = root.appendingPathComponent("clip.mp4")
        try await videoFixture(clip)
        let stop = root.appendingPathComponent("stop-test")
        var desktopTest: Task<Void, Never>?
        defer { desktopTest?.cancel() }
        for _ in 0..<2400 {
            for approval in web.approvals where approval.device.name.hasPrefix("Acceptance ") {
                web.approve(approval.id, canManage: approval.device.name.hasSuffix("Manager"))
            }
            let directRequested = FileManager.default.fileExists(atPath: direct.path)
            if directRequested != web.allowsDirectAccess { web.setDirectAccess(directRequested) }
            if let link = web.pairingURL?.absoluteString, descriptor["pair"] != link {
                descriptor["pair"] = link
                try JSONEncoder().encode(descriptor).write(to: handoff, options: .atomic)
            }
            if desktopTest == nil, FileManager.default.fileExists(atPath: desktop.path) {
                desktopTest = Task {
                    var request = NativeRequest(prompt: "A desktop submission of a green apple on a white plate.",
                                                output: root.appendingPathComponent("outputs/desktop.png").path)
                    request.model = Qwen21Installation.modelID; request.model_variant = Qwen21Installation.turboID
                    request.width = 384; request.height = 384; request.steps = 8
                    request.residency = "streamed"; request.allow_approximation = true
                    do { _ = try await store.generateInQueue(modelURL: model, request: request) }
                    catch { print("Desktop job: \(error.localizedDescription)") }
                }
            }
            if FileManager.default.fileExists(atPath: stop.path) { break }
            try await Task.sleep(for: .milliseconds(200))
        }
    }
}
