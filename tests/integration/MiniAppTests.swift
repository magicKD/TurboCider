import AppKit
import Foundation
import WebKit

private actor MaterialFixture {
    let webp: Data
    let png: Data
    private(set) var requests: [URL] = []
    private var failsFull = false
    private var delaysFull = false

    init(webp: Data, png: Data) { self.webp = webp; self.png = png }
    func failFullImages(_ enabled: Bool) { failsFull = enabled }
    func delayFullImages(_ enabled: Bool) { delaysFull = enabled }

    private var catalogs: [String: Data] {
        [
            "poses": Data(#"""
            [
              {"id":"full_single_P020","no":"P020","zh":"站姿示例","line":"全身","cat":"站姿","angles":[
                {"zh":"正面","en":"front","t":"p/t/front.webp","i":"p/i/front.webp"},
                {"zh":"斜侧","en":"three-quarter","t":"p/t/diag.webp","i":"p/i/diag.webp"}]},
              {"id":"unsafe","zh":"不读取","angles":[{"t":"https://example.invalid/thumbnail.webp","i":"p/i/front.webp"}]}
            ]
            """#.utf8),
            "depths": Data(#"""
            {"items":[{"id":"full_single_P020","pack":"gia-depth","zh":"站姿示例","angles":[
                {"zh":"正面","t":"https://pwimg.0vo.pro/d/thumbs/front.jpg","i":"https://pwimg.0vo.pro/d/img/front.png"},
                {"zh":"斜侧","t":"https://pwimg.0vo.pro/d/thumbs/diag.jpg","i":"https://pwimg.0vo.pro/d/img/diag.png"}]}]}
            """#.utf8),
            "portraits": Data(#"""
            [{"id":"f71f980dde3","ti":"短发 · 正脸","sec":"正脸","t":"r/t/face.webp","i":"r/m/face.webp"}]
            """#.utf8),
            "costumes": Data(#"""
            [{"id":"SPR-0001","ti":"牛仔套装","cat":"春季","t":"c/t/coat.webp","i":"c/i/coat.webp"}]
            """#.utf8),
            "hair": Data(#"""
            [{"id":"H001","ti":"长发","t":"https://charadb-2fi.pages.dev/thumb/H001.webp","i":"https://charadb-2fi.pages.dev/img/H001.jpg"}]
            """#.utf8),
            "dynasties": Data(#"""
            [{"id":"C001","ti":"历代服饰","angles":[
              {"zh":"正面","t":"y/t/c001-front.webp","i":"y/i/c001-front.webp"},
              {"zh":"斜侧","t":"y/t/c001-diag.webp","i":"y/i/c001-diag.webp"}]}]
            """#.utf8)
        ]
    }

    func read(_ url: URL, limit: Int) async throws -> Data {
        requests.append(url)
        try assertSource(url)
        let files = Dictionary(uniqueKeysWithValues: catalogs.map { label, body in
            ("\(label).\(PeopleMaterialSources.sha256(body).prefix(10)).json", body)
        })
        let script = Data(files.keys.sorted().map { "'/assets/\($0)'" }.joined(separator: ",").utf8)
        let scriptName = "app.\(PeopleMaterialSources.sha256(script).prefix(10)).js"
        if url == PeopleMaterialSources.posewear { return Data("<script src=\"/assets/\(scriptName)\"></script>".utf8) }
        if url.lastPathComponent == scriptName { return script }
        if let data = files[url.lastPathComponent] { return data }
        if url.path.contains("/y/i/") {
            if failsFull { throw PeopleStudioMaterials.Failure.http(503) }
            if delaysFull { try await Task.sleep(for: .seconds(2)) }
        }
        if ["webp", "jpg"].contains(url.pathExtension) { return webp }
        if url.pathExtension == "png" { return png }
        throw PeopleStudioMaterials.Failure.http(503)
    }

    private func assertSource(_ url: URL) throws {
        if !PeopleMaterialSources.allows(url) { throw PeopleStudioMaterials.Failure.invalidSource }
    }

}

@main
@MainActor
struct MiniAppTests {
    struct Check: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw Check(message) }
    }

    static func rejected(_ code: String, _ work: () async throws -> Any) async throws {
        do { _ = try await work() }
        catch let failure as MiniAppRuntime.Failure {
            try require(failure.code == code, "Expected \(code), got \(failure.code): \(failure.message)")
            return
        }
        throw Check("Expected rejection: \(code)")
    }

    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
            exit(1)
        }
    }

    static func run() async throws {
        let args = CommandLine.arguments
        guard args.count == 3 else { throw Check("Supply built-in app directory and example .tcapp.") }
        try testCanvasGeometry()
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TurboCiderMiniAppsTests-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        setenv("TURBOCIDER_MODEL_LIBRARY", directory.appendingPathComponent("model-library").path, 1)
        let store = NativeJobStore(directory: directory)
        let studio = StudioState(directory: directory)
        let root = URL(fileURLWithPath: args[1], isDirectory: true).standardizedFileURL
        let liveMaterials = ProcessInfo.processInfo.environment["TURBOCIDER_TEST_LIVE_PEOPLE_MATERIALS"] == "1"
        let fixture = try MaterialFixture(webp: Data(contentsOf: root.appendingPathComponent("samples/demo-pose-front.webp")),
                                          png: Data(contentsOf: root.appendingPathComponent("samples/expr.png")))
        let materialRoot = directory.appendingPathComponent("test-materials")
        let materials: PeopleStudioMaterials
        if liveMaterials { materials = PeopleStudioMaterials(root: materialRoot) }
        else { materials = PeopleStudioMaterials(root: materialRoot, reader: { try await fixture.read($0, limit: $1) }) }
        let runtime = MiniAppRuntime(store: store, studio: studio, builtIn: root, materials: materials)
        guard let official = runtime.package(MiniAppPackage.officialID) else { throw Check(runtime.message ?? "Missing bundled application") }
        let example = try MiniAppPackage(source: URL(fileURLWithPath: args[2]))
        try require(official.builtIn && !example.builtIn && official.digest.count == 64, "Package trust and identity")
        try require(!MiniAppPackage.validPath("../asset.png") && !MiniAppPackage.validPath("a/%2E%2E") &&
                    !MiniAppPackage.validPath("a//b") && MiniAppPackage.validPath("assets/one.png"), "Package path guard")
        try runtime.install(example)
        let repeated = MiniAppRuntime(store: store, studio: studio, builtIn: root)
        try require(repeated.package(example.id)?.digest == example.digest, "Installed bundle must survive restart")
        let modelPath = ProcessInfo.processInfo.environment["TURBOCIDER_MINI_APP_TEST_MODEL"]
        if let modelPath { studio.draft.modelPaths[Qwen21Installation.modelID] = modelPath }
        else {
            let fake = directory.appendingPathComponent("metadata-only-qwen")
            for path in ["diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                         "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors",
                         "processor/tokenizer.json"] {
                let url = fake.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                if path.hasPrefix("text_encoders/") {
                    let source = directory.appendingPathComponent("planning-encoder.safetensors")
                    try Data("planning fixture".utf8).write(to: source)
                    try FileManager.default.createSymbolicLink(at: url, withDestinationURL: source)
                } else {
                    try Data("planning fixture".utf8).write(to: url)
                }
            }
            studio.draft.modelPaths[Qwen21Installation.modelID] = fake.path
        }
        let image = try await runtime.call("media.bundled", payload: ["path": "samples/demo-person.webp"], app: official) as! [String: Any]
        let again = try await runtime.call("media.bundled", payload: ["path": "samples/demo-person.webp"], app: official) as! [String: Any]
        try require(image["id"] as? String == again["id"] as? String, "Bundled material identity is stable")
        try await rejected("mediaNotFound") {
            try await runtime.call("media.get", payload: ["id": image["id"]!], app: example)
        }
        try await rejected("permissionDenied") {
            try await runtime.call("adapters.list", payload: [:], app: example)
        }
        let imageBytes = official.files["samples/demo-person.webp"]!.base64EncodedString()
        let exampleImage = try await runtime.call("media.import", payload: ["data": imageBytes, "name": "Example.webp"], app: example) as! [String: Any]
        try require(exampleImage["id"] as? String != image["id"] as? String, "Imported media has an application-owned identifier")
        _ = try await runtime.call("storage.set", payload: ["key": "test", "value": "official"], app: official)
        let other = try await runtime.call("storage.get", payload: ["key": "test"], app: example)
        try require(other is NSNull, "Storage is scoped")
        try await rejected("invalidRequest") {
            try await runtime.call("storage.get", payload: ["key": "../other"], app: example)
        }
        try await rejected("jobNotFound") {
            try await runtime.call("jobs.get", payload: ["id": UUID().uuidString], app: example)
        }

        let (officialHost, officialView) = try await open(official, runtime)
        let (exampleHost, exampleView) = try await open(example, runtime)
        defer { officialHost.disconnect(officialView); exampleHost.disconnect(exampleView) }
        let bridge = try await officialView.callAsyncJavaScript("""
            const blocked = await fetch('http://127.0.0.1:8765/').then(() => false, () => true);
            await new Promise(resolve => setTimeout(resolve, 25));
            return {
              media: await fetch('samples/demo-person.webp').then(r => r.ok && r.headers.get('Content-Type') === 'image/webp'),
              context: (await TurboCider.call('handshake')).appID,
              preference: TurboCiderStorage.test,
              isolated: await fetch('tcapp://dev.turbocider.image-sample/index.html').then(r => !r.ok, () => true),
              networkBlocked: blocked && miniAppTestErrors.some(error => error.startsWith('connect-src: http://127.0.0.1')),
              ui: !!window.WB && !!window.APP && !!window.HIS,
              scripts: [typeof PeopleStudioNative, typeof PoseWearCatalog, typeof APP, typeof WB, typeof HIS],
              scriptErrors: window.miniAppTestErrors
            };
            """, arguments: [:], in: nil, contentWorld: .page) as? [String: Any] ?? [:]
        try require(bridge["context"] as? String == official.id, "WebKit bridge cannot read the app origin: \(bridge)")
        try require(bridge["media"] as? Bool == true && bridge["ui"] as? Bool == true, "Built-in WebKit assets and UI did not load: \(bridge)")
        try require(bridge["preference"] as? String == "official" && bridge["isolated"] as? Bool == true &&
                    bridge["networkBlocked"] as? Bool == true, "WebKit isolation and storage: \(bridge)")
        let separate = try await exampleView.callAsyncJavaScript("""
            try { await TurboCider.media.get(id); return 'leaked'; }
            catch (error) { return error.code; }
            """, arguments: ["id": image["id"]!], in: nil, contentWorld: .page)
        try require(separate as? String == "mediaNotFound", "Foreign media must not cross WebView scopes")
        let flow = try await officialView.callAsyncJavaScript("""
            await Promise.all(PoseWearCatalog.COLLECTIONS.map(tab => APP.load(tab.id)));
            await APP.setTab('studio');
            await WB.setSel({
              portrait: [{k:'r:studio-portrait'}], garment: [{k:'c:studio-costume'}],
              pose: [{k:'p:studio-standing',ai:0}], expr: [],
              extra: []
            });
            const prompt = document.querySelector('#wbPrompt').value;
            return { prompt, assets: [WB._W.persons.length, WB._W.garments.length, WB._W.poses.length],
                     gallery: PoseWearCatalog.COLLECTIONS.length, ready: document.querySelector('#wbGo').disabled === false };
            """, arguments: [:], in: nil, contentWorld: .page) as? [String: Any] ?? [:]
        try require(flow["gallery"] as? Int == 12 && flow["assets"] as? [Int] == [1, 1, 1] &&
                    flow["ready"] as? Bool == true && !(flow["prompt"] as? String ?? "").isEmpty, "Person, outfit and pose should compose: \(flow)")
        let paired = try await officialView.callAsyncJavaScript("""
            const sketch = APP.ITEMS['p:studio-standing'];
            const match = APP.poseCounterpart(sketch, 1);
            return [match?.item.type, match?.item.angles[match.angle].zh];
            """, arguments: [:], in: nil, contentWorld: .page)
        try require(paired as? [String] == ["depth", "斜侧"], "A pose must pair with the matching depth camera angle")
        try await testMaterials(officialView, exampleView, materials, fixture: fixture, materialRoot: materialRoot, live: liveMaterials)
        let payload: [String: Any] = ["requestID": UUID().uuidString, "workflow": "test.dependency",
            "shots": [
                ["prompt": "Retain the person's identity", "references": [["mediaID": image["id"]!, "role": "person"]]],
                ["prompt": "Change the pose", "references": [["shotIndex": 0, "role": "person"]]]
            ]]
        let stages = try await runtime.call("workflow.plan", payload: payload, app: official) as? [String: Any]
        try require((stages?["jobs"] as? [[String: Any]])?.compactMap { $0["referenceCount"] as? Int } == [1, 1],
                    "Dependent shots must preflight in order")
        let geometry = try await officialView.callAsyncJavaScript("""
            const options = Array.from(document.querySelector('#wbAR').options, option => option.value);
            const plan = await TurboCider.workflow.plan({
              requestID: TurboCider.createRequestID(), workflow: 'test.canvas',
              shots: [{prompt: 'Preserve the original portrait composition', references: [{mediaID,role:'person'}],
                       width: 384, height: 768, referencePixelBudget: 512*512}]
            });
            const shot = plan.jobs[0];
            return {dimensions: [options.includes('1:2'), shot.width, shot.height, shot.referencePixelBudget], residency: shot.residency};
            """, arguments: ["mediaID": image["id"]!], in: nil, contentWorld: .page) as? [String: Any]
        try require((geometry?["dimensions"] as? [NSNumber])?.map(\.intValue) == [1, 384, 768, 512 * 512] &&
                    (modelPath != nil || geometry?["residency"] as? String == "streamed"),
                    "WebKit selection and plan must preserve independent canvas and reference budgets: \(String(describing: geometry))")
        var invalid = payload
        invalid["shots"] = [["prompt": "Forward reference", "references": [["shotIndex": 1, "role": "person"]]]]
        try await rejected("invalidDependency") {
            try await runtime.call("workflow.plan", payload: invalid, app: official)
        }
        if modelPath == nil {
            let queued = UUID(), requestID = UUID()
            let request = NativeRequest(prompt: "queued", output: directory.appendingPathComponent("outputs/queued.png").path)
            try store.enqueueMiniAppJobs([NativeJob(id: queued, createdAt: Date(), request: request, state: "queued", phase: "queued",
                completed: 0, total: 8, elapsed: 0, miniAppOrigin: MiniAppJobOrigin(appID: official.id,
                    appVersion: official.manifest.version, appName: official.manifest.name, workflow: "test.cancel", requestID: requestID,
                    label: nil, references: []))])
            _ = try await runtime.call("jobs.cancel", payload: ["id": queued.uuidString], app: official)
            try require(store.jobs.first?.state == "cancelled", "Cancelling a queued job must persist without invoking the GPU")
        }
        if modelPath == nil {
            let savedDirectory = directory.appendingPathComponent("reopened")
            let output = savedDirectory.appendingPathComponent("outputs/test.png")
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            try official.files["samples/expr.png"]!.write(to: output)
            let id = UUID(), requestID = UUID()
            let request = NativeRequest(prompt: "Saved history", output: output.path)
            let job = NativeJob(id: id, createdAt: Date(), request: request, state: "succeeded", phase: "done",
                completed: 8, total: 8, elapsed: 1, miniAppOrigin: MiniAppJobOrigin(appID: official.id,
                    appVersion: official.manifest.version, appName: official.manifest.name, workflow: "test.saved",
                    requestID: requestID, label: "Example", context: #"{"inputs":{"poses":[{}]}}"#, references: []))
            try JSONEncoder().encode([job]).write(to: savedDirectory.appendingPathComponent("jobs.json"))
            let savedStore = NativeJobStore(directory: savedDirectory)
            let reopened = MiniAppRuntime(store: savedStore, studio: studio, builtIn: root)
            let (reopenedHost, reopenedView) = try await open(official, reopened)
            defer { reopenedHost.disconnect(reopenedView) }
            let history = try await historySnapshot(reopenedView)
            try require(history.first?.first as? String == id.uuidString, "Persisted history reload in WebKit: \(history)")
        }
        if modelPath != nil, ProcessInfo.processInfo.environment["TURBOCIDER_MINI_APP_TEST_GPU"] == "1" {
            let staged = ProcessInfo.processInfo.environment["TURBOCIDER_MINI_APP_TEST_STAGED"] == "1"
            let portrait = !staged && ProcessInfo.processInfo.environment["TURBOCIDER_MINI_APP_TEST_PORTRAIT"] == "1"
            if liveMaterials || portrait {
                let selection = try await officialView.callAsyncJavaScript("""
                    await APP.setTab('studio');
                    await WB.setSel({
                      portrait: [{k: live ? 'r:f71f980dde3' : 'r:studio-portrait'}],
                      garment: [{k: live ? 'c:SPR-0001' : 'c:studio-costume'}],
                      pose: [{k: live ? 'p:full_single_P020' : 'p:studio-standing',ai:0}],
                      expr: [], extra: portrait ? [{k: 's:studio-scene'}] : []
                    });
                    return [WB._W.persons.length, WB._W.garments.length, WB._W.poses.length, WB._W.extraRefs.length];
                    """, arguments: ["live": liveMaterials, "portrait": portrait], in: nil, contentWorld: .page)
                try require(selection as? [Int] == [1, 1, 1, portrait ? 1 : 0], "Requested catalog references not selected")
            }
            _ = try await officialView.callAsyncJavaScript("""
                document.querySelector('#wbTwoStage').checked = useStages;
                document.querySelector('#wbAR').value = ratio;
                """, arguments: ["useStages": staged, "ratio": portrait ? "1:2" : "1:1"], in: nil, contentWorld: .page)
            let complete = try await officialView.callAsyncJavaScript("""
                const result = await document.querySelector('#wbGo').onclick();
                if (!result.got.length) return {error: result.errs.map(e => e.stack).join('; ')};
                const record = result.got[0];
                const after = await fetch(record.after);
                const image = await createImageBitmap(await after.blob());
                await document.querySelector('#wbUse').onclick();
                return {width: image.width, height: image.height, id: record.tcID, source: WB._W.persons[0].tcMediaID};
                """, arguments: [:], in: nil, contentWorld: .page) as? [String: Any] ?? [:]
            try require(complete["width"] as? Int == (portrait ? 384 : 512) &&
                        complete["height"] as? Int == (portrait ? 768 : 512) &&
                        complete["id"] as? String == complete["source"] as? String, "GPU and reuse in WebKit: \(complete)")
            let job = try unwrap(store.jobs.first, "Missing persisted job")
            try require(job.hasOutput && job.request.steps == 8 &&
                        job.request.inputs?.count == (staged ? 2 : portrait ? 4 : 3) &&
                        job.miniAppOrigin?.workflow == (staged ? "people.outfit-pose.staged" : "people.outfit-pose"),
                        "Expected the requested Turbo workflow")
            if portrait {
                let receipt = try JSONSerialization.jsonObject(with: Data((job.resultJSON ?? "{}").utf8)) as? [String: Any]
                let pixels = receipt?["qwen21_reference_pixels"] as? Int ?? 0
                try require(job.request.qwen21_reference_pixel_budget == 512 * 512 && job.request.qwen21_reference_size == nil &&
                            (1...(512 * 512)).contains(pixels), "Four reference rasters must fit the single pixel budget")
                if job.request.residency == "streamed" {
                    let kv = try unwrap(receipt?["qwen21_reference_kv"] as? [String: Any], "Missing streamed KV receipt")
                    if ProcessInfo.processInfo.environment["TURBOCIDER_QWEN21_STREAM_PREFIX_KV"] == "disk" {
                        try require(kv["mode"] as? String == "disk" && kv["layer_reads"] as? Int == 32 * 7 &&
                                    (kv["active_bytes"] as? Int ?? Int.max) < (kv["total_bytes"] as? Int ?? 0),
                                    "File-backed KV must read one layer at a time during all seven cached steps")
                    }
                    try require(receipt?["qwen_convrot_streaming"] != nil, "Missing streamed weight receipt")
                }
                print("PASS four-reference 384×768 Turbo, \(pixels) encoded reference pixels, \(job.request.residency ?? "unknown")")
            }
            if liveMaterials {
                try require(["full_single_P020", "f71f980dde3", "SPR-0001"].allSatisfy {
                    job.miniAppOrigin?.context?.contains($0) == true
                },
                            "Generation must freeze real online references in the native job")
            }
            if staged {
                let prior = try unwrap(store.jobs.dropFirst().first, "Missing staged parent")
                try require(prior.hasOutput && prior.request.inputs?.count == 2 &&
                            job.miniAppOrigin?.dependencies == [prior.id] && job.request.inputs?.first?.path == prior.request.output,
                            "The outfit stage must consume the published pose output")
            }
            if let artifacts = ProcessInfo.processInfo.environment["TURBOCIDER_MINI_APP_TEST_ARTIFACTS"] {
                let output = URL(fileURLWithPath: artifacts, isDirectory: true)
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                let prefix = staged ? "people-studio-staged" : (portrait ? "people-studio-portrait" : "people-studio-single")
                let nativeImage = try Data(contentsOf: URL(fileURLWithPath: job.request.output))
                try nativeImage.write(to: output.appendingPathComponent(prefix + ".png"), options: .atomic)
                try JSONEncoder().encode(store.jobs).write(to: output.appendingPathComponent(prefix + "-jobs.json"), options: .atomic)
            }
            try await rejected("jobNotFound") {
                try await runtime.call("jobs.get", payload: ["id": job.id.uuidString], app: example)
            }
            let savedStore = NativeJobStore(directory: directory)
            try require(savedStore.jobs.first?.id == job.id && savedStore.jobs.first?.hasOutput == true,
                        "Finished job and provenance survive restart")
            let newRuntime = MiniAppRuntime(store: savedStore, studio: studio, builtIn: root)
            let (reopenedHost, reopenedView) = try await open(official, newRuntime)
            defer { reopenedHost.disconnect(reopenedView) }
            let history = try await historySnapshot(reopenedView)
            try require(history.first?.first as? String == job.id.uuidString, "History reload in WebKit: \(history)")
            if liveMaterials {
                let restored = try await reopenedView.callAsyncJavaScript("""
                    const status = await TurboCider.call('materials.status');
                    const record = HIS.getList().find(item => item.tcID === jobID);
                    await APP.setTab('studio');
                    const problems = await WB.restoreRecord(record);
                    const refs = [WB._W.persons[0]?.tcMediaID, APP.ITEMS[WB._W.garments[0]?.k]?.tcMediaID,
                                  APP.ITEMS[WB._W.poses[0]?.k]?.tcMediaID];
                    const inputs = [record.inputs.persons[0].mediaID, record.inputs.garments[0].mediaID, record.inputs.poses[0].mediaID];
                    return !status.online && !problems.length && refs.every((id, i) => id === inputs[i]);
                    """, arguments: ["jobID": job.id.uuidString], in: nil, contentWorld: .page)
                try require(restored as? Bool == true, "History must restore the exact saved real references offline")
                print("PASS live references preserved for offline history re-edit")
            }
            print("PASS WebKit → native GPU → published output → history → reuse (\(staged ? "two stages" : "one pass"))")
        }
        try runtime.uninstall(example)
        try await rejected("appChanged") {
            try await runtime.call("handshake", payload: [:], app: example)
        }
        print("PASS WebKit official/community UI, media, typed dependencies, cancellation, isolation and persisted installation")
    }

    private static func testCanvasGeometry() throws {
        let asset = StudioAsset(path: "/portrait.png", name: "Portrait", width: 1000, height: 2000)
        let outfit = StudioAsset(path: "/outfit.png", name: "Outfit", width: 512, height: 768)
        let shot = try JSONDecoder().decode(MiniAppShot.self, from: Data(#"""
            {"prompt":"Fit the portrait","matchReferenceAspect":true,"referencePixelBudget":262144,
             "references":[{"role":"garment"},{"role":"person"}]}
            """#.utf8))
        let output = URL(fileURLWithPath: "/output.png")
        let request = try shot.native(model: Qwen21Installation.modelID, variant: Qwen21Installation.turboID,
                                      steps: 8, output: output, assets: [outfit, asset], loras: [], supportsWeightStreaming: true)
        let ratio = Double(request.width) / Double(request.height)
        let pixels = request.width * request.height
        try require(abs(ratio - 0.5) < 0.04 && abs(pixels - 512 * 512) < 16 * 1024 &&
                    request.residency == "streamed" && request.qwen21_reference_pixel_budget == 512 * 512 && request.qwen21_reference_size == nil,
                    "Fitting must preserve the person's aspect and resolve to the shared-pixel weight-streamed path")
        let adapter = NativeLoRA(path: "/adapter.safetensors", strength: 1, role: "transformer")
        let adapted = try shot.native(model: Qwen21Installation.modelID, variant: Qwen21Installation.turboID,
                                      steps: 8, output: output, assets: [outfit, asset], loras: [adapter])
        try require(adapted.width == 512 && adapted.height == 512 && adapted.loras?.count == 1 &&
                    adapted.residency == "component_staged" && adapted.qwen21_reference_pixel_budget == 512 * 512,
                    "Automatic aspect with a Turbo LoRA must use its supported canvas and keep the adapter")
    }

    private static func testMaterials(_ official: WKWebView, _ example: WKWebView, _ materials: PeopleStudioMaterials,
                              fixture: MaterialFixture, materialRoot: URL, live: Bool) async throws {
        let initial = await materials.status()
        let requestCount = await fixture.requests.count
        try require(!initial.online && requestCount == 0 && !FileManager.default.fileExists(atPath: materialRoot.path),
                    "Default gallery must make no remote requests and no library download")
        let demo = try await official.callAsyncJavaScript("""
            await APP.setTab('studio');
            await document.querySelector('#wbExample').onclick();
            document.querySelector('#tcMaterials').click();
            await new Promise(resolve => setTimeout(resolve, 40));
            const result = [WB._W.persons.length, WB._W.garments.length, WB._W.poses.length,
                            document.querySelector('#tcMaterialsOnline').checked];
            document.querySelector('#tcMaterialsClose').click();
            return result;
            """, arguments: [:], in: nil, contentWorld: .page)
        let demoValues = demo as? [NSNumber] ?? []
        try require(demoValues.map(\.intValue) == [1, 1, 1, 0], "Offline Demo and initial materials toggle: \(String(describing: demo))")
        let denied = try await example.callAsyncJavaScript("""
            try { await TurboCider.call('materials.status'); return 'leaked'; }
            catch (error) { return error.code; }
            """, arguments: [:], in: nil, contentWorld: .page)
        try require(denied as? String == "permissionDenied", "Community app must not use the first-party material transport")
        let remote = try await official.callAsyncJavaScript("""
            await document.querySelector('#tcMaterials').onclick();
            const optIn = document.querySelector('#tcMaterialsOnline');
            optIn.checked = true;
            await optIn.onchange();
            document.querySelector('#tcMaterialsClose').click();
            const names = live ? PoseWearCatalog.COLLECTIONS.map(item => item.id) : ['pose','depth','costume','portrait','hair','dyn'];
            const lists = await Promise.all(names.map(name => APP.load(name)));
            const counts = Object.fromEntries(names.map((name, i) => [name, lists[i].filter(it => it.materialID).length]));
            const pose = APP.ITEMS['p:full_single_P020'];
            const match = pose && APP.poseCounterpart(pose, 1);
            const images = await Promise.all(lists.map(async (list, i) => {
              const row = list.find(item => item.materialID && APP.ok(item));
              if (!row) return [names[i], false];
              const response = await fetch(row.t);
              if (!response.ok) return [names[i], false];
              const bitmap = await createImageBitmap(await response.blob());
              const ok = bitmap.width > 0 && bitmap.height > 0;
              bitmap.close();
              return [names[i], ok];
            }));
            const view = pose?.angles?.[1];
            const picked = view && await TurboCider.call('materials.claim', {id:view.materialID});
            const hair = lists[names.indexOf('hair')].find(item => item.materialID);
            const previewOnly = await TurboCider.call('materials.claim',{id:hair.materialID}).then(() => false, () => true);
            return {counts, images, sid:match?.item.id, angle:match?.angle, previewOnly,
                    claimed:picked?.url?.includes('/_media/') || false,
                    copiedURL:PoseWearReferences.reference(pose, 1, location.href).url};
            """, arguments: ["live": live], in: nil, contentWorld: .page) as? [String: Any] ?? [:]
        let counts = remote["counts"] as? [String: Int] ?? [:]
        let images = remote["images"] as? [[Any]] ?? []
        let expected = live ? 12 : 6
        try require(counts.count == expected && counts.values.allSatisfy { $0 > 0 } && images.count == expected &&
                    images.allSatisfy { $0.last as? Bool == true }, "Actual catalog and preview decoding: \(remote)")
        try require(remote["sid"] as? String == "full_single_P020" && remote["angle"] != nil &&
                    remote["claimed"] as? Bool == true && remote["previewOnly"] as? Bool == true &&
                    (remote["copiedURL"] as? String)?.hasPrefix("https://posewear.pages.dev/p/i/") == true,
                    "Online view pairing and claim: \(remote)")
        try require(!FileManager.default.fileExists(atPath: materialRoot.path), "Online browsing and single-image claim downloaded a full pack")
        if live {
            print("PASS live material sources in WKWebView: \(counts.sorted { $0.key < $1.key })")
            if ProcessInfo.processInfo.environment["TURBOCIDER_TEST_LIVE_PEOPLE_PACK"] == "1" {
                await materials.enableOnline(false)
                try await materials.startDownload(.prop)
                let pack = try await watchPack(materials, "installed", collection: .prop, attempts: 3_000)
                let offline = PeopleStudioMaterials(root: materialRoot, reader: { _, _ in throw Check("Offline pack attempted to fetch") })
                let offlineRows = try await offline.rows(.prop)
                let row = try unwrap(offlineRows.first, "Missing installed props")
                let fields = try JSONSerialization.jsonObject(with: row) as? [String: Any]
                let id = try unwrap(fields?["materialID"] as? String, "Missing installed prop image")
                let image = try await offline.sourceImage(id, full: true)
                try require(offlineRows.count == counts["prop"] && image.mime == "image/webp" && pack.completed == pack.total && pack.bytes > 0,
                            "The published props must be available after reopening offline")
                print("PASS live licensed prop pack: \(offlineRows.count) entries, \(pack.completed) views, \(pack.bytes) bytes")
                await materials.enableOnline(true)
            }
            return
        }
        try require(counts["pose"] == 1, "Directory must filter unsuitable series and image paths")
        do {
            try await materials.startDownload(.portrait)
            throw Check("Portrait catalog incorrectly accepted an offline pack")
        } catch PeopleStudioMaterials.Failure.notDownloadable {}
        await materials.enableOnline(false)
        let before = await fixture.requests.count
        let offlineDemo = try await materials.rows(.pose)
        let after = await fixture.requests.count
        try require(offlineDemo.isEmpty && after == before, "Turn-off should hide online rows without network")
        await fixture.failFullImages(true)
        try await materials.startDownload(.dyn)
        try await watchPack(materials, "failed")
        await fixture.failFullImages(false)
        await fixture.delayFullImages(true)
        try await materials.startDownload(.dyn)
        await materials.cancelDownload(.dyn)
        try await watchPack(materials, "cancelled")
        await fixture.delayFullImages(false)
        try await materials.startDownload(.dyn)
        try await watchPack(materials, "installed")
        let offline = PeopleStudioMaterials(root: materialRoot, reader: { _, _ in throw Check("Offline pack attempted to fetch") })
        let reopened = try await offline.rows(.dyn)
        let picked = try await offline.sourceImage("dyn/C001/v1", full: true)
        try require(reopened.count == 1 && picked.mime == "image/webp", "The completed downloaded pack must open in a new offline host")
        print("PASS materials opt-in, source scope, single image claim, download failure/cancel/retry and offline restart")
    }

    @discardableResult
    static func watchPack(_ runtime: PeopleStudioMaterials, _ phase: String, collection: PeopleMaterialCollection = .dyn,
                          attempts: Int = 100) async throws -> PeopleStudioMaterials.CollectionStatus {
        for _ in 0..<attempts {
            if let status = await runtime.status().collections.first(where: { $0.id == collection }) {
                if status.phase == phase { return status }
                if status.phase == "failed" { throw Check("\(collection) download failed: \(status.error ?? "unknown error")") }
            }
            try await Task.sleep(for: .milliseconds(40))
        }
        throw Check("Expected \(collection) download phase \(phase)")
    }

    static func historySnapshot(_ view: WKWebView) async throws -> [[Any]] {
        try await view.callAsyncJavaScript("""
            await APP.setTab('history');
            await HIS.refresh();
            return HIS.getList().map(rec => [rec.tcID, rec.inputs?.poses?.length || 0]);
            """, arguments: [:], in: nil, contentWorld: .page) as? [[Any]] ?? []
    }

    static func unwrap<Value>(_ value: Value?, _ message: String) throws -> Value {
        guard let value else { throw Check(message) }
        return value
    }

    static func open(_ package: MiniAppPackage, _ runtime: MiniAppRuntime) async throws -> (MiniAppWebHost, WKWebView) {
        let host = MiniAppWebHost(package: package, runtime: runtime)
        let view = host.makeView()
        view.configuration.userContentController.addUserScript(WKUserScript(source: """
            window.miniAppTestErrors = [];
            window.addEventListener('error', e => miniAppTestErrors.push(e.message || e.target.src || e.target.href || 'load error'), true);
            window.addEventListener('securitypolicyviolation', e => miniAppTestErrors.push(e.violatedDirective + ': ' + e.blockedURI));
            """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        for _ in 0..<100 {
            try await Task.sleep(for: .milliseconds(100))
            if (try? await view.evaluateJavaScript("Boolean(window.TurboCider && document.readyState === 'complete')")) as? Bool == true {
                return (host, view)
            }
        }
        throw Check("WKWebView did not load \(package.id): \(String(describing: view.url)), \(runtime.message ?? "")")
    }
}
