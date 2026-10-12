import Foundation
import Darwin

private final class ProgressProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Int64] = []
    func record(_ completed: Int64, _ total: Int64) {
        lock.lock(); defer { lock.unlock() }
        if completed > 0 && total > 0 { samples.append(completed) }
    }
    var observed: Bool {
        lock.lock(); defer { lock.unlock() }
        return !samples.isEmpty
    }
}

@main struct HubClientTests {
    static func main() async throws {
        guard CommandLine.arguments.count == 3, let endpoint = URL(string: CommandLine.arguments[1]) else {
            throw LibraryFailure(message: "hub-client-tests FIXTURE_ENDPOINT TEMP_ROOT")
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[2]).resolvingSymlinksInPath()
        let store = try LibraryStore(root: root.appendingPathComponent("library"))
        let external = root.appendingPathComponent("shared-encoder")
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        try Data("local encoder".utf8).write(to: external.appendingPathComponent("config.json"))
        func check(_ condition: Bool, _ message: String) throws { if !condition { throw LibraryFailure(message: message) } }
        func checkBlobLink(_ file: URL) throws {
            let hash = try LibraryStore.sha256(file)
            let blob = store.root.appendingPathComponent("blobs/\(hash)")
            var snapshotInfo = stat(), blobInfo = stat()
            try check(lstat(file.path, &snapshotInfo) == 0 && lstat(blob.path, &blobInfo) == 0,
                      "Snapshot or content blob is missing")
            try check((snapshotInfo.st_mode & S_IFMT) == S_IFREG && snapshotInfo.st_ino == blobInfo.st_ino && snapshotInfo.st_dev == blobInfo.st_dev,
                      "Downloaded snapshot must contain regular hard links to its content blobs")
        }
        let decoded = try JSONDecoder().decode(LibraryDownloadRequest.self, from: Data("{\"modelID\":\"fixture\",\"repository\":\"test/tiny\"}".utf8))
        try check(decoded.provider == .modelscope && decoded.include.isEmpty && decoded.destinations.isEmpty, "ModelScope/layout is not the JSON default")
        guard let qwen = LibraryRecipe.all.first(where: { $0.modelID == "qwen-image-2.1" }) else {
            throw LibraryFailure(message: "Qwen recipe missing")
        }
        try check(qwen.include.contains("processor/tokenizer.json"), "Qwen default recipe excludes its required tokenizer supplement")
        try check(qwen.downloadVariants.count == 8 && qwen.downloadVariants[0].id == "bf16", "Qwen BF16/Q4–Q8 catalog incomplete")
        for variant in qwen.downloadVariants {
            let request = try variant.request(modelID: qwen.modelID)
            try check(request.provider == .modelscope && request.repository == variant.repository,
                      "Variant did not preserve provider default/source")
            for supplement in request.supplements {
                try check(supplement.include.allSatisfy(request.include.contains), "Variant excludes a required supplementary file")
            }
            if variant.id != "bf16" {
                let gguf = variant.include.filter { $0.hasSuffix(".gguf") }
                try check(gguf.count == 1 && variant.destinations[gguf[0]] == "diffusion_models/" + gguf[0], "GGUF layout/source filename changed")
                try check(variant.runnable == (variant.id == "unsloth-q4_k_m"), "Unqualified GGUF was advertised as runnable")
                try check(!variant.include.contains("diffusion_models/qwen_image_2.1_bf16.safetensors"), "GGUF preset downloads a duplicate BF16 DiT")
            }
            let encoderOnly = try variant.request(modelID: qwen.modelID, componentKind: "encoder")
            try check(Set(encoderOnly.include) == Set(["text_encoders/qwen3vl_8b_bf16.safetensors", "processor/tokenizer.json"]) && encoderOnly.destinations.isEmpty,
                      "Encoder-only preset must include its tokenizer and exclude DiT")
            let ditOnly = try variant.request(modelID: qwen.modelID, componentKind: "dit")
            try check(ditOnly.include.count == 1 && ditOnly.supplements.isEmpty, "DiT-only preset resolves unrelated supplemental repositories")
            let vaeOnly = try variant.request(modelID: qwen.modelID, componentKind: "vae")
            try check(vaeOnly.include == ["vae/qwen_image_2.1_vae_bf16.safetensors"] && vaeOnly.destinations.isEmpty &&
                      !vaeOnly.supplements.contains(where: { $0.repository == "Qwen/Qwen-Image-2.1" }), "VAE-only preset resolves tokenizer")
            let withoutTokenizer = try encoderOnly.subset(include: ["text_encoders/qwen3vl_8b_bf16.safetensors"])
            try check(!withoutTokenizer.supplements.contains(where: { $0.repository == "Qwen/Qwen-Image-2.1" }), "Deselected tokenizer still resolves its repository")
            do {
                _ = try request.subset(include: [])
                throw LibraryFailure(message: "Empty selection became whole-repository download")
            } catch let error as LibraryFailure where error.message.hasPrefix("Empty selection became") { throw error }
            catch {}
            do {
                _ = try request.subset(include: ["unrelated/model.gguf"])
                throw LibraryFailure(message: "Preset subset admitted unrelated weights")
            } catch let error as LibraryFailure where error.message.hasPrefix("Preset subset admitted") { throw error }
            catch {}
        }
        try check(LibraryRecipe.all.allSatisfy { !$0.downloadVariants.isEmpty }, "Existing model lost its default download preset")
        guard let zImage = LibraryRecipe.all.first(where: { $0.modelID == "z-image-turbo" }) else {
            throw LibraryFailure(message: "Z-Image recipe missing")
        }
        try check(zImage.downloadVariants.count == ZImageVariant.all.count + 1 && zImage.downloadVariants[0].repository == zImage.repository,
                  "Unified catalog lost original Diffusers/Z-Image precision presets")
        for variant in zImage.downloadVariants.dropFirst() {
            let encoder = try variant.request(modelID: zImage.modelID, componentKind: "encoder")
            try check(encoder.include.contains("tokenizer/tokenizer.json") && encoder.include.contains("tokenizer/tokenizer_config.json") &&
                      encoder.supplements.count == 1, "Z-Image encoder scope lost paired tokenizer source")
        }
        for recipe in LibraryRecipe.all where recipe.modelID.hasPrefix("flux2-") || recipe.modelID == "ltx-2.5-distilled" || recipe.modelID == "wan2.1-1.3b-qad" {
            let preset = recipe.downloadVariants[0]
            for kind in ["dit", "encoder", "vae"] {
                let component = try preset.request(modelID: recipe.modelID, componentKind: kind)
                try check(!component.include.isEmpty, "Existing model component scope is empty: \(recipe.modelID)/\(kind)")
            }
        }
        for provider in HubProvider.allCases {
            let client = HubClient(provider: provider, token: "fixture-token", testEndpoint: endpoint)
            let downloader = LibraryDownloader(store: store, client: client)
            let request = LibraryDownloadRequest(modelID: "fixture", repository: "test/tiny", provider: provider,
                components: ["text_encoder": LibraryComponent(path: external.path, compatibility: "fixture-encoder-v1")])
            let plan = try await downloader.plan(request)
            try check(plan.snapshot.files.count == 3 && plan.files.count == 2, "Pagination/traversal/component filtering failed")
            if provider == .huggingface { try check(plan.cachedBytes > 0, "Known SHA did not reuse ModelScope bytes") }
            let installed = try await downloader.install(request)
            let path = URL(fileURLWithPath: installed.path)
            try check(try String(contentsOf: path.appendingPathComponent("config.json"), encoding: .utf8) == "{\"fixture\":true}", "Config download changed bytes")
            try checkBlobLink(path.appendingPathComponent("config.json"))
            try checkBlobLink(path.appendingPathComponent("weights.bin"))
            try check(path.appendingPathComponent("text_encoder").resolvingSymlinksInPath().path == external.path, "External encoder was copied instead of linked")
            try check(installed.manifest != nil, "Installed snapshot has no provenance record")
        }
        let client = HubClient(provider: .huggingface, testEndpoint: endpoint)
        let downloader = LibraryDownloader(store: store, client: client)
        let slowSnapshot = try await client.snapshot(repository: "test/slow")
        guard let slowFile = slowSnapshot.files.first else { throw LibraryFailure(message: "Slow fixture is empty") }
        let progressProbe = ProgressProbe()
        let progressFile = root.appendingPathComponent("progress.bin")
        try await client.download(slowFile, repository: "test/slow", destination: progressFile) {
            progressProbe.record($0, $1)
        }
        try check(progressProbe.observed, "Download progress never reached the URLSession delegate")
        for repository in ["test/badhash", "test/badsize", "test/unsafe", "test/cycle"] {
            do {
                _ = try await downloader.install(LibraryDownloadRequest(modelID: "fixture", repository: repository, provider: .huggingface))
                throw LibraryFailure(message: "Invalid fixture unexpectedly installed: \(repository)")
            } catch let error as LibraryFailure where error.message.hasPrefix("Invalid fixture") { throw error }
            catch {}
            try check(try store.read().installations.count == 2, "Failed download published a partial installation")
        }
        let cancelled = Task {
            try await downloader.install(LibraryDownloadRequest(modelID: "fixture", repository: "test/slow", provider: .huggingface))
        }
        try await Task.sleep(for: .milliseconds(250)); cancelled.cancel()
        do { _ = try await cancelled.value; throw LibraryFailure(message: "Cancelled download completed") }
        catch is CancellationError {}
        catch let error as URLError where error.code == .cancelled {}
        try check(try store.read().installations.count == 2, "Cancelled download published an installation")
        try check(try FileManager.default.contentsOfDirectory(atPath: store.root.appendingPathComponent("staging").path).isEmpty, "Partial staging files survived cancellation")
        let combined = LibraryDownloadRequest(modelID: "fixture", repository: "test/tiny", provider: .huggingface,
            include: ["weights.bin", "extra/tokenizer.json"], supplements: [.init(repository: "test/supplement", include: ["extra/tokenizer.json"])])
        let combinedPlan = try await downloader.plan(combined)
        try check(combinedPlan.files.count == 2 && combinedPlan.additionalSnapshots?.count == 1, "Supplementary plan lost files or provenance")
        try check(combinedPlan.files.first(where: { $0.path == "extra/tokenizer.json" })?.sourceRepository == "test/supplement", "Supplement source lost")
        let combinedInstall = try await downloader.install(combined)
        try check(try String(contentsOfFile: combinedInstall.path + "/extra/tokenizer.json", encoding: .utf8) == "{\"tokenizer_fixture\":true}", "Supplement fetched from wrong source")
        let repeatPlan = try await downloader.plan(combined)
        try check(repeatPlan.downloadBytes == 0, "Supplement bytes not reusable")
        // The selection is still explicit. Removing a supplement from include
        // must not silently download it, while a complete recipe includes it.
        var omittedSupplement = combined
        omittedSupplement.include = ["weights.bin"]
        let omittedPlan = try await downloader.plan(omittedSupplement)
        try check(omittedPlan.files.count == 1 && omittedPlan.files[0].path == "weights.bin", "Deselected supplement was silently re-added")

        // A root-level upstream file can be installed into a native component
        // directory without changing its bytes, hash or download URL.
        var mapped = combined
        mapped.destinations = ["weights.bin": "diffusion_models/original.gguf"]
        mapped.supplements = [.init(repository: "test/supplement", include: ["extra/tokenizer.json"],
            destinations: ["extra/tokenizer.json": "processor/tokenizer.json"])]
        let mappedDecoded = try JSONDecoder().decode(LibraryDownloadRequest.self, from: JSONEncoder().encode(mapped))
        try check(mappedDecoded.destinations == mapped.destinations && mappedDecoded.supplements[0].destinations == mapped.supplements[0].destinations,
                  "Destination mapping lost during helper JSON roundtrip")
        let mappedPlan = try await downloader.plan(mapped)
        let mappedFile = mappedPlan.files.first { $0.path == "weights.bin" }
        try check(mappedFile?.installationPath == "diffusion_models/original.gguf" && mappedFile?.sha256 == combinedPlan.files.first(where: { $0.path == "weights.bin" })?.sha256,
                  "Destination mapping altered source identity")
        let mappedInstall = try await downloader.install(mapped)
        try check(try String(contentsOfFile: mappedInstall.path + "/processor/tokenizer.json", encoding: .utf8) == "{\"tokenizer_fixture\":true}",
                  "Mapped supplement fetched from wrong source")
        try checkBlobLink(URL(fileURLWithPath: mappedInstall.path + "/diffusion_models/original.gguf"))
        try check(!FileManager.default.fileExists(atPath: mappedInstall.path + "/weights.bin"), "Mapped source was installed twice")
        let storedPlan = try JSONDecoder().decode(LibraryDownloadPlan.self, from: Data(contentsOf: URL(fileURLWithPath: mappedInstall.manifest!)))
        try check(storedPlan.files.contains { $0.path == "extra/tokenizer.json" && $0.sourceRepository == "test/supplement" && $0.destinationPath == "processor/tokenizer.json" },
                  "Installed provenance lost original and destination paths")

        func rejectMapping(_ candidate: LibraryDownloadRequest, _ label: String) async throws {
            do {
                _ = try await downloader.plan(candidate)
                throw LibraryFailure(message: "Invalid mapping accepted: \(label)")
            } catch let error as LibraryFailure where error.message.hasPrefix("Invalid mapping accepted") { throw error }
            catch {}
        }
        for destination in ["../escape", "/absolute", "unsafe\\path", "bad\u{0000}path"] {
            var invalid = mapped; invalid.destinations = ["weights.bin": destination]
            try await rejectMapping(invalid, destination)
        }
        var missingMapping = mapped; missingMapping.destinations = ["absent.gguf": "diffusion_models/absent.gguf"]
        try await rejectMapping(missingMapping, "missing source")
        var collision = mapped; collision.destinations = ["weights.bin": "PROCESSOR/Tokenizer.JSON"]
        try await rejectMapping(collision, "case-insensitive destination collision")
        collision.destinations = ["weights.bin": "processor"]
        try await rejectMapping(collision, "file/directory collision")
        var linkedCollision = mapped
        linkedCollision.destinations = ["weights.bin": "linked"]
        linkedCollision.components = ["linked/nested": LibraryComponent(path: external.path, compatibility: "fixture")]
        try await rejectMapping(linkedCollision, "file ancestor of linked component")
        var undeclaredSupplement = mapped
        undeclaredSupplement.supplements[0].destinations = ["absent.json": "processor/tokenizer.json"]
        try await rejectMapping(undeclaredSupplement, "undeclared supplement source")
        var conflict = combined
        conflict.include = ["config.json"]
        conflict.supplements = [.init(repository: "test/tiny", include: ["config.json"])]
        do {
            _ = try await downloader.plan(conflict)
            throw LibraryFailure(message: "Source path conflict accepted")
        } catch let error as LibraryFailure { try check(error.message.contains("conflicting"), "Wrong source conflict error") }
        var ambiguousSource = conflict
        ambiguousSource.destinations = ["config.json": "primary/config.json"]
        ambiguousSource.supplements[0].destinations = ["config.json": "supplement/config.json"]
        do {
            _ = try await downloader.plan(ambiguousSource)
            throw LibraryFailure(message: "Ambiguous source path accepted despite different destinations")
        } catch let error as LibraryFailure {
            try check(error.message.contains("conflicting selected source"), "Wrong ambiguous source error")
        }
        let fluxRequest = LibraryDownloadRequest(modelID: "flux2-klein-4b", repository: "test/flux", provider: .huggingface)
        let fluxInstall = try await downloader.install(fluxRequest)
        let fluxRoot = URL(fileURLWithPath: fluxInstall.path)
        let inspection = InstallationInspection.inspect(modelID: fluxRequest.modelID, root: fluxRoot)
        try check(inspection.status == "files_present" && inspection.checkedWeightFiles == 3,
                  "Downloaded FLUX layout rejected by installation inspection: \(inspection.issues.map(\.message))")
        for component in ["transformer", "text_encoder", "vae"] {
            try checkBlobLink(fluxRoot.appendingPathComponent("\(component)/model.safetensors"))
        }
        let cachedFlux = try await downloader.plan(fluxRequest)
        try check(cachedFlux.downloadBytes == 0 && cachedFlux.cachedBytes > 0, "FLUX snapshot was not fully reusable")
        let repeatedFlux = try await downloader.install(fluxRequest)
        let repeatedRoot = URL(fileURLWithPath: repeatedFlux.path)
        try check(InstallationInspection.inspect(modelID: fluxRequest.modelID, root: repeatedRoot).status == "files_present",
                  "Cached FLUX installation rejected by inspection")
        for component in ["transformer", "text_encoder", "vae"] {
            try checkBlobLink(repeatedRoot.appendingPathComponent("\(component)/model.safetensors"))
        }
        let delegate = HubTransferDelegate()
        let response = HTTPURLResponse(url: URL(string: "https://huggingface.co/file")!, statusCode: 302, httpVersion: nil, headerFields: nil)!
        var redirected = URLRequest(url: URL(string: "https://cdn.example/file")!)
        redirected.setValue("secret", forHTTPHeaderField: "Authorization"); redirected.setValue("secret", forHTTPHeaderField: "Cookie")
        let session = URLSession(configuration: .ephemeral); defer { session.invalidateAndCancel() }
        var stripped = false
        delegate.urlSession(session, task: session.dataTask(with: response.url!), willPerformHTTPRedirection: response, newRequest: redirected) { next in
            stripped = next != nil && next?.value(forHTTPHeaderField: "Authorization") == nil && next?.value(forHTTPHeaderField: "Cookie") == nil
        }
        try check(stripped, "Credentials leaked to the CDN redirect")
        print("PASS: MS traversal, HF pinned pagination, shared-component skip, cross-provider hard-link dedup, FLUX inspection/reuse, Qwen BF16/GGUF catalog, explicit supplement selection, mapped component hardlinks/provenance/roundtrip/collisions, integrity errors, cancellation, redirect credentials")
    }
}
