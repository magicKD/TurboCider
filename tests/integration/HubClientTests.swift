import Foundation

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
        let decoded = try JSONDecoder().decode(LibraryDownloadRequest.self, from: Data("{\"modelID\":\"fixture\",\"repository\":\"test/tiny\"}".utf8))
        try check(decoded.provider == .modelscope && decoded.include.isEmpty, "ModelScope is not the JSON default")
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
            try check(path.appendingPathComponent("text_encoder").resolvingSymlinksInPath().path == external.path, "External encoder was copied instead of linked")
            try check(installed.manifest != nil, "Installed snapshot has no provenance record")
        }
        let client = HubClient(provider: .huggingface, testEndpoint: endpoint)
        let downloader = LibraryDownloader(store: store, client: client)
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
        var conflict = combined
        conflict.include = ["config.json"]
        conflict.supplements = [.init(repository: "test/tiny", include: ["config.json"])]
        do {
            _ = try await downloader.plan(conflict)
            throw LibraryFailure(message: "Source path conflict accepted")
        } catch let error as LibraryFailure { try check(error.message.contains("conflicting"), "Wrong source conflict error") }
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
        print("PASS: MS traversal, HF pinned pagination, shared-component skip, cross-provider dedup, multi-repository provenance/reuse/conflict, integrity errors, path rejection, cancellation, redirect credentials")
    }
}
