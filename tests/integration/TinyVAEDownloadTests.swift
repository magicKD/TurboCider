import Foundation
import CryptoKit

private final class TinyDownloadProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes: Int64 = 0
    func record(_ event: LibraryDownloadEvent) {
        lock.lock()
        defer { lock.unlock() }
        bytes = max(bytes, event.completedBytes)
    }
    var observed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return bytes > 0
    }
}

@main struct TinyVAEDownloadTests {
    static func main() async throws {
        guard CommandLine.arguments.count == 3, let endpoint = URL(string: CommandLine.arguments[1])
        else {
            throw LibraryFailure(message: "tiny-vae-download-tests FIXTURE_ENDPOINT ROOT")
        }
        func check(_ condition: Bool, _ reason: String) throws {
            if !condition { throw LibraryFailure(message: reason) }
        }
        let store = try LibraryStore(
            root: URL(fileURLWithPath: CommandLine.arguments[2]).appendingPathComponent("library"))
        let data = Data(String(repeating: "tiny-fixture-weights", count: 32).utf8)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        func fixture(
            _ id: String = "fixture", sha: String? = nil, bytes: Int64? = nil,
            source: String = "weights.bin"
        ) -> TinyVAEAsset {
            TinyVAEAsset(
                id: id, title: "fixture", modelID: "z-image-turbo", repository: "test/tiny",
                revision: String(repeating: "a", count: 40),
                sourceFile: source, size: bytes ?? Int64(data.count), sha256: sha ?? digest)
        }
        let client = HubClient(provider: .huggingface, testEndpoint: endpoint)
        let downloader = TinyVAEDownloader(store: store, client: client),
            progress = TinyDownloadProgress()
        let first = try await downloader.install(fixture()) { progress.record($0) }
        let file = URL(fileURLWithPath: first.path).appendingPathComponent("fixture.safetensors")
        try check(
            try Data(contentsOf: file) == data && progress.observed, "download or progress missing")
        let repeatItem = try await downloader.install(fixture())
        try check(
            first.id == repeatItem.id && store.read().installations.count == 1,
            "cache reuse redownloaded or republished installation")
        let blobs = try FileManager.default.contentsOfDirectory(
            atPath: store.root.appendingPathComponent("blobs").path)
        try check(blobs == [digest], "unverified blob entered content store")
        for bad in [
            fixture("bad-hash", sha: String(repeating: "0", count: 64)),
            fixture("bad-size", bytes: Int64(data.count + 1)),
            fixture(bytes: Int64(data.count + 1)),
        ] {
            do {
                _ = try await downloader.install(bad)
                throw LibraryFailure(message: "accepted corrupted asset")
            } catch let error as LibraryFailure where error.message == "accepted corrupted asset" {
                throw error
            } catch {}
            try check(
                try store.read().installations.count == 1,
                "failed download published partial installation")
        }
        let isolated = try LibraryStore(
            root: store.root.deletingLastPathComponent().appendingPathComponent(
                "network-size-rejection"))
        do {
            _ = try await TinyVAEDownloader(store: isolated, client: client).install(
                fixture(bytes: Int64(data.count + 1)))
            throw LibraryFailure(message: "network size mismatch accepted")
        } catch let error as LibraryFailure where error.message == "network size mismatch accepted"
        { throw error } catch {}
        try check(
            try isolated.read().installations.isEmpty, "bad network size published a snapshot")
        let slow = fixture(
            "slow", sha: String(repeating: "f", count: 64), bytes: 131072, source: "slow.bin")
        let work = Task { try await downloader.install(slow) }
        try await Task.sleep(for: .milliseconds(100))
        work.cancel()
        do {
            _ = try await work.value
            throw LibraryFailure(message: "cancel was ignored")
        } catch is CancellationError {} catch let error as LibraryFailure
            where error.message == "cancel was ignored"
        { throw error } catch let error as URLError where error.code == .cancelled {}
        try check(
            try store.read().installations.count == 1, "cancelled download published installation")
        try check(
            try FileManager.default.contentsOfDirectory(
                atPath: store.root.appendingPathComponent("staging").path
            ).isEmpty,
            "cancelled/failed staging leaked")
        let recovered = try await downloader.install(fixture())
        try check(recovered.id == first.id, "retry lost verified cache")
        try check(
            TinyVAEAsset.all.count == 2
                && Set(TinyVAEAsset.all.map(\.modelID))
                    == Set(["z-image-turbo", "qwen-image-2.1"]),
            "catalog incomplete")
        try check(
            HubClient(publicHuggingFaceMirror: true).endpoint.host == "hf-mirror.com",
            "mirror endpoint missing")
        print(
            "PASS Tiny VAE public pinned download, progress, SHA/size rejection, cancel/cleanup, retry, index and cache reuse"
        )
    }
}
