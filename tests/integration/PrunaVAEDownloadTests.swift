import Foundation
import CryptoKit

private struct PrunaFixture: PinnedVAEAsset {
    let size: Int64
    let sha256: String
    let id = "pruna-fixture", title = "Pruna fixture", repository = "test/pruna"
    let revision = String(repeating: "a", count: 40)
    let sourceFile = "vae/diffusion_pytorch_model.safetensors"
    let installationModelID = "ltx-prunavaed", filename = "prunavaed.safetensors"
    let manifestFilename = "prunavaed-source.json"
    var hubFile: HubFile { HubFile(path: sourceFile, size: size, sha256: sha256, revision: revision) }
}

@main struct PrunaVAEDownloadTests {
    static func main() async throws {
        func check(_ value: Bool, _ reason: String) throws {
            if !value { throw LibraryFailure(message: reason) }
        }
        let endpoint = URL(string: CommandLine.arguments[1])!
        let root = URL(fileURLWithPath: CommandLine.arguments[2])
        let data = Data(String(repeating: "pruna-fixture-weights", count: 32).utf8)
        let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let asset = PrunaFixture(size: Int64(data.count), sha256: sha)
        let store = try LibraryStore(root: root.appendingPathComponent("good"))
        let client = HubClient(provider: .modelscope, testEndpoint: endpoint)
        let downloader = PinnedVAEDownloader(store: store, client: client)
        let item = try await downloader.install(asset)
        try check(item.modelID == "ltx-prunavaed", "decoder registered as a base model")
        try check(try Data(contentsOf: URL(fileURLWithPath: item.path).appendingPathComponent(asset.filename)) == data,
                  "download payload mismatch")
        try check(FileManager.default.fileExists(atPath: URL(fileURLWithPath: item.path).appendingPathComponent(asset.manifestFilename).path),
                  "missing source manifest")
        let repeated = try await downloader.install(asset)
        try check(repeated.id == item.id, "cache reuse created another installation")
        for (name, bad) in [("sha", PrunaFixture(size: Int64(data.count), sha256: String(repeating: "0", count: 64))),
                            ("size", PrunaFixture(size: Int64(data.count + 1), sha256: sha))] {
            let isolated = try LibraryStore(root: root.appendingPathComponent(name))
            var rejected = false
            do { _ = try await PinnedVAEDownloader(store: isolated, client: client).install(bad) }
            catch { rejected = true }
            try check(rejected && (try isolated.read()).installations.isEmpty, "unverified VAE was published")
        }
        let pin = PrunaVAEAsset.model
        try check(pin.repository == "PrunaAI/PrunaVAED" && pin.sourceFile == asset.sourceFile
                  && pin.size == 1_327_909_418
                  && pin.sha256 == "4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d",
                  "Pruna source pin changed")
        print("PASS PrunaVAED ModelScope download, integrity rejection, publication and verified reuse")
    }
}
