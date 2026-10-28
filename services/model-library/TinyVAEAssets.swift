import Foundation

enum TinyVAESource: String, CaseIterable, Identifiable {
    case huggingface, mirror
    var id: String { rawValue }
    var title: String { self == .mirror ? "Hugging Face 镜像" : "Hugging Face 官方" }
}

struct TinyVAEAsset: Sendable, Identifiable, Codable {
    let id: String
    let title: String
    let modelID: String
    let repository: String
    let revision: String
    let sourceFile: String
    let size: Int64
    let sha256: String
    var installationModelID: String { "tiny-vae-" + id.replacingOccurrences(of: "_", with: "-") }
    var filename: String { id + ".safetensors" }
    var hubFile: HubFile {
        HubFile(path: sourceFile, size: size, sha256: sha256, revision: revision)
    }
    static let all: [Self] = [
        .init(
            id: "taef1", title: "TAEF1 · Z-Image", modelID: "z-image-turbo",
            repository: "madebyollin/taef1",
            revision: "b1b2d00e9e440cfbf3dedb34266864da86016ceb",
            sourceFile: "diffusion_pytorch_model.safetensors",
            size: 9_848_636,
            sha256: "47a6c2bff850da04b267cab70fe3553fef57255eb9a8e76852baa0a87850e54d"),
        .init(
            id: "taeqi2_1", title: "TAEQI2.1 · Qwen Image 2.1", modelID: "qwen-image-2.1",
            repository: "madebyollin/taeqi2_1",
            revision: "5d62f11fee0068b4df7bd699ddab685e32174010",
            sourceFile: "taeqi2_1.safetensors",
            size: 30_591_264,
            sha256: "6578b31cc5e336233a2c842eba61b56425bbcac97c7088705f7c7db93291887e"),
    ]
}

/// Pinned public assets, no Python/conversion and no server metadata can change
/// the chosen file/architecture. Uses the same lease, verified content store,
/// atomic installation and index as the ordinary model-library downloader.
struct TinyVAEDownloader: Sendable {
    let store: LibraryStore
    let client: HubClient
    func install(
        _ asset: TinyVAEAsset,
        progress: @escaping @Sendable (LibraryDownloadEvent) -> Void = { _ in }
    ) async throws -> LibraryInstallation {
        let lease = try store.acquireLease()
        defer { withExtendedLifetime(lease) {} }
        try LibraryStore.validateIdentifier(asset.id)
        try LibraryStore.validateRelativePath(asset.sourceFile)
        try LibraryStore.validateIdentifier(asset.revision)
        @Sendable func event(_ phase: String, _ bytes: Int64, _ files: Int = 0, file: String? = nil)
        {
            progress(
                LibraryDownloadEvent(
                    phase: phase, file: file ?? asset.filename,
                    completedFiles: files, totalFiles: 1, completedBytes: bytes,
                    totalBytes: asset.size))
        }
        event("resolving", 0)
        for item in try store.read().installations
        where item.modelID == asset.installationModelID && item.revision == asset.revision {
            let path = URL(fileURLWithPath: item.path).appendingPathComponent(asset.filename)
            if FileManager.default.fileExists(atPath: path.path),
                try path.resourceValues(forKeys: [.fileSizeKey]).fileSize == Int(asset.size),
                try LibraryStore.sha256(path) == asset.sha256
            {
                try Task.checkCancellation()
                event("complete", asset.size, 1, file: item.path)
                return item
            }
        }
        let staging = try store.managedDirectory("staging").appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        let cached = try store.managedDirectory("blobs").appendingPathComponent(asset.sha256)
        let blob: URL
        if FileManager.default.fileExists(atPath: cached.path) {
            guard cached.resolvingSymlinksInPath().path == cached.path,
                try cached.resourceValues(forKeys: [.fileSizeKey]).fileSize == Int(asset.size),
                try LibraryStore.sha256(cached) == asset.sha256
            else { throw LibraryFailure(message: "Tiny VAE 缓存损坏；原文件已保留，请检查模型库。") }
            blob = cached
        } else {
            let capacity =
                try FileManager.default.attributesOfFileSystem(forPath: store.root.path)[
                    .systemFreeSize] as? NSNumber
            guard let capacity, capacity.int64Value > asset.size + (16 << 20) else {
                throw LibraryFailure(message: "没有足够空间下载 Tiny VAE。")
            }
            let temporary = staging.appendingPathComponent(".download")
            try await client.download(
                asset.hubFile, repository: asset.repository, destination: temporary
            ) { bytes, _ in
                event("downloading", min(bytes, asset.size))
            }
            try Task.checkCancellation()
            event("verifying", asset.size)
            blob = try store.commitBlob(
                temporary: temporary, expectedSize: asset.size, expectedSHA256: asset.sha256)
        }
        try Task.checkCancellation()
        try LibraryStore.linkBlob(blob, at: asset.filename, in: staging)
        let manifest = staging.appendingPathComponent("tiny-vae-source.json")
        try JSONEncoder().encode(asset).write(to: manifest, options: .withoutOverwriting)
        try Task.checkCancellation()
        let item = try store.publish(
            staging: staging, modelID: asset.installationModelID, name: asset.title,
            provider: client.endpoint.host ?? "huggingface", repository: asset.repository,
            revision: asset.revision, components: [:])
        event("complete", asset.size, 1, file: item.path)
        return item
    }
}
