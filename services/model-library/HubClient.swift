import Foundation
import CryptoKit

enum HubProvider: String, Codable, Sendable, CaseIterable {
    case modelscope, huggingface
    var endpoint: URL { URL(string: self == .modelscope ? "https://modelscope.cn" : "https://huggingface.co")! }
    var defaultRevision: String { self == .modelscope ? "master" : "main" }
}

struct HubFile: Codable, Sendable {
    var path: String
    var size: Int64
    var sha256: String?
    var gitOID: String?
    var revision: String
    /// Set by a multi-repository installation plan, never accepted from a hub.
    var sourceRepository: String? = nil
}

struct HubSnapshot: Codable, Sendable {
    var provider: HubProvider
    var repository: String
    var revision: String
    var files: [HubFile]
    var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }
}

/// Never forward hub credentials to a CDN on a cross-origin redirect.
final class HubTransferDelegate: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
    let progress: @Sendable (Int64, Int64) -> Void
    init(progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) { self.progress = progress }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        let localTest = response.url?.scheme == "http" && ["127.0.0.1", "localhost"].contains(response.url?.host ?? "") &&
            request.url?.host == response.url?.host && request.url?.port == response.url?.port
        guard request.url?.scheme == "https" || localTest else {
            completionHandler(nil); return
        }
        var next = request
        if response.url?.host != request.url?.host || response.url?.port != request.url?.port || response.url?.scheme != request.url?.scheme {
            next.setValue(nil, forHTTPHeaderField: "Authorization")
            next.setValue(nil, forHTTPHeaderField: "Cookie")
        }
        completionHandler(next)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        progress(totalBytesWritten, totalBytesExpectedToWrite)
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}

struct HubClient: Sendable {
    let provider: HubProvider
    let endpoint: URL
    private let token: String?

    /// A custom endpoint is injectable only in tests; shipping requests select a provider.
    init(provider: HubProvider = .modelscope, token: String? = nil, testEndpoint: URL? = nil) {
        self.provider = provider; endpoint = testEndpoint ?? provider.endpoint; self.token = token
    }

    private func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 24 * 3600
        config.httpCookieStorage = nil; config.urlCache = nil
        return URLSession(configuration: config, delegate: HubTransferDelegate(), delegateQueue: nil)
    }

    private func request(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("TurboCider/0.2 ModelLibrary", forHTTPHeaderField: "User-Agent")
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            if provider == .modelscope { request.setValue("m_session_id=\(token)", forHTTPHeaderField: "Cookie") }
        }
        return request
    }

    private func url(_ path: String, query: [String: String] = [:]) -> URL {
        var components = URLComponents(url: endpoint.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { components.queryItems = query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) } }
        return components.url!
    }

    static func validateRepository(_ repository: String) throws {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { throw LibraryFailure(message: "Repository must be owner/name.") }
        for part in parts { try LibraryStore.validateIdentifier(String(part)) }
    }

    private func check(_ response: URLResponse) throws -> HTTPURLResponse {
        guard let http = response as? HTTPURLResponse else { throw LibraryFailure(message: "Hub returned a non-HTTP response.") }
        guard http.statusCode == 200 else {
            let hint = [401, 403].contains(http.statusCode) ? " Accept the model license on the provider site and configure an access token if required." : " Check repository, revision and network access."
            throw LibraryFailure(message: "\(provider.rawValue) HTTP \(http.statusCode)." + hint)
        }
        return http
    }

    private func json(_ url: URL, session: URLSession) async throws -> (Any, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request(url))
        let http = try check(response)
        guard data.count < 32 * 1024 * 1024 else { throw LibraryFailure(message: "Hub metadata response is too large.") }
        return (try JSONSerialization.jsonObject(with: data), http)
    }

    func snapshot(repository: String, revision: String? = nil) async throws -> HubSnapshot {
        try Self.validateRepository(repository)
        let revision = revision ?? provider.defaultRevision
        try LibraryStore.validateIdentifier(revision)
        let session = session(); defer { session.invalidateAndCancel() }
        var files: [HubFile] = []
        var resolved = revision
        if provider == .huggingface {
            let (rawInfo, _) = try await json(url("api/models/\(repository)/revision/\(revision)"), session: session)
            guard let info = rawInfo as? [String: Any], let sha = info["sha"] as? String else {
                throw LibraryFailure(message: "Hugging Face did not return a resolved revision.")
            }
            try LibraryStore.validateIdentifier(sha); resolved = sha
            var next: URL? = url("api/models/\(repository)/tree/\(sha)", query: ["recursive": "true"])
            var visited: Set<URL> = []
            while let page = next {
                try Task.checkCancellation()
                guard visited.insert(page).inserted, visited.count <= 1000 else { throw LibraryFailure(message: "Hub pagination is cyclic or too large.") }
                let (raw, response) = try await json(page, session: session)
                guard let entries = raw as? [[String: Any]] else { throw LibraryFailure(message: "Invalid Hugging Face file tree.") }
                for entry in entries where entry["type"] as? String == "file" {
                    guard let path = entry["path"] as? String, let size = entry["size"] as? NSNumber else { throw LibraryFailure(message: "Incomplete file metadata.") }
                    let lfs = entry["lfs"] as? [String: Any]
                    files.append(HubFile(path: path, size: size.int64Value, sha256: lfs?["oid"] as? String,
                                         gitOID: lfs == nil ? entry["oid"] as? String : nil, revision: resolved))
                }
                next = try nextPage(response.value(forHTTPHeaderField: "Link"))
            }
        } else {
            // Shallow traversal avoids silently truncated recursive MS listings.
            var pending = [""]; var seen: Set<String> = []
            while let directory = pending.popLast() {
                try Task.checkCancellation()
                guard seen.insert(directory).inserted, seen.count <= 1000 else { throw LibraryFailure(message: "ModelScope directory tree is cyclic or too large.") }
                var query = ["Revision": revision, "Recursive": "False"]
                if !directory.isEmpty { query["Root"] = directory }
                let (raw, _) = try await json(url("api/v1/models/\(repository)/repo/files", query: query), session: session)
                guard let envelope = raw as? [String: Any], envelope["Success"] as? Bool != false else { throw LibraryFailure(message: "ModelScope could not list the repository.") }
                let data = envelope["Data"]
                let entries = (data as? [[String: Any]]) ?? (data as? [String: Any])?["Files"] as? [[String: Any]]
                guard let entries, entries.count < 3000 else { throw LibraryFailure(message: "ModelScope directory listing may be truncated; choose a smaller repository.") }
                for entry in entries {
                    guard let path = entry["Path"] as? String else { throw LibraryFailure(message: "ModelScope file path is missing.") }
                    try LibraryStore.validateRelativePath(path)
                    guard directory.isEmpty || path.hasPrefix(directory + "/") else { throw LibraryFailure(message: "ModelScope listing escaped its requested directory.") }
                    if entry["Type"] as? String == "tree" { pending.append(path); continue }
                    guard let size = entry["Size"] as? NSNumber else { throw LibraryFailure(message: "ModelScope file size is missing.") }
                    files.append(HubFile(path: path, size: size.int64Value, sha256: entry["Sha256"] as? String,
                                         revision: entry["Revision"] as? String ?? revision))
                }
            }
        }
        var names: Set<String> = []; var total: Int64 = 0
        for file in files {
            try LibraryStore.validateRelativePath(file.path)
            try LibraryStore.validateIdentifier(file.revision)
            guard names.insert(file.path.lowercased()).inserted, file.size >= 0,
                  file.size <= Int64.max - total else { throw LibraryFailure(message: "Ambiguous filenames or invalid file sizes in hub metadata.") }
            if let hash = file.sha256, hash.range(of: "^[0-9a-fA-F]{64}$", options: .regularExpression) == nil {
                throw LibraryFailure(message: "Invalid upstream SHA-256.")
            }
            total += file.size
        }
        guard !files.isEmpty else { throw LibraryFailure(message: "Repository contains no files.") }
        return HubSnapshot(provider: provider, repository: repository, revision: resolved, files: files.sorted { $0.path < $1.path })
    }

    private func nextPage(_ header: String?) throws -> URL? {
        guard let header else { return nil }
        for section in header.components(separatedBy: ",") where section.contains("rel=\"next\"") {
            guard let start = section.firstIndex(of: "<"), let end = section.firstIndex(of: ">"), start < end,
                  let next = URL(string: String(section[section.index(after: start)..<end]), relativeTo: endpoint)?.absoluteURL,
                  next.host == endpoint.host, next.scheme == endpoint.scheme, next.port == endpoint.port,
                  next.path.hasPrefix("/api/models/") else { throw LibraryFailure(message: "Untrusted hub pagination URL.") }
            return next
        }
        return nil
    }

    func download(_ file: HubFile, repository: String, destination: URL,
                  progress: @escaping @Sendable (Int64, Int64) -> Void = { _, _ in }) async throws {
        try Self.validateRepository(repository); try LibraryStore.validateRelativePath(file.path)
        try LibraryStore.validateIdentifier(file.revision)
        let source = provider == .huggingface ? url("\(repository)/resolve/\(file.revision)/\(file.path)") :
            url("api/v1/models/\(repository)/repo", query: ["Revision": file.revision, "FilePath": file.path])
        let session = session(); defer { session.invalidateAndCancel() }
        let (temporary, response) = try await session.download(for: request(source), delegate: HubTransferDelegate(progress: progress))
        defer { try? FileManager.default.removeItem(at: temporary) }
        _ = try check(response)
        try Task.checkCancellation()
        let size = try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
        guard size == file.size else { throw LibraryFailure(message: "Downloaded file is incomplete: \(file.path)") }
        if let oid = file.gitOID {
            guard oid.range(of: "^[0-9a-fA-F]{40}$", options: .regularExpression) != nil else { throw LibraryFailure(message: "Invalid Git blob identity.") }
            var hash = Insecure.SHA1(); hash.update(data: Data("blob \(size)\0".utf8))
            let input = try FileHandle(forReadingFrom: temporary); defer { try? input.close() }
            while let data = try input.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
            guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == oid.lowercased() else {
                throw LibraryFailure(message: "Git blob verification failed: \(file.path)")
            }
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
    }
}
