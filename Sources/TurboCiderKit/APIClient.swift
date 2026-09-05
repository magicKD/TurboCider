import Foundation

public struct TurboCiderClient: Sendable {
    public let baseURL: URL
    public let bearerToken: String?
    private let session: URLSession

    public static var defaultBaseURL: URL {
        if let configured = ProcessInfo.processInfo.environment["TURBOCIDER_API_URL"],
           let url = URL(string: configured) {
            return url
        }
        return URL(string: "http://127.0.0.1:11435")!
    }

    public init(
        baseURL: URL = TurboCiderClient.defaultBaseURL,
        bearerToken: String? = ProcessInfo.processInfo.environment["TURBOCIDER_API_TOKEN"],
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.bearerToken = bearerToken
        self.session = session
    }

    public func health() async throws {
        let _: [String: String] = try await request(path: "/health")
    }

    public func models() async throws -> [TCModelDescriptor] {
        let response: TCModelsResponse = try await request(path: "/v1/models")
        return response.data
    }

    public func system() async throws -> TCSystemReport {
        try await request(path: "/v1/system")
    }

    public func plans(for requestBody: TCGenerationRequest) async throws -> [TCPlanCandidate] {
        let response: TCPlansResponse = try await request(
            path: "/v1/plans",
            method: "POST",
            body: requestBody
        )
        return response.data
    }

    public func submit(_ requestBody: TCGenerationRequest) async throws -> TCJobRecord {
        try await request(path: "/v1/jobs", method: "POST", body: requestBody)
    }

    public func job(id: String) async throws -> TCJobRecord {
        try await request(path: "/v1/jobs/\(id)")
    }

    public func jobs() async throws -> [TCJobRecord] {
        let response: TCJobsResponse = try await request(path: "/v1/jobs")
        return response.data
    }

    public func cancel(id: String) async throws -> TCJobRecord {
        try await request(path: "/v1/jobs/\(id)", method: "DELETE")
    }

    public func events(id: String) -> AsyncThrowingStream<TCJobRecord, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var streamRequest = try urlRequest(
                        path: "/v1/jobs/\(id)/events",
                        method: "GET",
                        bodyData: nil
                    )
                    streamRequest.setValue(
                        "text/event-stream",
                        forHTTPHeaderField: "Accept"
                    )
                    let (bytes, response) = try await session.bytes(for: streamRequest)
                    guard let http = response as? HTTPURLResponse else {
                        throw URLError(.badServerResponse)
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        throw URLError(.badServerResponse)
                    }
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = Data(line.dropFirst(6).utf8)
                        let job = try JSONDecoder().decode(TCJobRecord.self, from: payload)
                        continuation.yield(job)
                        if job.isTerminal {
                            continuation.finish()
                            return
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func request<Response: Decodable>(
        path: String,
        method: String = "GET"
    ) async throws -> Response {
        try await request(path: path, method: method, bodyData: nil)
    }

    private func request<Response: Decodable, Body: Encodable>(
        path: String,
        method: String,
        body: Body
    ) async throws -> Response {
        let data = try JSONEncoder().encode(body)
        return try await request(path: path, method: method, bodyData: data)
    }

    private func request<Response: Decodable>(
        path: String,
        method: String,
        bodyData: Data?
    ) async throws -> Response {
        let request = try urlRequest(
            path: path,
            method: method,
            bodyData: bodyData
        )
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            if let apiError = try? JSONDecoder().decode(TCAPIError.self, from: data) {
                throw apiError
            }
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(Response.self, from: data)
    }

    private func urlRequest(
        path: String,
        method: String,
        bodyData: Data?
    ) throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = bodyData
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if bodyData != nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        if let bearerToken {
            request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        }
        return request
    }
}
