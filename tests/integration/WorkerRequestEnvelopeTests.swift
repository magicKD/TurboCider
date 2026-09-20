import Foundation

@main struct WorkerRequestEnvelopeTests {
    static func main() throws {
        let input = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let output = try WorkerRequestEnvelope.encode(
            jobID: UUID(uuidString: "00000000-0000-4000-8000-000000000001")!,
            requestID: UUID(uuidString: "00000000-0000-4000-8000-000000000002")!,
            installation: URL(fileURLWithPath: "/tmp/model"), nativeRequest: input)
        let decoded = try JSONSerialization.jsonObject(with: output) as! [String: Any]
        precondition(decoded["request_digest"] as? String == "802fd8d904d4e5421f73382626dc03c5bb20b433c16aa82c9816bb559b927292")
        try output.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
        print("Swift worker envelope golden PASS")
    }
}
