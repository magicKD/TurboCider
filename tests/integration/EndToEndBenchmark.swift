import Foundation
import CryptoKit
import Darwin

/// Measures the actual App job path: session creation, events, PNG export and history persistence.
/// Run the identical executable beside baseline/candidate dylibs in separate fresh processes.
@main
struct EndToEndBenchmark {
    @MainActor static func main() async throws {
        guard [5, 6].contains(CommandLine.arguments.count) else {
            throw NativeFailure(message: "benchmark MODEL OUTPUT MODE MANIFEST")
        }
        let model = URL(fileURLWithPath: CommandLine.arguments[1])
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        let mode = CommandLine.arguments[3]
        let store = NativeJobStore(directory: output)
        var runs: [[String: Any]] = []
        let interactive = CommandLine.arguments.count == 6 && CommandLine.arguments[5] == "--interactive"
        var index = 0
        while true {
            if interactive {
                guard let command = readLine(), command == "run" else { break }
            } else if index == 6 { break }
            var request = NativeRequest(
                prompt: "A red fox sitting in a snowy forest, soft morning light, detailed photography.",
                output: output.appendingPathComponent("\(index).png").path)
            request.execution = mode
            if mode == "gpu_ane" {
                request.allow_approximation = true
                request.ane_manifest = CommandLine.arguments[4]
            }
            let start = DispatchTime.now().uptimeNanoseconds
            let job = try await store.generate(modelURL: model, request: request)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            guard job.state == "succeeded", let result = job.resultJSON else {
                throw NativeFailure(message: "generation failed")
            }
            let image = try Data(contentsOf: URL(fileURLWithPath: request.output))
            let digest = SHA256.hash(data: image).map { String(format: "%02x", $0) }.joined()
            runs.append(["wall": wall, "png_sha256": digest,
                         "metrics": try JSONSerialization.jsonObject(with: Data(result.utf8))])
            if interactive {
                print("RUN \(wall)")
                fflush(stdout)
            }
            index += 1
        }
        guard runs.count >= 2 else { throw NativeFailure(message: "At least two runs required") }
        try await store.unload()
        let warm = runs.dropFirst().map { $0["wall"] as! Double }.sorted()
        let report: [String: Any] = ["mode": mode, "scope": "App NativeJobStore including events and persistence; excludes window rendering",
                                    "cold_e2e_seconds": runs[0]["wall"]!,
                                    "warm_median_seconds": warm[warm.count / 2], "runs": runs]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("report.json"))
        print("\(mode): App warm median \(warm[warm.count / 2]) seconds")
    }
}
