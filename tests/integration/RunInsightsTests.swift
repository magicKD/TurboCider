import Foundation

@main struct RunInsightsTests {
    static func main() throws {
        func check(_ ok: Bool, _ text: String) throws { if !ok { throw NativeFailure(message: text) } }
        let empty = RunInsights(json: "{}")
        let streamed = RunInsights(json: #"{"block_residency":{"streamed_blocks":17,"request_bytes_loaded":1073741824,"request_wait_seconds":1.5}}"#)
        try check(streamed.streamedBlocks == 17 && streamed.streamReadBytes == 1073741824 && streamed.streamWaitSeconds == 1.5,
                  "Streaming measurements were lost")
        for value in ["-1", "1.5", "true", "1e100"] {
            let invalidStream = RunInsights(json: "{\"block_residency\":{\"streamed_blocks\":\(value)}}")
            try check(invalidStream.streamedBlocks == nil, "Invalid streaming layer count was accepted")
        }
        try check(empty.promptCacheHit == nil && empty.activeBytes == nil && empty.coreMLCalls == nil, "Missing metrics became zero usage")
        let gpu = RunInsights(json: #"{"prompt_cache_hit":true,"memory":{"mlx_active_bytes":1073741824},"hybrid":null}"#)
        try check(gpu.promptCacheHit == true && gpu.coreMLCalls == nil, "GPU result implied Core ML use")
        try check(RunInsights.memory(gpu.activeBytes!) == "1.00 GiB", "Memory units are wrong")
        let invalid = RunInsights(json: #"{"prompt_cache_hit":1,"memory":{"mlx_active_bytes":true},"hybrid":{"calls_session_total":-1,"bucket":1.2}}"#)
        try check(invalid.promptCacheHit == nil && invalid.activeBytes == nil && invalid.coreMLCalls == nil && invalid.coreMLRows == nil, "Invalid telemetry presented as hardware measurement")
        if CommandLine.arguments.count > 1 {
            let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
            let jobs = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
            let results = try jobs.map { try JSONSerialization.data(withJSONObject: $0["result"]!) }.map { RunInsights(json: String(decoding: $0, as: UTF8.self)) }
            try check(results.count == 3 && results[1].promptCacheHit == true, "Recorded App cache hit lost")
            try check(results.map(\.coreMLCalls) == [288,576,864] && results.map(\.coreMLRows) == [1536,1536,1056], "Per-session counters or shape changes misreported")
            try check(results.allSatisfy { $0.coreMLOutputCopyBytes == 0 }, "Zero-copy evidence misreported")
        }
        print("PASS: optional metrics, cache reporting, memory scope, session counters and recorded shape transitions")
    }
}
