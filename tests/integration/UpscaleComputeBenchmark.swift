import Foundation

/// Run each backend in a fresh process. Optional `resident` measures explicit
/// preload + warmup followed by three images using the same cached MLModel.
@main struct UpscaleBenchmark {
    static func main() async throws {
        let args = CommandLine.arguments
        guard (5...6).contains(args.count), let compute = UpscaleCompute(rawValue: args[4]),
              args.count == 5 || args[5] == "resident" else {
            throw NativeFailure(message: "Usage: upscale-benchmark MODEL INPUT OUTPUT gpu|ane [resident]")
        }
        let model = URL(fileURLWithPath: args[1]), source = URL(fileURLWithPath: args[2])
        let output = URL(fileURLWithPath: args[3]), start = ContinuousClock.now
        func seconds(_ since: ContinuousClock.Instant) -> Double {
            let d = since.duration(to: .now).components
            return Double(d.seconds) + Double(d.attoseconds) / 1e18
        }
        var report: [String: Any] = ["compute": compute.rawValue, "model": model.lastPathComponent,
            "output": output.path, "cache_scope": "fresh process; OS Core ML compilation caches not cleared",
            "ane_residency": "not measured"]
        if args.count == 6 {
            let session = UpscaleSession()
            let info = try await session.prepare(model: model, compute: compute)
            report["preload_and_warmup_seconds"] = seconds(start)
            report["scale"] = info.scale
            var timings: [Double] = []
            for _ in 0..<3 {
                let begin = ContinuousClock.now
                try await session.render(source: source, destination: output, expected: info) { _, _ in }
                timings.append(seconds(begin))
            }
            let hit = try await session.prepare(model: model, compute: compute)
            report["cache_hit"] = hit.cacheHit
            report["resident_render_seconds"] = timings
            report["resident_render_median_seconds"] = timings.sorted()[1]
            await session.release()
        } else {
            let predictor = try CoreMLUpscalePredictor(url: model, compute: compute)
            report["load_seconds"] = seconds(start)
            let loaded = ContinuousClock.now
            try ImageUpscaler.render(source: source, destination: output, predictor: predictor) { _, _ in }
            report["process_first_render_seconds"] = seconds(loaded)
            report["total_seconds"] = seconds(start)
            report["scale"] = predictor.scale
        }
        print(String(decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), as: UTF8.self))
    }
}
