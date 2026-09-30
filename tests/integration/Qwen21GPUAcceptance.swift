import AppKit
import Combine
import Foundation

/// Opt-in real-model acceptance through the same draft/store path as the App.
/// This verifies transport, progress and output integrity; visual edit fidelity
/// must be reviewed in the saved images and is not certified by these checks.
@main struct Qwen21GPUAcceptance {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        let quick = args.count == 8 && args[7] == "--quick"
        let loraOnly = args.count == 8 && args[7] == "--lora-only"
        guard args.count == 7 || quick || loraOnly else {
            throw NativeFailure(message: "qwen21-gpu-acceptance MODEL LORA REF1 REF2 REF3 OUTPUT [--quick | --lora-only]")
        }
        func check(_ condition: @autoclosure () throws -> Bool, _ reason: String) throws {
            if try !condition() { throw NativeFailure(message: reason) }
        }
        let model = URL(fileURLWithPath: args[1]).standardizedFileURL
        let adapter = URL(fileURLWithPath: args[2]).standardizedFileURL
        let references = args[3...5].map { URL(fileURLWithPath: $0).standardizedFileURL }
        let root = URL(fileURLWithPath: args[6]).standardizedFileURL
        try check(!FileManager.default.fileExists(atPath: root.appendingPathComponent("jobs.json").path),
                  "Acceptance output must be a fresh directory")
        let studio = StudioState(directory: root)
        let store = NativeJobStore(directory: root)
        studio.selectModel("qwen-image-2.1")
        studio.selectInstallation(modelID: "qwen-image-2.1", path: model.path)
        await studio.addFiles(references)
        try check(studio.draft.assets.count == 3, "Cannot import three acceptance references: \(studio.message ?? "")")
        let assets = studio.draft.assets
        for (asset, source) in zip(assets, references) {
            try check(try Data(contentsOf: URL(fileURLWithPath: asset.path)) == Data(contentsOf: source),
                      "Acceptance import changed or reordered reference bytes")
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var reports: [[String: Any]] = []
        var lastJob: NativeJob?
        for turbo in loraOnly ? [true] : [false, true] {
            let counts = quick ? (turbo ? [3] : [0]) : Array(0...3)
            for count in counts {
                let name = "\(turbo ? "lora6" : "base40")-refs\(count)"
                studio.draft.assets = Array(assets.prefix(count))
                studio.draft.operation = count == 0 ? "image.generate" : "image.edit"
                studio.draft.prompt = count == 0
                    ? "A ceramic teapot on a wooden table, warm sunlight, detailed photography."
                    : (count == 1
                        ? "Change the main subject in <image1> to matte red. Preserve its shape, composition and background."
                        : "Create one coherent photograph combining the main subjects of " +
                          (1...count).map { "<image\($0)>" }.joined(separator: ", ") +
                          ". Keep each subject recognizable, on a wooden table in warm sunlight.")
                studio.draft.width = 512; studio.draft.height = 512
                studio.draft.steps = turbo ? 6 : 40
                studio.draft.seedText = "42"; studio.draft.randomSeed = false
                studio.draft.residency = "component_staged"
                studio.draft.acceleration = StudioAcceleration(policy: "gpu")
                studio.draft.promptEnhance = false
                studio.draft.promptEnhanceEditExperimental = false
                studio.draft.loras = turbo ? [StudioLoRA(path: adapter.path)] : []
                if turbo { studio.applyQwen21TurboPreset() }
                let destination = root.appendingPathComponent("outputs/\(name).png")
                let request = try studio.draft.request(output: destination)
                try check(request.execution == "gpu" && request.width == 512 && request.height == 512 &&
                          request.steps == (turbo ? 6 : 40) && request.inputs?.count == count &&
                          request.inputs?.map(\.path) == Array(assets.prefix(count)).map(\.path),
                          "Invalid acceptance request for \(name)")
                try check(request.loras?.count == (turbo ? 1 : nil) &&
                          request.allow_approximation == (turbo ? true : nil),
                          "Adapter or approximation intent wrong for \(name)")
                try encoder.encode(request).write(to: root.appendingPathComponent("\(name)-request.json"), options: .withoutOverwriting)
                var states = Set<String>(), phases = Set<String>()
                var completedSteps = Set<Int>()
                let observation = store.$jobs.sink { jobs in
                    guard let job = jobs.first(where: { $0.request.output == destination.path }) else { return }
                    states.insert(job.state); phases.insert(job.phase)
                    if job.phase == "denoise" { completedSteps.insert(job.completed) }
                }
                print("START \(name): 512×512, \(request.steps) steps, GPU, \(count) ordered references")
                let job: NativeJob
                do { job = try await store.generate(modelURL: model, request: request) }
                catch { observation.cancel(); throw error }
                observation.cancel()
                lastJob = job
                try check(job.state == "succeeded" && !store.busy && states.contains("running") &&
                          states.contains("succeeded") && phases.contains("denoise") && completedSteps.contains(request.steps),
                          "App did not publish complete progress/success for \(name)")
                guard let json = job.resultJSON,
                      let native = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any],
                      let plan = native["plan"] as? [String: Any] else {
                    throw NativeFailure(message: "Missing native report for \(name)")
                }
                try Data(json.utf8).write(to: root.appendingPathComponent("\(name)-native.json"), options: .withoutOverwriting)
                try check(native["actual_denoise_steps"] as? Int == request.steps &&
                          plan["execution"] as? String == "gpu" && native["encoder_execution"] as? String == "gpu" &&
                          native["width"] as? Int == 512 && native["height"] as? Int == 512,
                          "Native route/dimensions/steps differ for \(name)")
                let expectedReferenceTokens = Array(assets.prefix(count)).reduce(0) { total, asset in
                    let ratio = Double(asset.width) / Double(asset.height)
                    let width = Int((sqrt(1024 * 1024 * ratio) / 32).rounded(.toNearestOrEven)) * 32
                    let height = Int((sqrt(1024 * 1024 / ratio) / 32).rounded(.toNearestOrEven)) * 32
                    return total + (width / 16) * (height / 16)
                }
                try check(native["reference_tokens"] as? Int == expectedReferenceTokens,
                          "Reference encoder did not consume all \(count) images for \(name)")
                try check((native["lora_applied_projections"] as? Int ?? 0) == (turbo ? 227 : 0),
                          "Wrong adapter projection coverage for \(name)")
                guard let bitmap = NSBitmapImageRep(data: try Data(contentsOf: destination)) else {
                    throw NativeFailure(message: "Undecodable output for \(name)")
                }
                try check(bitmap.pixelsWide == 512 && bitmap.pixelsHigh == 512 && bitmap.hasAlpha,
                          "PNG did not preserve RGBA canvas for \(name)")
                studio.save()
                let restored = StudioState(directory: root)
                restored.reuse(job)
                let reused = try restored.draft.request(output: root.appendingPathComponent("unused-reuse.png"))
                try check(reused.execution == request.execution && reused.steps == request.steps &&
                          reused.seed == request.seed && reused.prompt == request.prompt &&
                          reused.inputs?.map(\.path) == request.inputs?.map(\.path) &&
                          reused.loras?.map(\.path) == request.loras?.map(\.path) &&
                          reused.allow_approximation == request.allow_approximation,
                          "Persistence/reuse changed \(name)")
                reports.append(["case": name, "output": destination.path, "app_elapsed_seconds": job.elapsed,
                                "seconds_per_step": job.secondsPerStep.map { $0 as Any } ?? NSNull(),
                                "reference_tokens": expectedReferenceTokens, "native": native,
                                "states": states.sorted(), "phases": phases.sorted(),
                                "denoise_completed": completedSteps.sorted()])
                print("PASS \(name): App output, progress, native GPU receipt, adapter/reference coverage, persistence and reuse")
            }
        }
        try await store.unload()
        try check(store.loadedPath == nil && !store.busy, "Acceptance left the model loaded")
        try check(NativeJobStore(directory: root).jobs.count == reports.count &&
                  NativeJobStore(directory: root).jobs.allSatisfy { $0.state == "succeeded" },
                  "Acceptance history did not survive reopening")
        if let lastJob { studio.reuse(lastJob); studio.save() }
        let report: [String: Any] = [
            "scope": "Real App transport/lifecycle acceptance; saved outputs require visual review for generation and editing fidelity",
            "mode": loraOnly ? "lora_zero_through_three_references" :
                (quick ? "quick_base_text_and_lora_three_references" : "base_and_lora_zero_through_three_references"),
            "model": model.path, "adapter": adapter.path, "cases": reports,
            "all_transport_checks_passed": true, "visual_quality_accepted": false
        ]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("acceptance-report.json"), options: .withoutOverwriting)
        print("PASS Qwen 2.1 GPU App acceptance: \(reports.count) jobs; visual quality still requires review")
    }
}
