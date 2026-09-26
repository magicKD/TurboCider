import AppKit
import Foundation

@main struct Qwen21AppTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        let peEdit = args.count == 7 && args[4] == "--pe-i2i"
        guard args.count == 4 || args.count == 5 || peEdit else { throw NativeFailure(message: "qwen21-app-tests MODEL REFERENCE.png OUTPUT_DIRECTORY [COMPILED_ANE_MANIFEST | --pe-i2i PE_INSTALL MASK.png]") }
        let root = URL(fileURLWithPath: args[3]).standardizedFileURL
        let model = URL(fileURLWithPath: args[1]).standardizedFileURL
        let studio = StudioState(directory: root)
        let store = NativeJobStore(directory: root)
        func check(_ value: Bool, _ reason: String) throws {
            if !value { throw NativeFailure(message: reason) }
        }
        studio.selectModel("qwen-image-2.1")
        try check(studio.draft.modelID == "qwen-image-2.1" && studio.draft.steps == 40 && !studio.draft.audio,
                  "Qwen21 discovery/defaults failed")
        try check(studio.draft.residency == "component_staged", "Qwen21 did not select staged residency")
        studio.selectInstallation(modelID: "qwen-image-2.1", path: model.path)
        if peEdit {
            // Full experimental inference, deliberately separate from one-step smoke.
            try check(store.jobs.isEmpty && studio.draft.assets.isEmpty,
                      "PE-I2I test requires a fresh output directory")
            studio.changeOperation("image.edit")
            let originals = [URL(fileURLWithPath: args[2]), URL(fileURLWithPath: args[6])]
            await studio.addFiles(originals)
            try check(studio.draft.assets.count == 2, "App did not import both source and mask: \(studio.message ?? "")")
            // Import must retain the files' identities, not just their count.
            for (asset, original) in zip(studio.draft.assets, originals) {
                try check(try Data(contentsOf: URL(fileURLWithPath: asset.path)) == Data(contentsOf: original),
                          "App changed or reordered imported source/mask bytes")
            }
            studio.draft.prompt = "Use <image2> as a spatial mask for <image1>: white marks the area to edit and black marks the area to preserve. Change only the teapot lid and its knob inside the white region to matte red. Preserve the original ceramic color of the teapot body, handle and spout. Keep the wooden table, lighting and composition unchanged. Do not include the mask in the output."
            studio.draft.width = 1024; studio.draft.height = 1024
            studio.draft.steps = 40; studio.draft.randomSeed = false; studio.draft.seedText = "42"
            studio.draft.residency = "component_staged"
            studio.draft.acceleration = StudioAcceleration(policy: "gpu")
            studio.draft.promptEnhance = true
            studio.draft.promptEnhanceEditExperimental = true
            studio.draft.promptEnhancerPath = URL(fileURLWithPath: args[5]).standardizedFileURL.path
            let output = root.appendingPathComponent("pe-i2i-mask.png")
            let request = try studio.draft.request(output: output)
            try check(request.prompt_enhance == true && request.prompt_enhance_edit_experimental == true,
                      "App lost explicit experimental PE opt-in")
            try check(request.inputs?.map(\.path) == studio.draft.assets.map(\.path) &&
                      request.inputs?.allSatisfy { $0.role == "reference" && $0.kind == "image" } == true,
                      "App lost source/mask reference order")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(request).write(to: root.appendingPathComponent("request.json"), options: .atomic)
            studio.save()
            let restored = StudioState(directory: root)
            try check(restored.draft.promptEnhanceEditExperimental && restored.draft.promptEnhance &&
                      restored.draft.assets.map(\.path) == studio.draft.assets.map(\.path),
                      "App did not persist experimental opt-in and reference order")
            let job = try await store.generate(modelURL: model, request: request)
            try check(job.state == "succeeded", "App PE-I2I mask generation failed")
            guard let json = job.resultJSON else { throw NativeFailure(message: "Missing PE-I2I result") }
            try Data(json.utf8).write(to: root.appendingPathComponent("native-report.json"), options: .atomic)
            let report = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
            let pe = report["prompt_enhancement"] as? [String: Any] ?? [:]
            try check(pe["backend"] as? String == "native_qwen35_pe_i2i_experimental" &&
                      pe["complete"] as? Bool == true && pe["visual_precision"] as? String == "float32" &&
                      pe["experimental_edit"] as? Bool == true && pe["quality_accepted"] as? Bool == false,
                      "Incomplete or mislabeled experimental PE rewrite")
            try check(report["actual_denoise_steps"] as? Int == 40 && report["reference_tokens"] as? Int == 8192,
                      "App did not finish 40 steps with two references")
            guard let bitmap = NSBitmapImageRep(data: try Data(contentsOf: output)) else {
                throw NativeFailure(message: "Invalid PE-I2I PNG")
            }
            try check(bitmap.pixelsWide == 1024 && bitmap.pixelsHigh == 1024 && bitmap.hasAlpha,
                      "App PE-I2I lost canvas or alpha")
            restored.reuse(job)
            try check(restored.draft.promptEnhanceEditExperimental && restored.draft.promptEnhance &&
                      restored.draft.assets.map(\.path) == request.inputs?.map(\.path),
                      "Job reuse lost experimental opt-in or reference order")
            try await store.unload()
            print("PASS full experimental App PE-I2I job-store inference, source/mask ordering, 40 steps, RGBA, persistence and reuse; visual mask is not hard inpainting and quality remains unaccepted")
            return
        }
        studio.draft.prompt = "A ceramic teapot on a wooden table, warm sunlight, detailed photography."
        studio.draft.width = 512; studio.draft.height = 512
        // Smoke test only: do not use this one-step artifact as quality evidence.
        studio.draft.steps = 1; studio.draft.randomSeed = false; studio.draft.seedText = "42"
        let generated = root.appendingPathComponent("smoke-t2i.png")
        let request = try studio.draft.request(output: generated)
        try check(request.model == "qwen-image-2.1", "App request lost model identity")
        try await store.prepare(modelURL: model, request: request, warmup: false)
        try check(!FileManager.default.fileExists(atPath: generated.path), "Preparation wrote an image")
        let first = try await store.generate(modelURL: model, request: request)
        try check(first.state == "succeeded", "Qwen21 App text-to-image smoke failed")
        let firstResult = try JSONSerialization.jsonObject(with: Data(first.resultJSON!.utf8)) as! [String: Any]
        try check(firstResult["prompt_cache_hit"] as? Bool == true, "Prepared Qwen21 text cache was not reused")
        guard let bitmap = NSBitmapImageRep(data: try Data(contentsOf: generated)) else { throw NativeFailure(message: "Invalid Qwen21 App PNG") }
        try check(bitmap.pixelsWide == 512 && bitmap.pixelsHigh == 512 && bitmap.hasAlpha, "Qwen21 App lost dimensions or alpha")
        studio.changeOperation("image.edit")
        await studio.addFiles([URL(fileURLWithPath: args[2])])
        studio.draft.prompt = "Change the teapot color to matte red. Keep everything else unchanged."
        let editRequest = try studio.draft.request(output: root.appendingPathComponent("smoke-edit.png"))
        try check(editRequest.inputs?.count == 1 && editRequest.inputs?.first?.role == "reference", "App did not forward Qwen21 reference")
        let edit = try await store.generate(modelURL: model, request: editRequest)
        try check(edit.state == "succeeded", "Qwen21 App edit transport smoke failed")
        let editResult = try JSONSerialization.jsonObject(with: Data(edit.resultJSON!.utf8)) as! [String: Any]
        try check(editResult["reference_tokens"] as? Int == 4096,
                  "Qwen21 reference did not use the official default 1024-square preprocessing area")
        var hybridResults: [[String: Any]] = []
        if args.count == 5 {
            studio.changeOperation("image.generate")
            studio.draft.width = 512; studio.draft.height = 512; studio.draft.steps = 2
            studio.draft.residency = "resident"
            studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: args[4])
            studio.draft.prompt = "A ceramic teapot on a wooden table, warm sunlight, detailed photography."
            var previousCalls = 0
            for iteration in 0..<2 {
                let hybridRequest = try studio.draft.request(output: root.appendingPathComponent("smoke-hybrid-\(iteration).png"))
                try check(hybridRequest.execution == "gpu_ane" && hybridRequest.allow_approximation == true,
                          "App did not opt into the experimental Qwen21 hybrid route")
                let job = try await store.generate(modelURL: model, request: hybridRequest)
                try check(job.state == "succeeded", "Qwen21 App hybrid smoke failed")
                try check(job.routeSummary?.contains("BF16/FP16") == true,
                          "App route summary mislabeled Qwen21 FP16 as INT8")
                let report = try JSONSerialization.jsonObject(with: Data(job.resultJSON!.utf8)) as! [String: Any]
                let metrics = report["hybrid"] as! [String: Any]
                let calls = metrics["runtime_calls_session_total"] as! Int
                try check(calls - previousCalls == 32, "Qwen21 hybrid did not run exactly one decode across 32 layers")
                try check(report["runtime_precision"] as? String == "bf16_gpu+fp16_mlp_fp16_io",
                          "Qwen21 FP16 hybrid report mislabeled precision")
                if iteration == 1 {
                    try check(report["prompt_cache_hit"] as? Bool == true, "App hybrid resident cache missed")
                }
                previousCalls = calls
                hybridResults.append(report)
            }
        }
        studio.save()
        let restored = StudioState(directory: root)
        try check(restored.draft.modelID == "qwen-image-2.1" && restored.draft.modelPath == model.path,
                  "Qwen21 model selection was not persisted")
        try await store.unload()
        let report: [String: Any] = ["scope": "App transport/lifecycle smoke only; one step is not a quality test",
                                    "text_to_image": firstResult,
                                    "edit": editResult, "hybrid": hybridResults]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print("PASS Qwen21 App discovery, preparation, cached text generation, RGBA PNG, reference forwarding, edit transport, persistence and unload (not quality)")
    }
}
