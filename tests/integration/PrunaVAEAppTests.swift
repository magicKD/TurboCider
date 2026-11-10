import Foundation
import CryptoKit

@main struct PrunaVAEAppTests {
    @MainActor static func main() async throws {
        // Opt-in real-model acceptance through the production App worker.
        // Usage: --generate MODEL_DIRECTORY PACKAGED_CLI REQUEST.json
        if CommandLine.arguments.count == 5, CommandLine.arguments[1] == "--generate" {
            let args = CommandLine.arguments
            let request = try JSONDecoder().decode(NativeRequest.self,
                from: Data(contentsOf: URL(fileURLWithPath: args[4])))
            guard request.ltx_video_vae == "prunavaed", LTXWorker.accepts(request) else {
                throw NativeFailure(message: "Expected a budgeted LTX PrunaVAED request")
            }
            let data = try await LTXWorker.generate(model: URL(fileURLWithPath: args[2]),
                request: request, executable: URL(fileURLWithPath: args[3]), onEvent: { event in
                    if let data = try? JSONEncoder().encode(event) {
                        FileHandle.standardError.write(data)
                        FileHandle.standardError.write(Data([10]))
                    }
                })
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data([10]))
            return
        }
        func check(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-pruna-app-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var draft = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(draft.ltxVideoVAE == "conv" && draft.ltxPrunaVAEPath.isEmpty, "old drafts changed decoder")
        draft.modelID = "ltx-2.5-distilled"; draft.operation = "video.generate"
        draft.modelPaths[draft.modelID] = root.path; draft.prompt = "A red fox"
        draft.width = 704; draft.height = 448; draft.frames = 97; draft.steps = 11; draft.fps = 24
        draft.audio = false; draft.residency = "component_staged"
        draft.ltxVideoVAE = "prunavaed"
        do {
            try draft.validateLTXVideoVAE()
            throw NativeFailure(message: "accepted missing PrunaVAED")
        } catch let error as NativeFailure {
            try check(error.message.contains("请先"), error.message)
        }
        let path = root.appendingPathComponent("prunavaed.safetensors")
        FileManager.default.createFile(atPath: path.path, contents: nil)
        let file = try FileHandle(forWritingTo: path)
        try file.truncate(atOffset: UInt64(PrunaVAEAsset.model.size)); try file.close()
        draft.ltxPrunaVAEPath = path.path
        let request = try draft.request(output: root.appendingPathComponent("video.mp4"))
        try check(request.ltx_video_vae == "prunavaed" && request.ltx_video_vae_path == path.path
                  && request.allow_approximation == true, "draft failed to bind decoder/path/approximation")
        let v2 = try JSONDecoder().decode(NativeRequestV2.self, from: JSONEncoder().encode(NativeRequestV2(legacy: request)))
        try check(v2.execution.ltx_video_vae == "prunavaed" && v2.execution.ltx_video_vae_path == path.path,
                  "v2 conversion dropped VAE")
        let restored = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
        try check(restored.ltxVideoVAE == "prunavaed" && restored.ltxPrunaVAEPath == path.path, "draft persistence lost VAE")
        let studio = StudioState(directory: root.appendingPathComponent("studio"))
        studio.reuse(NativeJob(id: UUID(), createdAt: Date(), request: request, state: "complete",
                             phase: "complete", completed: 1, total: 1, elapsed: 1, modelPath: root.path))
        try check(studio.draft.ltxVideoVAE == "prunavaed" && studio.draft.ltxPrunaVAEPath == path.path,
                  "history reuse lost VAE")
        draft.streaming.selection = .tier12
        do {
            try draft.validateLTXVideoVAE()
            throw NativeFailure(message: "accepted unqualified public streaming")
        } catch let error as NativeFailure {
            try check(error.message.contains("内存档位"), error.message)
        }
        draft.streaming.selection = .off; draft.ltxVideoVAE = "conv"
        let conv = try draft.request(output: root.appendingPathComponent("conv.mp4"))
        try check(conv.ltx_video_vae == "conv" && conv.ltx_video_vae_path == nil, "Conv inherited Pruna path")
        var imageDraft = draft
        imageDraft.modelID = "flux2-klein-4b"; imageDraft.operation = "image.generate"
        imageDraft.modelPaths[imageDraft.modelID] = root.path
        imageDraft.width = 512; imageDraft.height = 512; imageDraft.frames = 1; imageDraft.steps = 4
        imageDraft.ltxVideoVAE = "prunavaed"; imageDraft.streaming.selection = .tier12
        let imageRequest = try imageDraft.request(output: root.appendingPathComponent("image.png"))
        try check(imageRequest.ltx_video_vae == nil && imageRequest.ltx_video_vae_path == nil,
                  "LTX VAE selection leaked into image-model request")
        _ = try imageDraft.streamingQueryRequest()
        var result: [String: Any] = ["schema_version": 1, "model": request.model, "operation": "video.generate",
            "output": request.output, "width": request.width, "height": request.height,
            "steps": request.steps, "seed": request.seed, "warmup": false, "ltx_video_vae": "conv"]
        do {
            try LTXWorker.validateResult(JSONSerialization.data(withJSONObject: result), request: v2)
            throw NativeFailure(message: "worker accepted wrong decoder")
        } catch let error as NativeFailure {
            try check(error.message.contains("worker_result_mismatch"), error.message)
        }
        result["ltx_video_vae"] = "prunavaed"
        try LTXWorker.validateResult(JSONSerialization.data(withJSONObject: result), request: v2)
        let insights = RunInsights(json: String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
        try check(insights.ltxVideoVAELabel == "PrunaVAED", "run insights lost decoder")
        try check(HubClient(provider: .modelscope).endpoint.host == "modelscope.cn"
                  && PrunaVAEAsset.model.repository == "PrunaAI/PrunaVAED"
                  && PrunaVAEAsset.model.sha256.count == 64, "download source is not pinned ModelScope")
        print("PASS PrunaVAED defaults, missing files, persistence, request conversion, history, memory gate and worker correlation")
    }
}
