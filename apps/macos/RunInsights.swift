import Foundation
import CoreFoundation

/// Snapshot metrics retain their native scope; absence is never zero usage.
struct RunInsights {
    let promptCacheHit: Bool?
    let editPrefixCacheHit: Bool?
    let textSeconds: Double?
    let denoiseSeconds: Double?
    let vaeEncodeSeconds: Double?
    let vaeDecodeSeconds: Double?
    let isSeedVR2: Bool
    let isLTX: Bool
    let ltxComputeLabel: String?
    let ltxVideoVAELabel: String?
    let encoderLoadSeconds: Double?
    let connectorSeconds: Double?
    let stage1Seconds: Double?
    let stage2Seconds: Double?
    let latentUpsampleSeconds: Double?
    let stage1StepSeconds: [Double]?
    let stage2StepSeconds: [Double]?
    let audioDecodeSeconds: Double?
    let audioVAEDecodeSeconds: Double?
    let audioVocoderSeconds: Double?
    let audioBWESeconds: Double?
    let audioMuxSeconds: Double?
    let activeBytes: Double?
    let peakBytes: Double?
    let coreMLCalls: Int?
    let coreMLBlocks: Int?
    let coreMLRows: Int?
    let coreMLLoadSeconds: Double?
    let coreMLPredictionSeconds: Double?
    let coreMLOutputCopyBytes: Double?
    let computeUnits: String?
    let streamedBlocks: Int?
    let streamReadBytes: Double?
    let streamWaitSeconds: Double?
    let streamResidentWeightBytes: Double?
    let streamManagedWeightBytes: Double?
    let streamResidentLayers: Int?
    let streamWeightBudgetBytes: Double?
    let streamCacheLimitedByMemory: Bool?
    let streamCacheEvictions: Int?
    let qwen21DiTCacheMode: String?
    let qwen21DiTCacheThreshold: Double?
    let qwen21DiTCacheMaxConsecutive: Int?
    let qwen21DiTCacheSteps: Int?
    let qwen21DiTCacheSavedBlocks: Int?
    init(json: String) {
        let raw = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any] ?? [:]
        let memory = raw["memory"] as? [String: Any] ?? raw
        let hybrid = raw["hybrid"] as? [String: Any] ?? [:]
        let times = raw["timings_seconds"] as? [String: Any] ?? [:]
        func number(_ dict: [String: Any], _ key: String) -> Double? {
            guard let n = dict[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite, n.doubleValue >= 0 else { return nil }
            return n.doubleValue
        }
        func count(_ dict: [String: Any], _ key: String) -> Int? {
            guard let n = number(dict, key), n < Double(Int.max), n.rounded(.down) == n else { return nil }
            return Int(n)
        }
        let isLtxModel = raw["model"] as? String == "ltx-2.5-distilled"
        isLTX = isLtxModel
        switch isLtxModel ? raw["ltx_video_vae"] as? String : nil {
        case "prunavaed": ltxVideoVAELabel = "PrunaVAED"
        case "conv": ltxVideoVAELabel = "Conv VAE"
        default: ltxVideoVAELabel = nil
        }
        if let hit = raw["prompt_cache_hit"] as? NSNumber, CFGetTypeID(hit) == CFBooleanGetTypeID() { promptCacheHit = hit.boolValue }
        else if isLtxModel, let hit = raw["conditioning_cache_hit"] as? NSNumber, CFGetTypeID(hit) == CFBooleanGetTypeID() { promptCacheHit = hit.boolValue }
        else { promptCacheHit = nil }
        let selections = (raw["acceleration_selection"] as? String ?? "")
            .split(separator: ";").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        if raw["model"] as? String == "qwen-image-2.1", raw["operation"] as? String == "image.edit" {
            editPrefixCacheHit = selections.contains("edit prefix KV snapshot hit") ? true
                : selections.contains("edit prefix KV snapshot miss") ? false : nil
        } else { editPrefixCacheHit = nil }
        textSeconds = number(times, "text_encode"); denoiseSeconds = number(times, "denoise")
        vaeEncodeSeconds = number(times,"vae_encode")
        vaeDecodeSeconds = number(times,"vae_decode") ?? (isLtxModel ? number(times,"video_vae_decode") : nil)
        isSeedVR2 = raw["model"] as? String == "seedvr2-3b"
        encoderLoadSeconds = isLtxModel ? number(times,"text_encoder_load") : nil
        connectorSeconds = isLtxModel ? number(times,"connector") : nil
        stage1Seconds = isLtxModel ? number(times,"stage1") : nil
        stage2Seconds = isLtxModel ? number(times,"stage2") : nil
        latentUpsampleSeconds = isLtxModel ? number(times,"latent_upsample") : nil
        switch isLtxModel ? raw["convrot_compute_recipe"] as? String : nil {
        case "original-i8-f32-scale-w8a8-rne127-bf16-io-v1": ltxComputeLabel = "INT8 ConvRot · W8A8"
        case "convrot-w8-bf16-activations-v1": ltxComputeLabel = "INT8 ConvRot · BF16 激活"
        default: ltxComputeLabel = nil
        }
        let steps = raw["denoise_step_seconds"] as? [String:Any] ?? [:]
        func durations(_ key: String) -> [Double]? {
            guard isLtxModel, let values = steps[key] as? [Any], !values.isEmpty else { return nil }
            let parsed = values.compactMap { value -> Double? in
                guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
                      value.doubleValue.isFinite, value.doubleValue >= 0 else { return nil }
                return value.doubleValue
            }
            guard parsed.count == values.count, parsed.reduce(0,+).isFinite else { return nil }
            return parsed
        }
        stage1StepSeconds = durations("stage1"); stage2StepSeconds = durations("stage2")
        let hasLTXAudio = isLtxModel && (raw["audio"] as? NSNumber).map {
            CFGetTypeID($0) == CFBooleanGetTypeID() && $0.boolValue
        } == true
        audioDecodeSeconds = hasLTXAudio ? number(times,"audio_decode") : nil
        audioVAEDecodeSeconds = hasLTXAudio ? number(times,"audio_vae_decode") : nil
        audioVocoderSeconds = hasLTXAudio ? number(times,"audio_vocoder") : nil
        audioBWESeconds = hasLTXAudio ? number(times,"audio_bwe") : nil
        audioMuxSeconds = hasLTXAudio ? number(times,"audio_mux") : nil
        activeBytes = number(memory, "mlx_active_bytes"); peakBytes = number(memory, "mlx_peak_bytes")
        coreMLCalls = count(hybrid, "calls_session_total"); coreMLBlocks = count(hybrid, "block_count"); coreMLRows = count(hybrid, "bucket")
        coreMLLoadSeconds = number(hybrid, "load_seconds")
        coreMLPredictionSeconds = number(hybrid, "prediction_seconds_session_total")
        coreMLOutputCopyBytes = number(hybrid, "output_copy_bytes_session_total")
        computeUnits = hybrid["compute_units"] as? String
        let convrot = raw["qwen_convrot_streaming"] as? [String: Any] ?? [:]
        streamWeightBudgetBytes = number(convrot,"requested_weight_budget_bytes").flatMap { $0 > 0 ? $0 : nil }
        let cacheStage=convrot["denoiser"] as? [String:Any] ?? [:]
        streamCacheEvictions=count(cacheStage,"cache_evictions")
        if let flag=cacheStage["cache_limited_by_memory"] as? NSNumber, CFGetTypeID(flag)==CFBooleanGetTypeID() {
            streamCacheLimitedByMemory=flag.boolValue
        } else { streamCacheLimitedByMemory=nil }
        if let encoder = convrot["encoder"] as? [String: Any],
           let denoiser = convrot["denoiser"] as? [String: Any] {
            func sum(_ key: String) -> Double? {
                guard let a = number(encoder, key), let b = number(denoiser, key),
                      (a + b).isFinite else { return nil }
                return a + b
            }
            if let a = count(encoder, "completed_streamed_layers"),
               let b = count(denoiser, "completed_streamed_layers") {
                let total = a.addingReportingOverflow(b)
                streamedBlocks = total.overflow ? nil : total.partialValue
            } else { streamedBlocks = nil }
            func loadedBytes(_ stage: [String:Any]) -> Double? {
                guard let resident=number(stage,"resident_source_bytes"),
                      let streamed=number(stage,"streamed_source_bytes") else { return nil }
                let gathered: Double
                if stage["gathered_embedding_source_bytes"] == nil { gathered=0 }
                else if let value=number(stage,"gathered_embedding_source_bytes") { gathered=value }
                else { return nil }
                let total=resident+streamed+gathered
                return total.isFinite ? total : nil
            }
            if let a=loadedBytes(encoder),let b=loadedBytes(denoiser),(a+b).isFinite {
                streamReadBytes=a+b
            } else { streamReadBytes=nil }
            streamWaitSeconds = sum("wait_seconds")
            // Maxima describe sequential phases only when the native receipt
            // confirms that their weights were released before VAE decoding.
            if let released = convrot["components_released_before_vae"] as? NSNumber,
               CFGetTypeID(released) == CFBooleanGetTypeID(), released.boolValue {
                if let a = number(encoder, "resident_source_bytes"),
                   let b = number(denoiser, "resident_source_bytes") {
                    streamResidentWeightBytes = max(a, b)
                } else { streamResidentWeightBytes = nil }
                if let a = number(encoder, "managed_weight_capacity_bytes"),
                   let b = number(denoiser, "managed_weight_capacity_bytes"),
                   let ad = number(encoder, "derived_weight_capacity_bytes"),
                   let bd = number(denoiser, "derived_weight_capacity_bytes"),
                   (a + ad).isFinite, (b + bd).isFinite {
                    streamManagedWeightBytes = max(a + ad, b + bd)
                } else { streamManagedWeightBytes = nil }
                if let a = count(encoder, "prefix"), let b = count(denoiser, "prefix") {
                    streamResidentLayers = max(a, b)
                } else { streamResidentLayers = nil }
            } else {
                streamResidentWeightBytes = nil
                streamManagedWeightBytes = nil
                streamResidentLayers = nil
            }
        } else {
            let streaming = raw["block_residency"] as? [String: Any]
                ?? raw["block_streaming"] as? [String: Any]
                ?? raw["ltx_streaming_observation"] as? [String: Any] ?? [:]
            streamedBlocks = count(streaming, "streamed_blocks")
            streamReadBytes = number(streaming, "request_bytes_loaded")
            streamWaitSeconds = number(streaming, "request_wait_seconds")
            streamResidentWeightBytes = nil
            streamManagedWeightBytes = nil
            streamResidentLayers = nil
        }
        let dit = raw["qwen21_dbcache"] as? [String: Any] ?? [:]
        let mode = dit["mode"] as? String
        qwen21DiTCacheMode = ["off", "conservative", "balanced", "fast", "diagnostic"].contains(mode ?? "") ? mode : nil
        qwen21DiTCacheThreshold = number(dit, "threshold")
        qwen21DiTCacheMaxConsecutive = count(dit, "max_consecutive")
        qwen21DiTCacheSteps = count(dit, "cached_steps")
        qwen21DiTCacheSavedBlocks = count(dit, "saved_middle_blocks")
    }
    var cacheLabel: String {
        switch promptCacheHit { case .some(true): return "已复用文本编码"; case .some(false): return "本次重新编码文本"; case .none: return "文本缓存未报告" }
    }
    var encoderTotalSeconds: Double? {
        guard let load = encoderLoadSeconds, let encode = textSeconds,
              (load + encode).isFinite else { return nil }
        return load + encode
    }
    var stage1MeanStepSeconds: Double? {
        stage1StepSeconds.map { $0.reduce(0,+) / Double($0.count) }
    }
    var stage2MeanStepSeconds: Double? {
        stage2StepSeconds.map { $0.reduce(0,+) / Double($0.count) }
    }
    static func memory(_ bytes: Double) -> String { String(format: "%.2f GiB", bytes / 1_073_741_824) }
}
