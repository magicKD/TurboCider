import Foundation
import CoreFoundation

/// Snapshot metrics retain their native scope; absence is never zero usage.
struct RunInsights {
    let promptCacheHit: Bool?
    let editPrefixCacheHit: Bool?
    let textSeconds: Double?
    let denoiseSeconds: Double?
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
        if let hit = raw["prompt_cache_hit"] as? NSNumber, CFGetTypeID(hit) == CFBooleanGetTypeID() { promptCacheHit = hit.boolValue }
        else { promptCacheHit = nil }
        let selections = (raw["acceleration_selection"] as? String ?? "")
            .split(separator: ";").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        if raw["model"] as? String == "qwen-image-2.1", raw["operation"] as? String == "image.edit" {
            editPrefixCacheHit = selections.contains("edit prefix KV snapshot hit") ? true
                : selections.contains("edit prefix KV snapshot miss") ? false : nil
        } else { editPrefixCacheHit = nil }
        textSeconds = number(times, "text_encode"); denoiseSeconds = number(times, "denoise")
        activeBytes = number(memory, "mlx_active_bytes"); peakBytes = number(memory, "mlx_peak_bytes")
        coreMLCalls = count(hybrid, "calls_session_total"); coreMLBlocks = count(hybrid, "block_count"); coreMLRows = count(hybrid, "bucket")
        coreMLLoadSeconds = number(hybrid, "load_seconds")
        coreMLPredictionSeconds = number(hybrid, "prediction_seconds_session_total")
        coreMLOutputCopyBytes = number(hybrid, "output_copy_bytes_session_total")
        computeUnits = hybrid["compute_units"] as? String
        let streaming = raw["block_residency"] as? [String: Any] ?? [:]
        streamedBlocks = count(streaming, "streamed_blocks")
        streamReadBytes = number(streaming, "request_bytes_loaded")
        streamWaitSeconds = number(streaming, "request_wait_seconds")
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
    static func memory(_ bytes: Double) -> String { String(format: "%.2f GiB", bytes / 1_073_741_824) }
}
