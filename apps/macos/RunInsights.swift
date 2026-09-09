import Foundation
import CoreFoundation

/// Snapshot metrics retain their native scope; absence is never zero usage.
struct RunInsights {
    let promptCacheHit: Bool?
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
    init(json: String) {
        let raw = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any] ?? [:]
        let memory = raw["memory"] as? [String: Any] ?? raw
        let hybrid = raw["hybrid"] as? [String: Any] ?? [:]
        let times = raw["timings_seconds"] as? [String: Any] ?? [:]
        func number(_ dict: [String: Any], _ key: String) -> Double? {
            guard let n = dict[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite, n.doubleValue >= 0 else { return nil }
            return n.doubleValue
        }
        func count(_ key: String) -> Int? {
            guard let n = number(hybrid, key), n < Double(Int.max), n.rounded(.down) == n else { return nil }
            return Int(n)
        }
        if let hit = raw["prompt_cache_hit"] as? NSNumber, CFGetTypeID(hit) == CFBooleanGetTypeID() { promptCacheHit = hit.boolValue }
        else { promptCacheHit = nil }
        textSeconds = number(times, "text_encode"); denoiseSeconds = number(times, "denoise")
        activeBytes = number(memory, "mlx_active_bytes"); peakBytes = number(memory, "mlx_peak_bytes")
        coreMLCalls = count("calls_session_total"); coreMLBlocks = count("block_count"); coreMLRows = count("bucket")
        coreMLLoadSeconds = number(hybrid, "load_seconds")
        coreMLPredictionSeconds = number(hybrid, "prediction_seconds_session_total")
        coreMLOutputCopyBytes = number(hybrid, "output_copy_bytes_session_total")
        computeUnits = hybrid["compute_units"] as? String
    }
    var cacheLabel: String {
        switch promptCacheHit { case .some(true): return "已复用文本编码"; case .some(false): return "本次重新编码文本"; case .none: return "文本缓存未报告" }
    }
    static func memory(_ bytes: Double) -> String { String(format: "%.2f GiB", bytes / 1_073_741_824) }
}
