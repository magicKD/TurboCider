import Foundation

// Read-only advice, never an admission override, dimension change or hard RAM
// bound. Scratch figures are measured reference cases, not arbitrary-size fits.
struct SeedVR2ResourceAdvice {
    struct Notice: Identifiable, Equatable {
        let id: String
        let message: String
    }
    static func gib(_ bytes: UInt64) -> String {
        String(format: "%.1f GiB", Double(bytes) / Double(UInt64(1) << 30))
    }
    static func notices(operation: String, width: Int, height: Int, sourceFrames: Int?, configuredFrames: Int,
                        physicalMemory: UInt64, options: NativeSeedVR2VAEConfig, temporaryFreeBytes: UInt64?) -> [Notice] {
        let video = operation == "video.upscale"
        let frames = sourceFrames ?? configuredFrames
        let decoderRows = options.spatial == "file_rows"
        let encoderRows = options.encoder_spatial == "file_rows"
        var result: [Notice] = []
        func add(_ id: String, _ message: String) { result.append(Notice(id: id, message: message)) }
        if video && sourceFrames == nil {
            add("source_not_inspected", "请先导入完整源视频以确认实际帧数；当前配置值不能证明源视频满足模型帧数上限。")
        }
        if video, let sourceFrames, sourceFrames != configuredFrames {
            add("source_frame_mismatch", "配置帧数与已导入源视频不一致；必须保留完整的 \(sourceFrames) 帧，省内存建议不会截短源视频。")
        }
        if video && frames > 121 {
            add("model_frame_limit", "完整源视频有 \(frames) 帧，超过当前模型路径的 121 帧上限。逐帧导入、VAE 分块或较小缓存都不能绕过此限制；不会自动截帧。")
        }
        if physicalMemory <= (UInt64(16) << 30) {
            add("low_memory_unqualified", "当前工作流尚无 16 GiB 及以下设备的实机资格验证；即使应用省内存建议，保守的 native 入场检查仍可能拒绝任务。不会自动放宽内存检查。")
        }
        if video && (frames >= 65 || physicalMemory <= (UInt64(32) << 30)) {
            add("streaming_advice", "可使用分阶段驻留、逐帧导入/导出、独立帧历史与较小缓存；它们不消除完整条件和 DiT 状态，也不保证任务能装入本机内存。")
        }
        guard decoderRows || encoderRows else { return result }
        add("vae_only_saving", "空间行带只降低 VAE 相关驻留／工作区；完整条件和 DiT 状态仍占内存，整段超分峰值不一定同比下降。")
        if width > 1024 || height > 1024 {
            add("spatial_canvas_limit", "当前编码 head／解码行带方案仅支持 ≤1024 画布；请选择受支持的画布或完整空间路径。不会自动缩小输出。")
        }
        guard video else {
            add("image_no_history_files", "单帧图像不创建视频因果历史文件；临时文件配额不是 RAM 上限。")
            return result
        }
        if temporaryFreeBytes == nil {
            add("scratch_space_unknown", "无法读取当前临时卷的可用空间；不能据此承诺视频行带方案可运行。")
        }
        // Decoder/encoder files have separate lifetimes. Do not sum their
        // independently measured 6.5/2.25-GiB peaks as simultaneous storage.
        if width == 1024 && height == 1024 && frames >= 5 && options.spatial_rows == 64 {
            let measured: UInt64 = decoderRows ? (UInt64(13) << 29) : (UInt64(9) << 28)
            add("scratch_reference", "同尺寸已测视频行带案例的活跃临时文件峰值约 \(gib(measured))，累计 I/O 可达数百 GB。此参考值不是磁盘或 RAM 硬上限，其他尺寸与行带高度不能直接套用。")
            if options.spatial_disk_budget_bytes < measured {
                add("scratch_quota_low", "当前临时文件配额 \(gib(options.spatial_disk_budget_bytes)) 低于同尺寸已测峰值 \(gib(measured))，可能在处理中途失败；请先调整配置。")
            }
            if let free = temporaryFreeBytes, free < measured + (UInt64(512) << 20) {
                add("scratch_space_low", "当前临时卷可用 \(gib(free))，不足以容纳同尺寸已测临时文件峰值并预留 0.5 GiB。请先释放空间，或使用不落盘的方案；不会自动删除文件。")
            }
        } else {
            add("scratch_reference_unknown", "当前尺寸／帧数／行带高度没有对应的临时文件峰值凭证；配额 \(gib(options.spatial_disk_budget_bytes)) 只是上限，不是所需空间估算。请保留充足磁盘空间。")
        }
        return result
    }
}
