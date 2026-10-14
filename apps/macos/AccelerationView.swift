import SwiftUI
import AppKit

struct AccelerationView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @ObservedObject var library: ModelLibraryController
    private var model: StudioModel? { studio.models.first { $0.id == studio.draft.modelID } }
    private var supportsGPUANE: Bool { model?.supports_gpu_ane == true }
    private var supportsAutomaticGPUANE: Bool {
        studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo"
    }
    private var automaticBucket: Int? {
        AccelerationDiscovery.automaticBucket(
            modelID: studio.draft.modelID, operation: studio.draft.operation,
            width: studio.draft.width, height: studio.draft.height,
            steps: studio.draft.steps, residency: studio.draft.residency,
            hasInputs: !studio.draft.activeAssets.isEmpty)
    }
    private var discoveryID: String {
        let adapters = studio.draft.activeLoRAs.map { "\($0.path):\($0.strength):\($0.role)" }.joined(separator: "|")
        return "\(studio.draft.modelID)|\(studio.draft.modelPath)|\(studio.draft.operation)|\(studio.draft.width)x\(studio.draft.height)|\(studio.draft.steps)|\(studio.draft.residency)|\(adapters)"
    }
    private var config: StudioAcceleration { studio.draft.acceleration ?? StudioAcceleration(policy: studio.draft.profilePath.isEmpty ? "gpu" : "profile") }
    @State private var zImageBucket: Int?
    private var zImageCapacity: String? {
        guard studio.draft.modelID == "z-image-turbo", let zImageBucket else { return nil }
        let imageRows = ((studio.draft.width / 16) * (studio.draft.height / 16) + 31) / 32 * 32
        let textRows = min(512, max(0, (zImageBucket - imageRows) / 32 * 32))
        return textRows == 0
            ? "此分区无法容纳当前尺寸和提示词，请选择更大的分区或使用 GPU。"
            : "当前尺寸可容纳最多 \(textRows) 个编码后的文本 token；更长的提示词需要更大分区或使用 GPU。"
    }
    @State private var discoveryMessage = "正在检测本机分区…"
    private func update(_ change: (inout StudioAcceleration) -> Void) { var value = config; change(&value); value.automaticVersion = 1; studio.draft.acceleration = value }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("推理加速与准备").font(.title2)
            if studio.draft.modelID == "seedvr2-3b" { seedvr2VAEControls }
            Toggle("GPU", isOn: .constant(true)).toggleStyle(.checkbox).disabled(true)
            Toggle("额外启用 ANE", isOn: Binding(get: { studio.draft.usesANE }, set: { studio.setANEEnabled($0) }))
                .toggleStyle(.checkbox).disabled(store.busy || !supportsGPUANE).accessibilityIdentifier("enableANE")
            Text(studio.draft.modelID == "seedvr2-3b"
                 ? "默认只使用 GPU。手动混合模式需选择通过校验的原始 FP16 已编译分区；不自动编译或替换分区。"
                 : "默认只使用 GPU；勾选 ANE 后先复用已编译缓存，缺失时才编译源分区。")
                .font(.caption).foregroundStyle(.secondary)
            if studio.draft.usesANE, let status = store.accelerationStatus { Text(status).font(.caption).foregroundStyle(.secondary) }
            if config.policy == "auto" {
                Text(discoveryMessage).font(.caption).foregroundStyle(.secondary)
                Text(supportsAutomaticGPUANE
                     ? "只有机型、任务、尺寸、步数、token 容量和分区几何都命中实测案例时才使用混合路线；其余情况回退 GPU。实际 ANE 驻留由系统决定。"
                     : "当前模型尚无通过端到端性能门禁的自动 GPU + ANE 配置，自动模式使用稳定路径。")
                    .font(.caption).foregroundStyle(.secondary)
                if supportsAutomaticGPUANE { Button("重新检测本机加速") { Task { await discover() } }.disabled(store.busy) }
            }
            if config.policy == "gpu_ane" {
                Text(studio.draft.modelID == "seedvr2-3b"
                     ? "GPU 处理 attention、文本 FFN、VAE 与视频 FFN 的剩余通道；Core ML CPU+NE 处理原始 FP16 前缀，不量化权重。固定行数分批保留完整视频，实际 ANE 驻留由系统决定。"
                     : "GPU 处理 attention，Core ML 处理量化 MLP；结果可能与纯 GPU 略有不同。实际 ANE 驻留由系统决定。固定或变长分区的容量必须容纳文本和所有图片 token。")
                    .font(.caption).foregroundStyle(.secondary)
                if (studio.draft.modelID.hasPrefix("flux2-") || studio.draft.modelID == "z-image-turbo") && !studio.draft.activeLoRAs.isEmpty {
                    Text("带 LoRA 的 ANE 加速需要匹配同一文件与强度的分区。没有匹配缓存时会提示选择对应分区，或关闭 ANE 使用 GPU。")
                        .font(.caption).foregroundStyle(.orange)
                }
                Button(config.manifest.isEmpty ? "选择已编译分区 manifest…" : "更换已编译分区 manifest…") { choose(compiled: true) }
                if !config.manifest.isEmpty { Text(URL(fileURLWithPath: config.manifest).lastPathComponent).font(.caption).textSelection(.enabled) }
                if let zImageCapacity { Text(zImageCapacity).font(.caption).foregroundStyle(.secondary) }
            }
            if config.policy == "gpu" {
                if studio.draft.modelID.hasPrefix("flux2-") {
                    Toggle("编译融合单流计算块", isOn: Binding(get: { config.compileGPU ?? false }, set: { value in update { $0.compileGPU = value } })).disabled(store.busy).accessibilityIdentifier("compileGPU")
                    Text("保留块间取消；首次使用新形状会编译，预热可提前完成。不同尺寸的收益需分别实测，并非整张网络一次融合。").font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("当前模型使用自身的 native Metal/MLX 图与 pipeline cache；没有可单独开启的 FLUX 单块编译选项。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack {
                Button("加载当前配置") { prepare(warmup: false) }.accessibilityIdentifier("prepareModel")
                Button("预热当前任务") { prepare(warmup: true) }.accessibilityIdentifier("warmupModel")
                Button("卸载模型") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }.disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
            }.disabled(store.busy || store.externalServiceActive || studio.draft.modelPath.isEmpty)
            Text("预热使用当前提示词、输入图、尺寸和模式，实际运行一次但不保存结果；更换这些条件后可能需要重新准备。").font(.caption).foregroundStyle(.secondary)
            Divider()
            if studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo" {
                Button("选择已有 Core ML 源分区…") { choose(compiled: false) }.disabled(store.busy)
                CoreMLStorageView(store: store, studio: studio)
            }
        }.task(id: discoveryID) { await discover() }
            .task(id: "\(studio.draft.modelID)|\(config.manifest)") { await readZImageBucket() }
            .padding(20).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
    }
    private var seedvr2Config: NativeSeedVR2VAEConfig { studio.draft.seedvr2VAE ?? NativeSeedVR2VAEConfig() }
    private func updateSeedVR2(_ change: (inout NativeSeedVR2VAEConfig) -> Void) {
        var config = seedvr2Config; change(&config); studio.draft.seedvr2VAE = config; studio.save()
    }
    @ViewBuilder private var seedvr2VAEControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SeedVR2 VAE / 内存").font(.headline)
            Picker("编码空间阶段", selection: Binding(get: { seedvr2Config.encoder_spatial }, set: { value in updateSeedVR2 {
                $0.encoder_spatial = value
                if value == "file_rows" && !["singleton2d", "conv3d"].contains($0.convolution) { $0.convolution = "singleton2d" }
            } })) {
                Text("完整编码（原有路径）").tag("full")
                Text("全局统计编码 head 行带（视频较慢）").tag("file_rows")
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAEEncoderSpatial")
            Picker("解码空间阶段", selection: Binding(get: { seedvr2Config.spatial }, set: { value in updateSeedVR2 {
                $0.spatial = value
                if value == "file_rows" && !["singleton2d", "conv3d"].contains($0.convolution) { $0.convolution = "singleton2d" }
            } })) {
                Text("完整帧（原有路径）").tag("full")
                Text("全局统计行带 / 临时历史文件（较慢）").tag("file_rows")
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAESpatial")
            if seedvr2Config.spatial == "file_rows" || seedvr2Config.encoder_spatial == "file_rows" {
                Picker("空间行带高度", selection: Binding(get: { seedvr2Config.spatial_rows }, set: { value in updateSeedVR2 { $0.spatial_rows = value } })) {
                    Text("32 行").tag(32); Text("64 行").tag(64); Text("128 行").tag(128)
                }.disabled(store.busy).accessibilityIdentifier("seedvr2VAESpatialRows")
                Text("只调度高分辨率编码 head / 解码阶段，保留全帧统计/完整 attention 与时间降采样。当前仅支持 ≤1024 画布；临时文件最多 8 GiB，视频累计 I/O 会很大。卷积拆分可能改变 FP16 末位，不是硬 RAM 预算或 16 GB 可用性保证。")
                    .font(.caption).foregroundStyle(.orange)
            }
            Picker("视频输入驻留", selection: Binding(get: { seedvr2Config.encoder }, set: { value in updateSeedVR2 { $0.encoder = value } })) {
                Text("整段导入（原有路径）").tag("buffered")
                Text("逐帧导入编码（减少 RGB 驻留）").tag("streaming")
            }.disabled(store.busy || studio.draft.operation != "video.upscale")
                .accessibilityIdentifier("seedvr2VAEEncoder")
            Picker("视频输出驻留", selection: Binding(get: { seedvr2Config.decoder }, set: { value in updateSeedVR2 { $0.decoder = value } })) {
                Text("整段缓存（原有路径）").tag("buffered")
                Text("逐帧导出（降低输出驻留）").tag("streaming")
            }.disabled(store.busy || studio.draft.operation != "video.upscale")
                .accessibilityIdentifier("seedvr2VAEDecoder")
            Picker("卷积工作区", selection: Binding(get: { seedvr2Config.convolution }, set: { value in updateSeedVR2 { $0.convolution = value } })) {
                Text("速度优先 · singleton2d").tag("singleton2d")
                if seedvr2Config.spatial == "full" && seedvr2Config.encoder_spatial == "full" {
                    Text("大卷积分块 · mixed3d（较慢）").tag("mixed3d")
                    Text("全部卷积分块 · tiled3d（更慢）").tag("tiled3d")
                }
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAEConvolution")
            Picker("因果历史缓存", selection: Binding(get: { seedvr2Config.history }, set: { value in updateSeedVR2 { $0.history = value } })) {
                Text("紧凑历史 · compact（原有默认）").tag("compact")
                Text("独立帧引用 · frames（减少复制）").tag("frames")
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAEHistory")
            Picker("临时分配器缓存", selection: Binding(get: { seedvr2Config.allocator_cache_bytes }, set: { value in updateSeedVR2 { $0.allocator_cache_bytes = value } })) {
                Text("4 GiB（速度优先）").tag(UInt64(4) << 30)
                Text("1 GiB（减少缓存）").tag(UInt64(1) << 30)
                Text("不保留空闲缓存（较慢）").tag(UInt64(0))
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAECache")
            Text("逐帧导入按原有 5/4 帧分段编码；分块仅限制卷积工作区，不是整套 VAE 的空间 tiling，也不是硬内存上限。输入/输出流式仍保留完整条件/DiT latent；当前源视频上限仍为 121 帧。")
                .font(.caption).foregroundStyle(.secondary)
            if studio.draft.frames >= 65 || ProcessInfo.processInfo.physicalMemory <= (UInt64(32) << 30) {
                Text("较长视频或低内存设备：建议分阶段驻留 + 逐帧导出 + 较小缓存。1024 输出仍可能超过低内存机器容量，可先使用 512；本建议是启发式，不是已验证的设备预算。")
                    .font(.caption).foregroundStyle(.orange)
                Button("应用省内存建议") {
                    studio.draft.residency = "component_staged"
                    updateSeedVR2 {
                        $0.decoder = studio.draft.operation == "video.upscale" ? "streaming" : "buffered"
                        $0.allocator_cache_bytes = UInt64(1) << 30
                    }
                }.disabled(store.busy).accessibilityIdentifier("seedvr2ApplyMemoryAdvice")
            }
        }
    }
    private func readZImageBucket() async {
        zImageBucket = nil
        guard studio.draft.modelID == "z-image-turbo", !config.manifest.isEmpty else { return }
        let path = config.manifest
        // File access may wait for macOS permissions; never do it during rendering.
        let bucket = await Task.detached(priority: .utility) { () -> Int? in
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
                  let manifest = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let shape = manifest["shape"] as? [String: Any],
                  let buckets = shape["buckets"] as? [Int], !buckets.isEmpty else { return nil }
            return buckets.max()
        }.value
        guard !Task.isCancelled else { return }
        zImageBucket = bucket
    }
    private func discover() async {
        guard supportsAutomaticGPUANE else {
            discoveryMessage = supportsGPUANE
                ? "当前模型的 GPU + ANE 仍为显式实验配置；自动模式不会启用。"
                : "当前模型使用稳定 GPU 路径。"
            return
        }
        guard let requiredRows = automaticBucket else {
            discoveryMessage = "当前操作、尺寸、步数或驻留模式没有实测混合案例，自动模式使用 GPU。"
            return
        }
        let path = studio.draft.modelPath, preferred = config.manifest
        let minimumRows = (studio.draft.width / 16) * (studio.draft.height / 16) + 1
        let selectedCache = config.coreMLCache.map { URL(fileURLWithPath: $0) }
        let modelID = studio.draft.modelID
        let loras = studio.draft.activeLoRAs
        let result = await Task.detached {
            AccelerationDiscovery.find(modelPath: path, preferred: preferred,
                                       cache: selectedCache, minimumRows: minimumRows,
                                       requiredRows: requiredRows,
                                       enforceAutomaticPolicy: true,
                                       modelID: modelID, loras: loras)
        }.value
        guard !Task.isCancelled, studio.draft.modelPath == path else { return }
        let system = (try? JSONSerialization.jsonObject(with: Data(NativeEngine.system().utf8))) as? [String: Any]
        let gpu = system?["gpu"] as? String
        let memory = (system?["physical_memory_bytes"] as? NSNumber)?.uint64Value
        if let result {
            let supported = AccelerationDiscovery.automaticPolicyMatches(
                gpu: gpu ?? "unknown", memory: memory ?? 0,
                mlpWidth: result.mlpWidth, start: result.aneMLPStart, end: result.aneMLPEnd,
                modelID: modelID, bucket: result.rows)
            discoveryMessage = supported ? "已发现本机可用分区 · \(result.rows) token · ANE MLP \(result.aneMLPEnd - result.aneMLPStart)/\(result.mlpWidth)；生成时按实际输入复核。" : "已发现本地分区；此机型尚无自动混合策略验证，自动模式使用 GPU。"
            if config.policy == "auto" || config.manifest.isEmpty { update { $0.manifest = result.manifest; $0.sourceManifest = result.source } }
        } else { discoveryMessage = "未发现匹配当前权重的完整编译分区，自动模式使用 GPU。可先预编译或指定本地分区。" }
    }
    private func choose(compiled: Bool) {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        panel.message = compiled ? "选择引用 .mlmodelc 的分区 manifest" : "选择引用 .mlpackage 的源分区 manifest"
        if panel.runModal() == .OK, let url = panel.url {
            if studio.draft.modelID == "seedvr2-3b" {
                // SeedVR2 uses its exact original-FP16 prefix-bank ABI, not
                // the generic quantized library manifest schema. Native plan
                // and load perform the qualified-manifest/tree checks.
                update { $0.manifest = url.path; $0.sourceManifest = ""; $0.policy = "gpu_ane" }
                studio.message = nil; studio.save()
            } else {
                library.registerANE(url, modelID: studio.draft.modelID, studio: studio, select: true)
            }
        }
    }
    private func prepare(warmup: Bool) {
        guard !store.externalServiceActive else { studio.message = "请先停止本地 API。"; return }
        let snapshot = studio.draft
        studio.message = nil
        Task { do {
            let resolved = try await store.resolveAcceleration(snapshot)
            studio.rememberAcceleration(resolved)
            let output = store.directory.appendingPathComponent("unused-warmup.png")
            let request = try await Task.detached { try resolved.request(output: output) }.value
            try await store.prepare(modelURL: URL(fileURLWithPath: resolved.modelPath), request: request, warmup: warmup)
        } catch { studio.message = error is CancellationError ? "准备已取消" : error.localizedDescription } }
    }
}
