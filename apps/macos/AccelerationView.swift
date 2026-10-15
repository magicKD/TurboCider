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
    @State private var seedvr2TemporaryFreeBytes: UInt64?
    @State private var seedvr2ResourceRevision = 0
    @State private var qwenLowMemoryANEAvailable = false
    @State private var runtimeANEAvailable = false
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
                 : "默认只使用 GPU；普通 ANE 使用匹配的冻结分区，本机 Runtime 实验使用独立进程。")
                .font(.caption).foregroundStyle(.secondary)
            Text("本次执行设备：\(studio.draft.executionDeviceLabel)").font(.caption)
                .accessibilityIdentifier("accelerationExecutionDevice")
            if let notice = studio.draft.aneConfigurationNotice {
                Text(notice).font(.caption).foregroundStyle(.orange)
                    .accessibilityIdentifier("accelerationLoRANotice")
                Button("改用纯 GPU") { studio.setANEEnabled(false) }
                    .disabled(store.busy).accessibilityIdentifier("accelerationLoRAUseGPU")
            }
            if studio.draft.usesANE, let status = store.accelerationStatus { Text(status).font(.caption).foregroundStyle(.secondary) }
            if ["z-image-turbo", "qwen-image-2.1"].contains(studio.draft.modelID) {
                Toggle("Runtime · Hadamard W8A8 数据路径（本机实验）", isOn: Binding(
                    get: { studio.draft.usesRuntimeANE },
                    set: { enabled in
                        guard !enabled || runtimeANEAvailable else { return }
                        update {
                            $0.runtimeAneProfileID = enabled ? RuntimeImageWorker.profileID : nil
                            $0.runtimeDescriptor = nil
                            $0.policy = enabled ? "gpu_ane" : "gpu"
                            if enabled { $0.qwen21W8A8 = false; $0.compileGPU = false }
                        }
                        studio.draft.profilePath = ""
                    }))
                    .disabled(store.busy || (!runtimeANEAvailable && !studio.draft.usesRuntimeANE))
                    .accessibilityIdentifier("runtimeANEExperiment")
                Text("本机实验仅限 Apple M4 Pro / 48 GiB：Z-Image BF16 8 步，Qwen 基础 20 步或 Viggle r128 6 步、强度 1。固定 512 输出；Qwen 最多 2 张参考图、标准 1024 编码。当前版本待图像性能与质量复验，不保证更快；实际回退会显示在结果中。")
                Text("分区按实际 token 行数选择：Qwen 的 512 输出使用 1024 行；Z-Image 将图像与文本行数合并后选择容量，长文本分块处理。首次使用新容量需要编译，缓存可在此清理。")
                    .font(.caption).foregroundStyle(.secondary)
                if studio.draft.usesRuntimeANE {
                    Button("应用实验设置") {
                        studio.draft.width = 512; studio.draft.height = 512; studio.draft.frames = 1; studio.draft.audio = false; studio.draft.fps = 24
                        studio.draft.steps = studio.draft.modelID == "z-image-turbo" ? 8 : studio.draft.activeLoRAs.isEmpty ? 20 : 6
                        studio.draft.residency = studio.draft.modelID == "z-image-turbo" ? "resident" : "component_staged"
                        studio.draft.dynamicText = true; studio.draft.promptEnhance = false
                        studio.draft.promptEnhanceEditExperimental = false; studio.draft.qwen21DiTCache = "off"
                        studio.draft.qwen21ReferenceSize = 1024; studio.draft.streaming = StudioStreamingState()
                        studio.draft.loraStrategy = "inference_time"
                    }.disabled(store.busy).accessibilityIdentifier("runtimeANEQualifiedSettings")
                    Text("仅生成很小的形状描述，不导出完整权重。独立进程结束后释放模型；首次编译可能更慢。清理只删除 App 程序源缓存，系统缓存保留，不等于冷启动。")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("清理 App 程序源缓存") {
                        Task {
                            do { try await store.clearRuntimeProgramCache() } catch { studio.message = error.localizedDescription }
                        }
                    }.disabled(store.busy || store.externalServiceActive || store.resolvingAcceleration)
                        .accessibilityIdentifier("runtimeANEClearCache")
                }
                if !runtimeANEAvailable {
                    Text("当前设备尚未验收此实验；可关闭已有选择并继续使用 GPU。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if config.policy == "auto" {
                Text(discoveryMessage).font(.caption).foregroundStyle(.secondary)
                Text(supportsAutomaticGPUANE
                     ? "只有机型、任务、尺寸、步数、token 容量和分区几何都命中实测案例时才使用混合路线；其余情况回退 GPU。实际 ANE 驻留由系统决定。"
                     : "当前模型尚无通过端到端性能门禁的自动 GPU + ANE 配置，自动模式使用稳定路径。")
                    .font(.caption).foregroundStyle(.secondary)
                if supportsAutomaticGPUANE { Button("重新检测本机加速") { Task { await discover() } }.disabled(store.busy) }
            }
            if config.policy == "gpu_ane" && !studio.draft.usesRuntimeANE {
                if studio.draft.modelID == "qwen-image-2.1" {
                    Toggle("Qwen 固定量化分区（实验）", isOn: Binding(
                        get: { config.qwen21W8A8 ?? false },
                        set: { value in
                            guard !value || qwenLowMemoryANEAvailable else { return }
                            update { $0.qwen21W8A8 = value }
                            if value { studio.draft.residency = "component_staged" }
                        }))
                        .disabled(store.busy || (!qwenLowMemoryANEAvailable && config.qwen21W8A8 != true))
                        .accessibilityIdentifier("qwenLowMemoryANE")
                    Text("使用已编译的冻结量化分区，仅支持 Apple M5 Pro、24 GiB 内存和 512×512 文生图。需选择匹配分区并关闭参考图、LoRA 与提示词增强；画面可能与纯 GPU 不同。")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("此选项是冻结量化分区，与上面的 Runtime 实验不同。")
                        .font(.caption).foregroundStyle(.secondary)
                    if !qwenLowMemoryANEAvailable {
                        Text("当前设备不支持此实验；已保存的实验选项可关闭，普通 Qwen ANE 配置仍可使用。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                } else if studio.draft.modelID == "seedvr2-3b" {
                    Text("GPU 处理 attention、文本 FFN、VAE 与视频 FFN 的剩余通道；Core ML CPU+NE 处理原始 FP16 前缀，不量化权重。固定行数分批保留完整视频，实际 ANE 驻留由系统决定。")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("GPU 处理 attention，Core ML 处理量化 MLP；结果可能与纯 GPU 略有不同。实际 ANE 驻留由系统决定。固定或变长分区的容量必须容纳文本和所有图片 token。").font(.caption).foregroundStyle(.secondary)
                }
                if (studio.draft.modelID.hasPrefix("flux2-") || studio.draft.modelID == "z-image-turbo") && !studio.draft.activeLoRAs.isEmpty && !studio.draft.aneLoRARequiresGPU {
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
                Button(studio.draft.usesRuntimeANE ? "准备实验描述" : "加载当前配置") { prepare(warmup: false) }.accessibilityIdentifier("prepareModel")
                Button("预热当前任务") { prepare(warmup: true) }.disabled(studio.draft.usesRuntimeANE).accessibilityIdentifier("warmupModel")
                Button("卸载模型") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }.disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
            }.disabled(store.busy || store.externalServiceActive || studio.draft.modelPath.isEmpty)
            if !studio.draft.usesRuntimeANE { Text("预热使用当前提示词、输入图、尺寸和模式，实际运行一次但不保存结果；更换这些条件后可能需要重新准备。").font(.caption).foregroundStyle(.secondary) }
            Divider()
            if !studio.draft.usesRuntimeANE && (studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo") {
                Button("选择已有 Core ML 源分区…") { choose(compiled: false) }.disabled(store.busy)
                CoreMLStorageView(store: store, studio: studio)
            }
        }.task(id: discoveryID) { await discover() }
            .task { qwenLowMemoryANEAvailable = Qwen21LowMemoryANEHardware.isEligible(); runtimeANEAvailable = RuntimeImageWorker.hardwareEligible() }
            .task(id: "\(studio.draft.modelID)|\(config.manifest)") { await readZImageBucket() }
            .task(id: seedvr2ResourceCheckID) { await readSeedVR2ScratchSpace() }
            .padding(20).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
    }
    private var seedvr2Config: NativeSeedVR2VAEConfig { studio.draft.seedvr2VAE ?? NativeSeedVR2VAEConfig() }
    private var seedvr2ResourceCheckID: String {
        "\(studio.draft.modelID)|\(studio.draft.operation)|\(seedvr2Config.spatial)|\(seedvr2Config.encoder_spatial)|\(seedvr2ResourceRevision)"
    }
    private var seedvr2Notices: [SeedVR2ResourceAdvice.Notice] {
        SeedVR2ResourceAdvice.notices(operation: studio.draft.operation, width: studio.draft.width, height: studio.draft.height,
            sourceFrames: studio.draft.activeAssets.first(where: { $0.mediaKind == "video" })?.frames,
            configuredFrames: studio.draft.frames, physicalMemory: ProcessInfo.processInfo.physicalMemory,
            options: seedvr2Config, temporaryFreeBytes: seedvr2TemporaryFreeBytes)
    }
    private func readSeedVR2ScratchSpace() async {
        guard studio.draft.modelID == "seedvr2-3b" else { seedvr2TemporaryFreeBytes = nil; return }
        let key = seedvr2ResourceCheckID
        let bytes = await Task.detached(priority: .utility) { () -> UInt64? in
            // Filesystem queries may block. Never perform this work while
            // rendering the SwiftUI body or assume a failed query means zero.
            guard let values = try? FileManager.default.attributesOfFileSystem(forPath: FileManager.default.temporaryDirectory.path),
                  let free = values[.systemFreeSize] as? NSNumber else { return nil }
            return free.uint64Value
        }.value
        guard !Task.isCancelled, key == seedvr2ResourceCheckID else { return }
        seedvr2TemporaryFreeBytes = bytes
    }
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
                if value != "full" { $0.share_upscale_history = false }
                if value == "file_rows" && !["singleton2d", "conv3d"].contains($0.convolution) { $0.convolution = "singleton2d" }
            } })) {
                Text("完整帧（原有路径）").tag("full")
                Text("全局统计行带 / 临时历史文件（较慢）").tag("file_rows")
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAESpatial")
            if seedvr2Config.spatial == "file_rows" || seedvr2Config.encoder_spatial == "file_rows" {
                Picker("空间行带高度", selection: Binding(get: { seedvr2Config.spatial_rows }, set: { value in updateSeedVR2 { $0.spatial_rows = value } })) {
                    Text("32 行").tag(32); Text("64 行").tag(64); Text("128 行").tag(128)
                }.disabled(store.busy).accessibilityIdentifier("seedvr2VAESpatialRows")
                Text("只调度高分辨率编码 head / 解码阶段，保留全帧统计/完整 attention 与时间降采样。当前仅支持 ≤1024 画布；当前临时文件配额 \(SeedVR2ResourceAdvice.gib(seedvr2Config.spatial_disk_budget_bytes))，视频累计 I/O 会很大。卷积拆分可能改变 FP16 末位，不是硬 RAM 预算或 16 GB 可用性保证。")
                    .font(.caption).foregroundStyle(.orange)
                HStack {
                    Text(seedvr2TemporaryFreeBytes.map { "临时卷可用空间快照 \(SeedVR2ResourceAdvice.gib($0))" } ?? "临时卷空间待确认")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("刷新空间") { seedvr2ResourceRevision += 1 }.disabled(store.busy)
                }
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
            Picker("卷积工作区", selection: Binding(get: { seedvr2Config.convolution }, set: { value in updateSeedVR2 {
                $0.convolution = value
                if value != "singleton2d" { $0.share_upscale_history = false }
            } })) {
                Text("速度优先 · singleton2d").tag("singleton2d")
                if seedvr2Config.spatial == "full" && seedvr2Config.encoder_spatial == "full" {
                    Text("大卷积分块 · mixed3d（较慢）").tag("mixed3d")
                    Text("全部卷积分块 · tiled3d（更慢）").tag("tiled3d")
                }
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAEConvolution")
            Picker("因果历史缓存", selection: Binding(get: { seedvr2Config.history }, set: { value in updateSeedVR2 {
                $0.history = value
                if value != "frames" { $0.share_upscale_history = false }
            } })) {
                Text("紧凑历史 · compact（原有默认）").tag("compact")
                Text("独立帧引用 · frames（减少复制）").tag("frames")
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAEHistory")
            Toggle("共享空间上采样历史（减少帧副本）", isOn: Binding(get: { seedvr2Config.share_upscale_history }, set: { value in updateSeedVR2 {
                $0.share_upscale_history = value
                if value { $0.history = "frames" }
            } })).disabled(store.busy || seedvr2Config.spatial != "full" || seedvr2Config.convolution != "singleton2d")
                .accessibilityIdentifier("seedvr2VAEShareUpscaleHistory")
            if seedvr2Config.share_upscale_history {
                Text("仅复用已独立生成的空间上采样单帧，保留原有时间阶段和 FP16 数值。需要 frames + singleton2d + 完整解码；图像不保留无用历史。减少副本不等于整段峰值同比下降，也不是硬内存预算。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Picker("临时分配器缓存", selection: Binding(get: { seedvr2Config.allocator_cache_bytes }, set: { value in updateSeedVR2 { $0.allocator_cache_bytes = value } })) {
                Text("4 GiB（速度优先）").tag(UInt64(4) << 30)
                Text("1 GiB（减少缓存）").tag(UInt64(1) << 30)
                Text("不保留空闲缓存（较慢）").tag(UInt64(0))
            }.disabled(store.busy).accessibilityIdentifier("seedvr2VAECache")
            Text("逐帧导入按原有 5/4 帧分段编码；分块仅限制卷积工作区，不是整套 VAE 的空间 tiling，也不是硬内存上限。输入/输出流式仍保留完整条件/DiT latent；当前源视频上限仍为 121 帧。")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(seedvr2Notices) { notice in
                Text(notice.message).font(.caption).foregroundStyle(.orange)
                    .accessibilityIdentifier("seedvr2Advice-\(notice.id)")
            }
            if !seedvr2Notices.isEmpty || studio.draft.frames >= 65 || ProcessInfo.processInfo.physicalMemory <= (UInt64(32) << 30) {
                Button("应用省内存建议") {
                    studio.applySeedVR2MemoryAdvice()
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
        guard !studio.draft.usesRuntimeANE else { return }
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
            if resolved.usesRuntimeANE { studio.message = "实验描述已准备，生成时由独立进程加载并显示实际路线。"; return }
            let output = store.directory.appendingPathComponent("unused-warmup.png")
            let request = try await Task.detached { try resolved.request(output: output) }.value
            try await store.prepare(modelURL: URL(fileURLWithPath: resolved.modelPath), request: request, warmup: warmup)
        } catch { studio.message = error is CancellationError ? "准备已取消" : error.localizedDescription } }
    }
}
