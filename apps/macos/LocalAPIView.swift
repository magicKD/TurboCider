import SwiftUI
import AppKit

struct LocalAPIView: View {
    @ObservedObject var api: LocalAPIController
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @State private var previewJob: LocalAPIJob?
    @State private var importedJobID: String?
    @State private var resultMessage: String?
    private var connectionInstructions: String {
        """
        使用本机 TurboCider 的 Unix socket API：
        Socket: \(api.socketPath)
        先在 App 的“本地 API”页启动服务。每个连接发送一个 UTF-8 JSON 对象，以换行结尾；每个连接只处理一次请求。
        首先发送 {"action":"capabilities"} 获取操作、字段类型和限制；再发送 {"action":"models"} 查询模型能力（不是已安装模型列表）。
        发送 {"action":"installations"} 获取此服务所用模型库的 root 和 index；其中 installations、loras、anePartitions 是已登记的本机路径。files_verified=false 表示未检查权重完整性，不能据此保证可运行。
        模型库查询由服务自身配套助手执行，不使用调用方的环境或默认目录；旧版服务不支持此操作时需更新完整 App/CLI 包，不扫描目录猜模型。
        图片缩放可显式调用 image_prepare 的 input：source_path（绝对路径）、preset（original/automatic/fit512/portrait512/landscape512），需要缩小时提供全新 output_path（绝对 .png 路径）。它只用 CPU、保留比例和透明度，不裁剪、补边或放大，不修改原图。automatic 长边最多 1024；其余缩放边界为 512×512、512×768、768×512。
        必须使用 image_prepare 返回的 image_path；无需缩小时返回原图，output_path 不会被写。此同步文件操作会暂占 RPC 调度，不能保证立即取消；连接中断后先检查输出，不盲目重试。CLI 等价命令为 turbocider prepare-image INPUT.json。
        发送 {"action":"workflows"} 获取 Playground 工作流、图片角色和参数；使用 workflow_request 的 input（workflow_id、role_paths、request、可选 instruction）组合请求。它只组合参数，不读取图片或执行生成；返回的 request 仍须经过 plan。
        响应格式为 {"ok":true,"result":...} 或 {"ok":false,"error":"..."}。
        使用 plan 检查完整生成请求，使用 submit 提交 model_path 和 request，保存返回的 id；用 status 查询直到 succeeded/failed/cancelled/interrupted。
        模型、LoRA、参考图与输出均使用本机路径；参考图按 inputs 顺序编号，每步使用独立输出路径。上一任务 succeeded 后再将其输出用于下一步。
        提交后若连接中断，先用 jobs 核对是否已入队，不自动重复提交。等待超时不会取消任务；需要停止时显式调用 cancel 并继续查询状态。
        本地 API 页可分页查看 API 任务和图片结果，完成图可预览、定位或导入创作参考；API 历史仍与创作页分开保存，不自动复制素材。退出 App 会停止它启动的服务。所有工作只使用已有本地文件，不下载模型。
        """
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Label("本地 API", systemImage: "network").font(.largeTitle.weight(.semibold))
                Text("让本机脚本与其他应用通过同一任务服务调用 TurboCider。").foregroundStyle(.secondary)
                GroupBox {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Circle().fill(api.running ? .green : .secondary).frame(width: 9, height: 9)
                            Text(api.status).font(.headline); Spacer()
                            if api.changing { ProgressView().controlSize(.small) }
                            Button(api.running ? "停止服务" : "启动服务") {
                                if api.running { api.stop() } else { Task { await api.start(store: store) } }
                            }.disabled(api.changing || store.busy).accessibilityIdentifier("toggleLocalAPI")
                        }
                        LabeledContent("Socket", value: api.socketPath).font(.callout.monospaced()).textSelection(.enabled)
                        Text("启动时释放 App 的嵌入式模型会话；API 服务按需加载并复用模型。停止服务后可回到创作页生成。").font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button("复制 Socket 路径") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(api.socketPath, forType: .string) }
                            Button("复制 AI 接入说明") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(connectionInstructions, forType: .string) }
                                .accessibilityIdentifier("copyAIConnectionInstructions")
                            Button("打开日志目录") { NSWorkspace.shared.open(api.directory) }
                            if api.running { Button("刷新状态") { Task { await api.refresh() } }; Text("历史任务 \(api.jobCount)").font(.caption) }
                        }
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                }
                jobsSection
                Text("调用方式").font(.title2)
                Text("此服务使用 Unix socket JSON RPC，适合当前 Mac 的本地进程；不监听公网端口。每个连接发送一个以换行结尾的 JSON 对象。").foregroundStyle(.secondary)
                Text("turbocider rpc \(api.socketPath) rpc.json\n\nrpc.json:\n{\"action\":\"capabilities\"}")
                    .font(.system(.body, design: .monospaced)).textSelection(.enabled).padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading).background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
                if api.running { Text(api.activeJob.isEmpty ? "当前没有执行中的 API 任务" : "执行中的任务：\(api.activeJob)").font(.callout).textSelection(.enabled) }
                if api.running {
                    Text(api.externalWorkerActive ? "视频模型正在独立工作进程中运行" : api.sessionModel.isEmpty ? "服务尚未打开模型会话" : "服务会话已打开：\(api.sessionModel)")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Text("先用 capabilities 获取接口格式，用 models 查询模型能力、installations 读取已登记模型与 LoRA 路径。需要缩放参考图时显式调用 image_prepare，使用返回的 image_path；workflows 和 workflow_request 按图片角色组合请求，再用 plan 检查生成参数。submit 提交后保存任务 ID，用 status 查询进度，cancel 停止任务。复制接入说明后，可以交给本机 AI 编排多步创作。")
                    .font(.callout)
                Text("退出 App 会停止由此页面启动的服务；独立运行请使用 CLI 的 serve 命令。API 任务记录位于服务目录中，与创作页历史分开保存。")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = api.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            }.padding(28)
        }.task {
            if api.running { await api.refreshJobs() }
            while !Task.isCancelled { await api.refresh(); try? await Task.sleep(for: .seconds(3)) }
        }
        .onChange(of: api.running) { _, running in
            if running { Task { await api.refreshJobs(offset: 0) } }
        }
        .sheet(item: $previewJob) { job in
            VStack(alignment: .leading, spacing: 12) {
                HStack { Text("API 图片结果").font(.headline); Spacer(); Button("完成") { previewJob = nil } }
                if let url = job.outputURL { MediaPreview(path: url.path, maxPixel: 1600).frame(maxWidth: .infinity, maxHeight: .infinity) }
                Text(job.id).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
            }.padding(20).frame(width: 680, height: 620)
        }
    }

    private var jobsSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("API 任务与结果").font(.title3.weight(.semibold))
                    Spacer()
                    if api.jobsLoading { ProgressView().controlSize(.small) }
                    Button("刷新任务") { Task { await api.refreshJobs() } }
                        .disabled(!api.running || api.changing || api.jobsLoading)
                        .accessibilityIdentifier("refreshAPIJobs")
                }
                Text("按创建时间分页；图片可导入创作参考，任务记录仍保存在 API 服务中。")
                    .font(.caption).foregroundStyle(.secondary)
                if api.jobsStale {
                    Label(api.jobsUpdatedAt == nil ? (api.running ? "尚未成功读取任务列表" : "启动服务后可读取任务列表") : "显示上次读取的数据，当前列表可能已过时",
                          systemImage: "exclamationmark.circle")
                        .font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("apiJobsStale")
                }
                if let error = api.jobsError { Text("任务列表读取失败：\(error)").font(.callout).foregroundStyle(.red).textSelection(.enabled) }
                if let date = api.jobsUpdatedAt {
                    Text("上次刷新：\(date.formatted(date: .abbreviated, time: .standard))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if api.jobsTotal != nil, api.jobs.isEmpty {
                    Text(api.jobsStale ? "上次读取的这一页没有任务。" : "这一页没有任务。")
                        .font(.callout).foregroundStyle(.secondary)
                }
                ForEach(api.jobs) { job in jobRow(job) }
                if let total = api.jobsTotal {
                    HStack {
                        Button("上一页") { Task { await api.refreshJobs(offset: max(0, api.jobsOffset - LocalAPIController.jobsPageSize)) } }
                            .disabled(!api.running || api.changing || api.jobsLoading || api.jobsOffset == 0)
                            .accessibilityIdentifier("previousAPIJobs")
                        Text(api.jobs.isEmpty ? "共 \(total) 项" : "\(api.jobsOffset + 1)–\(api.jobsOffset + api.jobs.count) / \(total) 项")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("下一页") { if let next = api.jobsNextOffset { Task { await api.refreshJobs(offset: next) } } }
                            .disabled(!api.running || api.changing || api.jobsLoading || api.jobsNextOffset == nil)
                            .accessibilityIdentifier("nextAPIJobs")
                        Spacer()
                    }
                }
                if importedJobID != nil {
                    if studio.imageInputsBusy {
                        HStack {
                            Text("正在导入到创作参考…").font(.caption)
                            Button("取消导入") { studio.cancelImageImport() }.disabled(studio.cancellingImageImport)
                        }
                    } else if let message = studio.message { Text(message).font(.caption).foregroundStyle(.secondary) }
                    else { Text("已添加到创作参考。停止 API 服务后，可回到创作页编辑。") .font(.caption).foregroundStyle(.secondary) }
                }
                if let resultMessage { Text(resultMessage).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
            }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
        }.accessibilityElement(children: .contain).accessibilityIdentifier("apiJobsSection")
    }
    private func jobRow(_ job: LocalAPIJob) -> some View {
        HStack(alignment: .top, spacing: 14) {
            if job.isImage, let url = job.outputURL {
                Button { previewJob = job } label: {
                    MediaPreview(path: url.path, maxPixel: 320).frame(width: 116, height: 88)
                }.buttonStyle(.plain).help("预览图片结果")
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(studio.models.first(where: { $0.id == job.model })?.displayName ?? job.model).font(.headline)
                    Spacer()
                    Text(job.stateTitle).font(.caption.weight(.medium)).foregroundStyle(job.state == "failed" ? Color.red : Color.secondary)
                }
                HStack {
                    Text(job.createdAt.formatted(date: .abbreviated, time: .shortened))
                    if let elapsed = job.elapsedSeconds { Text("耗时 \(elapsed, specifier: "%.1f") 秒") }
                    if let phase = job.phase, ["queued", "running", "cancelling"].contains(job.state) { Text(phase) }
                }.font(.caption).foregroundStyle(.secondary)
                if !job.prompt.isEmpty { Text(job.prompt).font(.callout).lineLimit(2).textSelection(.enabled) }
                if let error = job.error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
                Text(job.id).font(.caption2.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                if let url = job.outputURL {
                    HStack {
                        if job.isImage {
                            Button("预览") { previewJob = job }
                            Button("打开图片") {
                                if !NSWorkspace.shared.open(url) { resultMessage = "无法打开结果文件，它可能已被移动或删除。" }
                            }
                            Button("导入为参考") { importResult(job) }
                                .disabled(store.busy || api.changing || studio.imageInputsBusy || !studio.supportsImageInputs)
                                .help("添加到当前创作草稿，不更改生成参数；需要图片编辑模式才能使用参考图。")
                                .accessibilityIdentifier("importAPIResult-\(job.id)")
                        }
                        Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    }.buttonStyle(.borderless)
                } else if job.state == "succeeded" {
                    Text("任务已完成，但没有可用的本机结果路径。") .font(.caption).foregroundStyle(.secondary)
                }
            }
        }.padding(.vertical, 8).accessibilityElement(children: .contain).accessibilityIdentifier("apiJob-\(job.id)")
    }
    private func importResult(_ job: LocalAPIJob) {
        guard job.isImage, let url = job.outputURL, !store.busy, !api.changing,
              !studio.imageInputsBusy, studio.supportsImageInputs else { return }
        resultMessage = nil
        if studio.beginImageImportFiles([url]) { importedJobID = job.id }
    }
}
