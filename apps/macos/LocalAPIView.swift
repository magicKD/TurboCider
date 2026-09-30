import SwiftUI
import AppKit

struct LocalAPIView: View {
    @ObservedObject var api: LocalAPIController
    @ObservedObject var store: NativeJobStore
    private var connectionInstructions: String {
        """
        使用本机 TurboCider 的 Unix socket API：
        Socket: \(api.socketPath)
        先在 App 的“本地 API”页启动服务。每个连接发送一个 UTF-8 JSON 对象，以换行结尾；每个连接只处理一次请求。
        首先发送 {"action":"capabilities"} 获取操作、字段类型和限制；再发送 {"action":"models"} 查询模型能力（不是已安装模型列表）。
        响应格式为 {"ok":true,"result":...} 或 {"ok":false,"error":"..."}。
        使用 plan 检查完整生成请求，使用 submit 提交 model_path 和 request，保存返回的 id；用 status 查询直到 succeeded/failed/cancelled/interrupted。
        模型、LoRA、参考图与输出均使用本机路径；参考图按 inputs 顺序编号，每步使用独立输出路径。上一任务 succeeded 后再将其输出用于下一步。
        提交后若连接中断，先用 jobs 核对是否已入队，不自动重复提交。等待超时不会取消任务；需要停止时显式调用 cancel 并继续查询状态。
        API 历史与创作页分开保存；退出 App 会停止它启动的服务。所有工作只使用已有本地文件，不下载模型。
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
                Text("先用 capabilities 获取接口格式，再用 models 查询模型能力、plan 检查生成参数。submit 提交后保存任务 ID，用 status 查询进度，cancel 停止任务。复制接入说明后，可以交给本机 AI 编排多步创作。")
                    .font(.callout)
                Text("退出 App 会停止由此页面启动的服务；独立运行请使用 CLI 的 serve 命令。API 任务记录位于服务目录中，与创作页历史分开保存。")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = api.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            }.padding(28)
        }.task { while !Task.isCancelled { await api.refresh(); try? await Task.sleep(for: .seconds(3)) } }
    }
}
