import SwiftUI
import AppKit

struct LocalAPIView: View {
    @ObservedObject var api: LocalAPIController
    @ObservedObject var store: NativeJobStore
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
                            Button("打开日志目录") { NSWorkspace.shared.open(api.directory) }
                            if api.running { Button("刷新状态") { Task { await api.refresh() } }; Text("历史任务 \(api.jobCount)").font(.caption) }
                        }
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                }
                Text("调用方式").font(.title2)
                Text("此服务使用 Unix socket JSON RPC，适合当前 Mac 的本地进程；不监听公网端口。每个连接发送一个以换行结尾的 JSON 对象。").foregroundStyle(.secondary)
                Text("turbocider rpc \(api.socketPath) rpc.json\n\nrpc.json:\n{\"action\":\"models\"}")
                    .font(.system(.body, design: .monospaced)).textSelection(.enabled).padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading).background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
                if api.running { Text(api.activeJob.isEmpty ? "当前没有执行中的 API 任务" : "执行中的任务：\(api.activeJob)").font(.callout).textSelection(.enabled) }
                if api.running {
                    Text(api.externalWorkerActive ? "视频模型正在独立工作进程中运行" : api.sessionModel.isEmpty ? "服务尚未打开模型会话" : "服务会话已打开：\(api.sessionModel)")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Text("支持 models、doctor、service_status、plan、submit、status、jobs 和 cancel。submit 提交模型路径及生成请求；status 返回真实阶段与进度。服务使用持久队列，相同模型与提示词可复用缓存。")
                    .font(.callout)
                Text("退出 App 会停止由此页面启动的服务；独立运行请使用 CLI 的 serve 命令。API 任务记录位于服务目录中，与创作页历史分开保存。")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = api.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            }.padding(28)
        }.task { while !Task.isCancelled { await api.refresh(); try? await Task.sleep(for: .seconds(3)) } }
    }
}
