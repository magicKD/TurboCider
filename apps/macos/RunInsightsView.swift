import SwiftUI

struct RunInsightsView: View {
    let json: String
    private var metrics: RunInsights { RunInsights(json: json) }
    var body: some View {
        let info = metrics
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(info.cacheLabel, systemImage: info.promptCacheHit == true ? "arrow.triangle.2.circlepath" : "text.bubble")
                    .foregroundStyle(info.promptCacheHit == true ? Color.green : Color.secondary)
                Spacer()
                if let seconds = info.textSeconds { Text(String(format: "文本 %.3f 秒", seconds)).monospacedDigit() }
                if let seconds = info.denoiseSeconds { Text(String(format: "采样 %.2f 秒", seconds)).monospacedDigit() }
            }.font(.callout)
            if info.activeBytes != nil || info.peakBytes != nil {
                HStack(spacing: 20) {
                    if let bytes = info.activeBytes { metric("MLX 活跃内存快照", RunInsights.memory(bytes)) }
                    if let bytes = info.peakBytes { metric("MLX 峰值记录", RunInsights.memory(bytes)) }
                    Spacer()
                }
                Text("请求完成或准备完成时的采样；不含 Core ML、系统和文件缓存。实时 App 内存见侧栏。").font(.caption).foregroundStyle(.secondary)
            }
            if let calls = info.coreMLCalls, calls > 0 {
                HStack(spacing: 20) {
                    metric("Core ML 会话累计调用", String(calls))
                    if let blocks = info.coreMLBlocks { metric("已加载分区", String(blocks)) }
                    if let rows = info.coreMLRows { metric("本次输入行数", String(rows)) }
                    Spacer()
                }
                HStack(spacing: 16) {
                    if let seconds = info.coreMLLoadSeconds { Text(String(format: "分区加载 %.2f 秒", seconds)) }
                    if let seconds = info.coreMLPredictionSeconds { Text(String(format: "会话累计预测 %.2f 秒", seconds)) }
                    if let bytes = info.coreMLOutputCopyBytes { Text(String(format: "累计输出复制 %.0f 字节", bytes)) }
                }.font(.caption).monospacedDigit()
                Text("Core ML：\(info.computeUnits ?? "未报告")。调用与加载记录可确认运行路径；ANE 实际占用仍未知。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.padding(14).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityIdentifier("runInsights")
    }
    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.headline).monospacedDigit()
        }
    }
}
