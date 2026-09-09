import Foundation
import Combine

@MainActor final class TensorCacheController: ObservableObject {
    @Published private(set) var report: TensorCacheReport?
    @Published private(set) var retentionDays = 0
    @Published private(set) var diagnosticDirectories: [String] = []
    @Published private(set) var busy = false
    @Published var message: String?
    func refresh() async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            retentionDays = try LibraryTool.decode(TensorCacheSettings.self, from: await LibraryTool.run(["cache", "settings"])).retentionDays
            diagnosticDirectories = try LibraryTool.decode(DiagnosticTensorRegistry.self, from: await LibraryTool.run(["cache", "dump-directories"])).directories
            report = try LibraryTool.decode(TensorCacheReport.self, from: await LibraryTool.run(["cache", "inventory"]))
        } catch { message = error.localizedDescription }
    }
    func retain(_ days: Int) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            retentionDays = try LibraryTool.decode(TensorCacheSettings.self, from: await LibraryTool.run(["cache", "retain", String(days)])).retentionDays
            message = days == 0 ? "已关闭自动清理。" : "App 运行期间会定期清理生成超过 \(days) 天的已登记张量。"
        } catch { message = error.localizedDescription }
    }
    func registerDumps(_ directory: URL, remove: Bool = false) async {
        guard !busy else { return }; busy = true; defer { busy = false }
        do {
            diagnosticDirectories = try LibraryTool.decode(DiagnosticTensorRegistry.self, from:
                await LibraryTool.run(["cache", remove ? "forget-dumps" : "register-dumps", directory.path])).directories
            report = try LibraryTool.decode(TensorCacheReport.self, from: await LibraryTool.run(["cache", "inventory"]))
            message = remove ? "已停止管理此目录，文件保留。" : "已登记输出张量目录，未删除文件；这些张量将使用上方保留期限。"
        } catch { message = error.localizedDescription }
    }
    func prune(store: NativeJobStore, days: Int) async {
        guard !busy, !store.busy, !store.externalServiceActive, !store.resolvingAcceleration else { return }
        busy = true; defer { busy = false }
        do {
            report = try LibraryTool.decode(TensorCacheReport.self, from: await store.pruneTensorCache(days: days))
            message = "已清理 \(report!.removedEntries) 项，释放 \(ByteCountFormatter.string(fromByteCount: report!.removedBytes, countStyle: .file))。"
        } catch { message = error.localizedDescription }
    }
    func automaticSweep(store: NativeJobStore) async {
        guard !busy, !store.busy, !store.externalServiceActive, !store.resolvingAcceleration else { return }
        do {
            let settings = try LibraryTool.decode(TensorCacheSettings.self, from: await LibraryTool.run(["cache", "settings"]))
            if settings.retentionDays > 0 { await prune(store: store, days: settings.retentionDays) }
        } catch { message = error.localizedDescription }
    }
}
