import SwiftUI
import Darwin
import IOKit

/// Public Mach counters only. GPU/ANE scheduling is not inferred from CPU load.
@MainActor final class ResourceMonitor: ObservableObject {
    @Published var cpuPercent: Double?
    @Published var residentBytes: UInt64?
    @Published var gpuPercent: Double?
    private var previous: [UInt32]?
    func sample() {
        gpuPercent = nil
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        if service != 0 {
            defer { IOObjectRelease(service) }
            if let raw = IORegistryEntryCreateCFProperty(service, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? [String: Any],
               let value = raw["Device Utilization %"] as? NSNumber {
                let percent = value.doubleValue
                if percent.isFinite && (0...100).contains(percent) { gpuPercent = percent }
            }
        }
        var cpu = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
        let port = mach_host_self()
        defer { mach_port_deallocate(mach_task_self_, port) }
        let status = withUnsafeMutablePointer(to: &cpu) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics(port, HOST_CPU_LOAD_INFO, $0, &count) }
        }
        if status == KERN_SUCCESS {
            let ticks = [cpu.cpu_ticks.0, cpu.cpu_ticks.1, cpu.cpu_ticks.2, cpu.cpu_ticks.3]
            if let previous {
                let deltas = zip(ticks, previous).map { UInt64($0 &- $1) }
                let total = deltas.reduce(0, +)
                cpuPercent = total > 0 ? 100 * Double(total - deltas[Int(CPU_STATE_IDLE)]) / Double(total) : nil
            }
            previous = ticks
        } else { cpuPercent = nil; previous = nil }
        var info = mach_task_basic_info()
        var infoCount = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &infoCount) }
        }
        residentBytes = result == KERN_SUCCESS ? UInt64(info.resident_size) : nil
    }
}

struct ResourceMonitorView: View {
    @StateObject private var monitor = ResourceMonitor()
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("实时资源").font(.caption.weight(.medium))
            Text(monitor.cpuPercent.map { String(format: "整机 CPU  %.0f%%", $0) } ?? "整机 CPU  —")
            Text(monitor.residentBytes.map { String(format: "App 驻留内存  %.2f GB", Double($0) / 1_073_741_824) } ?? "App 驻留内存  —")
            Text(monitor.gpuPercent.map { String(format: "整机 GPU  %.0f%%", $0) } ?? "整机 GPU  不可用")
                .help("Apple GPU 驱动计数器；包含其他 App 负载，系统升级后可能不可用，不代表本任务占用。")
            Text("ANE 占用  未知")
                .help("当前未获得可靠的 ANE 实时占用百分比；执行路径和 Core ML 调用不等于芯片利用率。驻留内存也不等于 MLX 分配内存。")
        }.font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        .task {
            while !Task.isCancelled {
                monitor.sample()
                do { try await Task.sleep(for: .seconds(2)) } catch { break }
            }
        }
    }
}
