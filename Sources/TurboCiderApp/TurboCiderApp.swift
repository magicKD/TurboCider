import Dispatch
import Foundation
import SwiftUI
import TurboCiderKit

@main
struct TurboCiderApp: App {
    init() {
        if CommandLine.arguments.contains("--smoke") {
            let daemon = TurboCiderDaemon()
            let client = TurboCiderClient()
            Task.detached {
                do {
                    try await daemon.ensureRunning(client: client)
                    let models = try await client.models()
                    let system = try await client.system()
                    guard !models.isEmpty else {
                        throw TurboCiderDaemonError.launchFailed(
                            "model registry returned no models"
                        )
                    }
                    guard system.framework == "TurboCider" else {
                        throw TurboCiderDaemonError.launchFailed(
                            "unexpected system response"
                        )
                    }
                    var report: [String: Any] = [
                        "app": "TurboCiderApp",
                        "status": "ok",
                        "api": client.baseURL.absoluteString,
                        "models": models.count,
                        "available_models": system.models.filter(\.available).count,
                    ]
                    if let smokeModel = ProcessInfo.processInfo.environment[
                        "TURBOCIDER_SMOKE_MODEL"
                    ] {
                        guard let descriptor = models.first(where: { $0.id == smokeModel }) else {
                            throw TurboCiderDaemonError.launchFailed(
                                "smoke model \(smokeModel) is not registered"
                            )
                        }
                        let task = descriptor.capabilities.tasks?.first ?? "video"
                        let request = TCGenerationRequest(
                            model: descriptor.id,
                            task: task,
                            prompt: "TurboCider App progress smoke",
                            output: TCOutputSpec(
                                type: task,
                                width: descriptor.capabilities.recommendedWidth ?? 32,
                                height: descriptor.capabilities.recommendedHeight ?? 32,
                                frames: descriptor.capabilities.recommendedFrames,
                                fps: descriptor.capabilities.recommendedFPS ?? 24,
                                audio: descriptor.capabilities.audioOutput ?? false
                            ),
                            sampling: TCSamplingSpec(
                                seed: 42,
                                steps: descriptor.capabilities.recommendedSteps
                            ),
                            policy: TCPolicySpec(
                                execution: .gpu,
                                approximation: .exact
                            )
                        )
                        let submitted = try await client.submit(request)
                        var terminal = submitted
                        var sawIntermediateProgress = false
                        var sawETA = false
                        for try await update in client.events(id: submitted.id) {
                            terminal = update
                            if update.progress > 0 && update.progress < 1 {
                                sawIntermediateProgress = true
                            }
                            if update.estimatedRemainingSeconds != nil {
                                sawETA = true
                            }
                        }
                        guard terminal.state == "succeeded" else {
                            throw TurboCiderDaemonError.launchFailed(
                                "smoke generation ended as \(terminal.state)"
                            )
                        }
                        guard sawIntermediateProgress else {
                            throw TurboCiderDaemonError.launchFailed(
                                "smoke generation exposed no intermediate progress"
                            )
                        }
                        guard sawETA else {
                            throw TurboCiderDaemonError.launchFailed(
                                "smoke generation exposed no ETA"
                            )
                        }
                        report["job"] = terminal.id
                        report["progress"] = terminal.progress
                        report["phase"] = terminal.phase
                        report["eta_observed"] = sawETA
                    }
                    let data = try JSONSerialization.data(
                        withJSONObject: report,
                        options: [.sortedKeys]
                    )
                    print(String(decoding: data, as: UTF8.self))
                    daemon.stop()
                    fflush(stdout)
                    exit(0)
                } catch {
                    daemon.stop()
                    fputs("TurboCiderApp smoke failed: \(error.localizedDescription)\n", stderr)
                    exit(1)
                }
            }
            dispatchMain()
        }
    }

    var body: some Scene {
        WindowGroup("TurboCider") {
            ContentView()
                .frame(minWidth: 780, minHeight: 580)
        }
        .windowStyle(.titleBar)
    }
}
