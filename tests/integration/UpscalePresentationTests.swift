import AppKit
import SwiftUI

/// Render only our own views; this is not OS screen capture or GUI-click QA.
@main struct UpscalePresentationTests {
    @MainActor static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw NativeFailure(message: "OUTPUT_DIRECTORY required") }
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        guard !FileManager.default.fileExists(atPath: root.path) else { throw NativeFailure(message: "fresh output required") }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        _ = NSApplication.shared
        func render<V: View>(_ name: String, width: CGFloat, height: CGFloat, _ content: V) throws {
            let host = NSHostingView(rootView: content.padding(14).frame(width: width, height: height)
                .background(Color(nsColor: .windowBackgroundColor)).environment(\.colorScheme,.light))
            host.frame = NSRect(x: 0, y: 0, width: width, height: height)
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { throw NativeFailure(message: "cannot render \(name)") }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            guard let png = bitmap.representation(using: .png, properties: [:]) else { throw NativeFailure(message: "cannot encode \(name)") }
            try png.write(to: root.appendingPathComponent(name + ".png"))
        }
        let studio = StudioState(directory: root.appendingPathComponent("state"))
        _ = studio.activateSeedVR2Upscaling(kind: "video")
        let source = StudioAsset(path: "metadata-only.mp4",name: "source.mp4",width: 512,height: 512,kind: "video",frames: 97,fps: 24)
        studio.draft.assets = [source]; studio.draft.initImageID = source.id
        studio.draft.width = 1024;studio.draft.height = 1024;studio.draft.frames = 97
        studio.draft.modelPaths["seedvr2-3b"] = "metadata-only"
        try render("video-header",width: 760,height: 100,
                   UpscaleWorkspaceHeader(kind: .constant("video"),modelID: .constant("seedvr2-3b"),locked: false))
        try render("image-header",width: 760,height: 100,
                   UpscaleWorkspaceHeader(kind: .constant("image"),modelID: .constant("real-esrgan-x2plus"),locked: false))
        try render("settings-speed",width: 760,height: 220,
                   SeedVR2UpscaleControls(studio: studio,locked: false,manageModels: {}))
        studio.applySeedVR2MemoryAdvice()
        try render("settings-memory",width: 760,height: 220,
                   SeedVR2UpscaleControls(studio: studio,locked: false,manageModels: {}))
        var request = NativeRequest(prompt: "",output: "metadata-only.mp4")
        request.model = "seedvr2-3b";request.operation = "video.upscale";request.width = 1024;request.height = 1024
        request.frames = 97;request.fps = 24
        for (phase,completed,total) in [("seedvr2_encode_progress",49,97),("seedvr2_dit",17,32),("seedvr2_decode_progress",30,97)] {
            let job = NativeJob(id: UUID(),createdAt: Date().addingTimeInterval(-160),request: request,state: "running",phase: phase,
                                completed: completed,total: total,elapsed: 160,phaseStartedElapsed: 100)
            try render(phase,width: 480,height: 170,SeedVR2RunProgressView(job: job,cancel: {}))
        }
        print("PASS own-view offscreen render: model headers, runtime preferences and narrow 97-frame stage progress; no GUI clicks")
    }
}
