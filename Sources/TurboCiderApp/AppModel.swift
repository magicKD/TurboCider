import Combine
import Foundation
import AppKit
import TurboCiderKit

@MainActor
final class AppModel: ObservableObject {
    @Published var models: [TCModelDescriptor] = []
    @Published var system: TCSystemReport?
    @Published var selectedModel = "minimax-h3-turbo"
    @Published var selectedTask = "video"
    @Published var prompt = "A cinematic red fox walking through fresh snow"
    @Published var execution = TCExecutionMode.auto
    @Published var profile = TCGenerationProfile.quality
    @Published var approximation = TCApproximationMode.validated
    @Published var width = 512
    @Published var height = 512
    @Published var frames = 22
    @Published var fps = 24
    @Published var steps = 4
    @Published var seed = 42
    @Published var includeAudio = true
    @Published var engineOptionsJSON = "{}"
    @Published var inputs: [TCInputAsset] = []
    @Published var job: TCJobRecord?
    @Published var errorMessage: String?
    @Published var isLoading = false

    private let client: TurboCiderClient
    private let daemon: TurboCiderDaemon
    private var pollTask: Task<Void, Never>?

    init(
        client: TurboCiderClient = TurboCiderClient(),
        daemon: TurboCiderDaemon = TurboCiderDaemon()
    ) {
        self.client = client
        self.daemon = daemon
    }

    var selectedDescriptor: TCModelDescriptor? {
        models.first { $0.id == selectedModel }
    }

    var taskTypes: [String] {
        selectedDescriptor?.capabilities.tasks ?? [
            selectedModel.contains("flux") ? "image" : "video"
        ]
    }

    var taskType: String {
        taskTypes.contains(selectedTask) ? selectedTask : (taskTypes.first ?? "video")
    }

    var availableExecutionModes: [TCExecutionMode] {
        guard let values = selectedDescriptor?.capabilities.execution,
              !values.isEmpty else {
            return TCExecutionMode.allCases
        }
        return TCExecutionMode.allCases.filter {
            $0 == .auto || values.contains($0.rawValue)
        }
    }

    var availableProfiles: [TCGenerationProfile] {
        guard let values = selectedDescriptor?.capabilities.profiles,
              !values.isEmpty else {
            return TCGenerationProfile.allCases
        }
        return TCGenerationProfile.allCases.filter { values.contains($0.rawValue) }
    }

    func supportsInput(_ type: String) -> Bool {
        selectedDescriptor?.capabilities.inputs?.contains(type) == true
    }

    func supportsReferenceRole(_ role: String) -> Bool {
        selectedDescriptor?.capabilities.referenceRoles?.contains(role) == true
    }

    var hasMultimodalInputs: Bool {
        selectedDescriptor?.capabilities.inputs?.contains(where: { $0 != "text" }) == true
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            try await daemon.ensureRunning(client: client)
            models = try await client.models()
            system = try await client.system()
            if !models.contains(where: { $0.id == selectedModel }), let first = models.first {
                selectedModel = first.id
            }
            applyModelDefaults()
            errorMessage = nil
        } catch {
            errorMessage = "TurboCider service is unavailable: \(error.localizedDescription)"
        }
    }

    func applyModelDefaults() {
        let capabilities = selectedDescriptor?.capabilities
        selectedTask = capabilities?.tasks?.first ?? (
            selectedModel.contains("flux") ? "image" : "video"
        )
        switch selectedModel {
        case "ltx-2.5-distilled":
            width = 704; height = 480; frames = 97; fps = 24; steps = 11
            includeAudio = true
        case "flux2-klein-4b":
            width = 512; height = 512; frames = 1; fps = 24; steps = 4
            includeAudio = false
        case "fastmetal-1.3b-qad":
            width = 832; height = 480; frames = 81; fps = 16; steps = 3
            includeAudio = false
        default:
            width = 512; height = 512; frames = 22; fps = 24; steps = 4
            includeAudio = true
        }
        width = capabilities?.recommendedWidth ?? width
        height = capabilities?.recommendedHeight ?? height
        frames = capabilities?.recommendedFrames ?? frames
        fps = capabilities?.recommendedFPS ?? fps
        steps = capabilities?.recommendedSteps ?? steps
        includeAudio = capabilities?.audioOutput ?? (taskType == "video")
        if capabilities?.audioRequired == true {
            includeAudio = true
        }
        if !availableExecutionModes.contains(execution) {
            execution = .auto
        }
        if !availableProfiles.contains(profile), let first = availableProfiles.first {
            profile = first
        }
        if capabilities?.inputs?.contains(where: { $0 != "text" }) != true {
            inputs = []
        }
        applyTaskConstraints()
    }

    func applyTaskConstraints() {
        if taskType == "image" {
            frames = 1
        }
        if taskType != "video" {
            includeAudio = false
        } else if selectedDescriptor?.capabilities.audioRequired == true {
            includeAudio = true
        }
    }

    func generate() async {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "Prompt must not be empty."
            return
        }
        isLoading = true
        defer { isLoading = false }
        do {
            let optionsData = Data(engineOptionsJSON.utf8)
            let engineOptions = try JSONDecoder().decode(
                [String: TCJSONValue].self,
                from: optionsData
            )
            let request = TCGenerationRequest(
                model: selectedModel,
                task: taskType,
                prompt: prompt,
                inputs: inputs,
                output: TCOutputSpec(
                    type: taskType,
                    width: width,
                    height: height,
                    frames: frames,
                    fps: fps,
                    audio: includeAudio
                ),
                sampling: TCSamplingSpec(seed: seed, steps: steps),
                policy: TCPolicySpec(
                    execution: execution,
                    profile: profile,
                    approximation: approximation,
                    persistent: selectedDescriptor?.capabilities.recommendedPersistent
                        ?? (selectedDescriptor?.engine == "flux2")
                ),
                engineOptions: engineOptions
            )
            job = try await client.submit(request)
            errorMessage = nil
            poll(jobID: job!.id)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func addInput(
        type: String,
        role: String,
        includeEmbeddedAudio: Bool = true
    ) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = role == "reference"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        guard panel.runModal() == .OK else { return }
        for url in panel.urls {
            inputs.append(TCInputAsset(
                type: type,
                role: role,
                path: url.path,
                includeEmbeddedAudio: includeEmbeddedAudio
            ))
        }
    }

    func addVideoWithAudio() {
        let videoPanel = NSOpenPanel()
        videoPanel.allowsMultipleSelection = false
        videoPanel.canChooseDirectories = false
        videoPanel.canChooseFiles = true
        videoPanel.message = "Choose the reference video"
        guard videoPanel.runModal() == .OK, let video = videoPanel.url else { return }

        let audioPanel = NSOpenPanel()
        audioPanel.allowsMultipleSelection = false
        audioPanel.canChooseDirectories = false
        audioPanel.canChooseFiles = true
        audioPanel.message = "Choose the separate reference audio"
        guard audioPanel.runModal() == .OK, let audio = audioPanel.url else { return }

        inputs.append(TCInputAsset(
            type: "video",
            role: "reference",
            path: video.path,
            audioPath: audio.path
        ))
    }

    func removeInput(id: UUID) {
        inputs.removeAll { $0.id == id }
    }

    func cancel() async {
        guard let id = job?.id else { return }
        do {
            job = try await client.cancel(id: id)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    var outputURL: URL? {
        guard job?.state == "succeeded", let path = job?.outputPaths.first else {
            return nil
        }
        return URL(fileURLWithPath: path)
    }

    func durationLabel(_ seconds: Double?) -> String? {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return nil }
        let rounded = Int(seconds.rounded())
        if rounded < 60 { return "\(rounded)s" }
        return "\(rounded / 60)m \(rounded % 60)s"
    }

    func openOutput() {
        guard let outputURL else { return }
        NSWorkspace.shared.open(outputURL)
    }

    func revealOutput() {
        guard let outputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([outputURL])
    }

    func shutdown() {
        pollTask?.cancel()
        daemon.stop()
    }

    private func poll(jobID: String) {
        pollTask?.cancel()
        pollTask = Task {
            do {
                for try await current in client.events(id: jobID) {
                    job = current
                    if current.isTerminal { return }
                }
            } catch {
                // Older or interrupted services may not sustain an SSE stream.
                // Continue with polling so the UI still reaches terminal state.
            }
            while !Task.isCancelled {
                do {
                    let current = try await client.job(id: jobID)
                    job = current
                    if current.isTerminal { return }
                } catch {
                    errorMessage = error.localizedDescription
                    return
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }
}
