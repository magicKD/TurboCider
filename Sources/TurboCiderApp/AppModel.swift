import Combine
import Foundation
import AppKit
import TurboCiderKit
import UniformTypeIdentifiers

@MainActor
final class AppModel: ObservableObject {
    @Published var models: [TCModelDescriptor] = []
    @Published var system: TCSystemReport?
    @Published var selectedModel = "minimax-h3-turbo"
    @Published var selectedTask = "video"
    @Published var selectedMode = "auto"
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
    @Published var imageStrength = 0.75
    @Published var engineOptionsJSON = "{}"
    @Published var inputs: [TCInputAsset] = []
    @Published var job: TCJobRecord?
    @Published var errorMessage: String?
    @Published var isLoading = false

    private let client: TurboCiderClient
    private let daemon: TurboCiderDaemon
    private var pollTask: Task<Void, Never>?
    private var stagedInputs: Set<String> = []

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

    var availableModes: [String] {
        let declared = selectedDescriptor?.capabilities.modes ?? []
        guard !declared.isEmpty else { return ["auto"] }
        let imageModes: Set<String> = ["text_to_image", "image_to_image", "image_edit"]
        let videoModes: Set<String> = ["text_to_video", "image_to_video", "keyframe_interpolation"]
        let taskModes = declared.filter {
            (taskType == "image" ? imageModes : videoModes).contains($0)
        }
        if !taskModes.isEmpty { return taskModes }
        return ["auto"]
    }

    var hasDeclaredModes: Bool {
        selectedDescriptor?.capabilities.modes?.isEmpty == false
    }

    var modeType: String {
        availableModes.contains(selectedMode) ? selectedMode : (availableModes.first ?? "auto")
    }

    var maxReferenceImages: Int {
        max(1, selectedDescriptor?.capabilities.maxReferenceImages ?? 1)
    }

    var modeHelp: String {
        switch modeType {
        case "image_to_image":
            return "Uses one init image. Strength controls how far generation may move away from it."
        case "image_edit":
            return "Uses one or more ordered reference images with the FLUX edit pipeline."
        case "image_to_video":
            return "Anchors the generated video to the selected first frame."
        case "keyframe_interpolation":
            return "Generates motion between the selected first and last keyframes."
        default:
            return taskType == "image" ? "Generates an image from text only." : "Generates a video from text only."
        }
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
        selectedMode = availableModes.first ?? "auto"
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
        if !availableModes.contains(selectedMode) {
            selectedMode = availableModes.first ?? "auto"
        }
        applyModeConstraints()
    }

    func applyModeConstraints() {
        guard hasDeclaredModes else { return }
        let acceptedRoles: Set<String>
        switch modeType {
        case "image_to_image":
            acceptedRoles = ["init_image"]
        case "image_edit":
            acceptedRoles = ["reference"]
        case "image_to_video":
            acceptedRoles = ["first_frame"]
        case "keyframe_interpolation":
            acceptedRoles = ["first_frame", "last_frame"]
        default:
            acceptedRoles = []
        }
        let removed = inputs.filter { !acceptedRoles.contains($0.role) }
        inputs.removeAll { !acceptedRoles.contains($0.role) }
        removed.forEach { removeStagedFile($0.path) }
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
                mode: modeType,
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
        includeEmbeddedAudio: Bool = true,
        strength: Double? = nil,
        frameIndex: Int? = nil
    ) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = role == "reference"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        switch type {
        case "image": panel.allowedContentTypes = [.image]
        case "video": panel.allowedContentTypes = [.movie]
        case "audio": panel.allowedContentTypes = [.audio]
        default: break
        }
        guard panel.runModal() == .OK else { return }
        let availableSlots = role == "reference"
            ? max(0, maxReferenceImages - inputs.filter { $0.role == "reference" }.count)
            : 1
        if availableSlots == 0 {
            errorMessage = "This model accepts at most \(maxReferenceImages) reference image(s)."
            return
        }
        do {
            let selected = Array(panel.urls.prefix(availableSlots))
            if role == "reference" && selected.count < panel.urls.count {
                errorMessage = "Only the first \(selected.count) file(s) were added; this model accepts at most \(maxReferenceImages) references."
            } else {
                errorMessage = nil
            }
            for url in selected {
                let staged = try stageInputFile(url)
                let asset = TCInputAsset(
                    type: type,
                    role: role,
                    path: staged.path,
                    includeEmbeddedAudio: includeEmbeddedAudio,
                    strength: strength,
                    frameIndex: frameIndex
                )
                if role == "reference" {
                    inputs.append(asset)
                } else {
                    let replaced = inputs.filter { $0.role == role }
                    inputs.removeAll { $0.role == role }
                    replaced.forEach { removeStagedFile($0.path) }
                    inputs.append(asset)
                }
            }
        } catch {
            errorMessage = "Could not prepare input media: \(error.localizedDescription)"
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

        do {
            let stagedVideo = try stageInputFile(video)
            let stagedAudio = try stageInputFile(audio)
            inputs.append(TCInputAsset(
                type: "video",
                role: "reference",
                path: stagedVideo.path,
                audioPath: stagedAudio.path
            ))
        } catch {
            errorMessage = "Could not prepare input media: \(error.localizedDescription)"
        }
    }

    func removeInput(id: UUID) {
        if let input = inputs.first(where: { $0.id == id }) {
            removeStagedFile(input.path)
            removeStagedFile(input.audioPath)
        }
        inputs.removeAll { $0.id == id }
    }

    func updateImageStrength(_ value: Double) {
        imageStrength = value
        for index in inputs.indices where ["init_image", "first_frame"].contains(inputs[index].role) {
            inputs[index].strength = value
        }
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

    private func stageInputFile(_ source: URL) throws -> URL {
        let manager = FileManager.default
        let support = manager.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        )[0].appendingPathComponent("TurboCider/inputs", isDirectory: true)
        try manager.createDirectory(at: support, withIntermediateDirectories: true)
        let accessed = source.startAccessingSecurityScopedResource()
        defer {
            if accessed { source.stopAccessingSecurityScopedResource() }
        }
        let suffix = source.pathExtension.isEmpty ? "" : "." + source.pathExtension
        let destination = support.appendingPathComponent(UUID().uuidString + suffix)
        try manager.copyItem(at: source, to: destination)
        stagedInputs.insert(destination.path)
        return destination
    }

    private func removeStagedFile(_ path: String?) {
        guard let path, stagedInputs.remove(path) != nil else { return }
        try? FileManager.default.removeItem(atPath: path)
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
