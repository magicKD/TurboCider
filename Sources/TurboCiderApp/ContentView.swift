import AVKit
import AppKit
import SwiftUI
import TurboCiderKit

struct ContentView: View {
    @StateObject private var model = AppModel()

    var body: some View {
        NavigationSplitView {
            List(selection: $model.selectedModel) {
                Section("Models") {
                    ForEach(model.models) { item in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(item.name)
                            Text(item.engine.uppercased())
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .tag(item.id)
                    }
                }
            }
            .frame(minWidth: 210)
        } detail: {
            Form {
                if let system = model.system {
                    Section("System") {
                        HStack {
                            Text(system.device?.chip ?? system.machine.uppercased())
                            if let cores = system.device?.gpuCores {
                                Text("\(cores)-core GPU").foregroundStyle(.secondary)
                            }
                            if let memory = system.device?.memoryGiB {
                                Text("\(Int(memory.rounded())) GB").foregroundStyle(.secondary)
                            }
                            Text(system.python).foregroundStyle(.secondary)
                            Spacer()
                            ForEach(system.models) { item in
                                Label(
                                    item.id.replacingOccurrences(of: "-", with: " "),
                                    systemImage: item.available
                                        ? "checkmark.circle.fill" : "xmark.circle.fill"
                                )
                                .foregroundStyle(item.available ? .green : .red)
                                .font(.caption)
                            }
                        }
                    }
                }

                Section("Prompt") {
                    TextEditor(text: $model.prompt)
                        .font(.body)
                        .frame(minHeight: 110)
                }

                if model.hasMultimodalInputs {
                    if model.hasDeclaredModes {
                        Section("Generation Mode") {
                            Picker("Mode", selection: $model.selectedMode) {
                                ForEach(model.availableModes, id: \.self) { mode in
                                    Text(modeLabel(mode)).tag(mode)
                                }
                            }
                            .pickerStyle(.segmented)
                            Text(model.modeHelp)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        if model.modeType != "text_to_image" && model.modeType != "text_to_video" {
                            Section("Image Inputs") {
                                HStack {
                                    switch model.modeType {
                                    case "image_to_image":
                                        Button("Choose Init Image") {
                                            model.addInput(
                                                type: "image", role: "init_image",
                                                strength: model.imageStrength
                                            )
                                        }
                                    case "image_edit":
                                        Button("Add Reference Images") {
                                            model.addInput(type: "image", role: "reference")
                                        }
                                        Text("\(model.inputs.count)/\(model.maxReferenceImages)")
                                            .foregroundStyle(.secondary)
                                    case "image_to_video":
                                        Button("Choose First Frame") {
                                            model.addInput(
                                                type: "image", role: "first_frame",
                                                strength: model.imageStrength, frameIndex: 0
                                            )
                                        }
                                    case "keyframe_interpolation":
                                        Button("Choose First Frame") {
                                            model.addInput(
                                                type: "image", role: "first_frame",
                                                strength: model.imageStrength, frameIndex: 0
                                            )
                                        }
                                        Button("Choose Last Frame") {
                                            model.addInput(
                                                type: "image", role: "last_frame",
                                                frameIndex: max(0, model.frames - 1)
                                            )
                                        }
                                    default:
                                        EmptyView()
                                    }
                                }
                                if ["image_to_image", "image_to_video"].contains(model.modeType) {
                                    HStack {
                                        Text("Strength")
                                        Slider(
                                            value: Binding(
                                                get: { model.imageStrength },
                                                set: { model.updateImageStrength($0) }
                                            ),
                                            in: 0...1,
                                            step: 0.05
                                        )
                                        Text(model.imageStrength.formatted(.number.precision(.fractionLength(2))))
                                            .monospacedDigit()
                                            .frame(width: 40)
                                    }
                                }
                                inputRows
                            }
                        }
                    } else {
                        Section("Multimodal Inputs") {
                            HStack {
                                if model.supportsInput("image") && model.supportsReferenceRole("first_frame") {
                                    Button("First Frame") { model.addInput(type: "image", role: "first_frame", frameIndex: 0) }
                                }
                                if model.supportsInput("image") && model.supportsReferenceRole("last_frame") {
                                    Button("Last Frame") { model.addInput(type: "image", role: "last_frame", frameIndex: max(0, model.frames - 1)) }
                                }
                                if model.supportsInput("image") && model.supportsReferenceRole("reference") {
                                    Button("Reference Image") { model.addInput(type: "image", role: "reference") }
                                }
                                if model.supportsInput("video") {
                                    Button("Reference Video") { model.addInput(type: "video", role: "reference") }
                                }
                                if model.supportsInput("audio") {
                                    Button("Reference Audio") { model.addInput(type: "audio", role: "reference") }
                                }
                            }
                            inputRows
                        }
                    }
                }

                Section("Generation") {
                    if model.taskTypes.count > 1 {
                        Picker("Task", selection: $model.selectedTask) {
                            ForEach(model.taskTypes, id: \.self) { Text($0).tag($0) }
                        }
                    }
                    Picker("Execution", selection: $model.execution) {
                        ForEach(model.availableExecutionModes, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    Picker("Profile", selection: $model.profile) {
                        ForEach(model.availableProfiles, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    Picker("Approximation", selection: $model.approximation) {
                        ForEach(TCApproximationMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    Grid(alignment: .leading) {
                        GridRow {
                            Text("Width"); TextField("Width", value: $model.width, format: .number)
                            Text("Height"); TextField("Height", value: $model.height, format: .number)
                        }
                        GridRow {
                            Text("Frames"); TextField("Frames", value: $model.frames, format: .number)
                            Text("FPS"); TextField("FPS", value: $model.fps, format: .number)
                        }
                        GridRow {
                            Text("Steps"); TextField("Steps", value: $model.steps, format: .number)
                            Text("Seed"); TextField("Seed", value: $model.seed, format: .number)
                        }
                        Toggle("Audio", isOn: $model.includeAudio)
                            .disabled(
                                model.taskType != "video"
                                    || model.selectedDescriptor?.capabilities.audioOutput == false
                                    || model.selectedDescriptor?.capabilities.audioRequired == true
                            )
                    }
                }

                DisclosureGroup("Advanced Engine Options") {
                    Text("Namespaced JSON passed directly to the selected engine adapter, for example {\"h3\":{\"super\":true}}.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextEditor(text: $model.engineOptionsJSON)
                        .font(.system(.body, design: .monospaced))
                        .frame(minHeight: 80)
                }

                Section("Job") {
                    if let job = model.job {
                        ProgressView(value: job.progress) {
                            Text("\(job.state) · \(job.phase)")
                        } currentValueLabel: {
                            Text("\(Int((job.progress * 100).rounded()))%")
                        }
                        HStack(spacing: 16) {
                            if let elapsed = model.durationLabel(job.elapsedSeconds) {
                                Label("Elapsed \(elapsed)", systemImage: "clock")
                            }
                            if !job.isTerminal,
                               let remaining = model.durationLabel(job.estimatedRemainingSeconds) {
                                Label("About \(remaining) remaining", systemImage: "hourglass")
                            }
                            if job.isTerminal,
                               let total = model.durationLabel(job.elapsedSeconds) {
                                Label("Finished in \(total)", systemImage: "checkmark.circle")
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        if let plan = job.planID { Text("Plan: \(plan)").font(.caption) }
                        ForEach(job.outputPaths, id: \.self) { Text($0).font(.caption).textSelection(.enabled) }
                        if let output = model.outputURL {
                            OutputPreview(url: output)
                                .frame(minHeight: 180, maxHeight: 320)
                            HStack {
                                Button("Open") { model.openOutput() }
                                Button("Show in Finder") { model.revealOutput() }
                            }
                        }
                        if !job.isTerminal {
                            Button("Cancel", role: .destructive) { Task { await model.cancel() } }
                        }
                    } else {
                        Text("No active job").foregroundStyle(.secondary)
                    }
                }

                if let error = model.errorMessage {
                    Section("Status") { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                }

                HStack {
                    Spacer()
                    Button("Generate") { Task { await model.generate() } }
                        .keyboardShortcut(.return, modifiers: [.command])
                        .disabled(model.isLoading || model.models.isEmpty)
                }
            }
            .formStyle(.grouped)
            .padding()
        }
        .task { await model.load() }
        .onChange(of: model.selectedModel) { _ in model.applyModelDefaults() }
        .onChange(of: model.selectedTask) { _ in model.applyTaskConstraints() }
        .onChange(of: model.selectedMode) { _ in model.applyModeConstraints() }
        .onDisappear { model.shutdown() }
    }

    private func modeLabel(_ mode: String) -> String {
        switch mode {
        case "text_to_image": return "Text to Image"
        case "image_to_image": return "Image to Image"
        case "image_edit": return "Multi-Image Edit"
        case "text_to_video": return "Text to Video"
        case "image_to_video": return "Image to Video"
        case "keyframe_interpolation": return "Keyframes"
        default: return mode.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private func roleLabel(_ role: String) -> String {
        switch role {
        case "init_image": return "Init image"
        case "first_frame": return "First frame"
        case "last_frame": return "Last frame"
        case "reference": return "Reference image"
        default: return role.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    @ViewBuilder
    private var inputRows: some View {
        ForEach(model.inputs) { input in
            HStack(spacing: 12) {
                if input.type == "image", let path = input.path,
                   let image = NSImage(contentsOfFile: path) {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 64, height: 48)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(roleLabel(input.role))
                        .font(.headline)
                    Text(URL(fileURLWithPath: input.path ?? "").lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    if let frameIndex = input.frameIndex {
                        Text("Frame \(frameIndex)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Button(role: .destructive) {
                    model.removeInput(id: input.id)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
            }
        }
    }
}

private struct OutputPreview: View {
    let url: URL

    var body: some View {
        if ["png", "jpg", "jpeg", "heic"].contains(url.pathExtension.lowercased()),
           let image = NSImage(contentsOf: url) {
            Image(nsImage: image)
                .resizable()
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 8))
        } else if ["mp4", "mov", "m4v"].contains(url.pathExtension.lowercased()) {
            VideoPlayer(player: AVPlayer(url: url))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        } else {
            VStack(spacing: 8) {
                Image(systemName: "doc")
                    .font(.largeTitle)
                Text("Preview unavailable")
                    .font(.headline)
                Text(url.lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}
