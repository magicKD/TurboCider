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
                    Section("Multimodal Inputs") {
                        HStack {
                            if model.supportsInput("image") && model.supportsReferenceRole("first_frame") {
                                Button("First Frame") { model.addInput(type: "image", role: "first_frame") }
                            }
                            if model.supportsInput("image") && model.supportsReferenceRole("last_frame") {
                                Button("Last Frame") { model.addInput(type: "image", role: "last_frame") }
                            }
                            if model.supportsInput("image") {
                                Button("Reference Image") { model.addInput(type: "image", role: "reference") }
                            }
                            if model.supportsInput("video") {
                                Button("Reference Video") { model.addInput(type: "video", role: "reference") }
                                Button("Silent Video") {
                                    model.addInput(
                                        type: "video",
                                        role: "reference",
                                        includeEmbeddedAudio: false
                                    )
                                }
                            }
                            if model.supportsInput("video") && model.supportsInput("audio") {
                                Button("Video + Audio") { model.addVideoWithAudio() }
                            }
                            if model.supportsInput("audio") {
                                Button("Reference Audio") { model.addInput(type: "audio", role: "reference") }
                            }
                        }
                        ForEach(model.inputs) { input in
                            HStack {
                                Text("\(input.role): \(input.path ?? "")")
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer()
                                Button(role: .destructive) { model.removeInput(id: input.id) } label: {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.borderless)
                            }
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
        .onDisappear { model.shutdown() }
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
