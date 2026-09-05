import Foundation

public enum TCExecutionMode: String, Codable, CaseIterable, Sendable {
    case auto
    case gpu
    case gpuANE = "gpu_ane"
}

public enum TCGenerationProfile: String, Codable, CaseIterable, Sendable {
    case quality
    case balanced
    case preview
}

public enum TCApproximationMode: String, Codable, CaseIterable, Sendable {
    case exact
    case validated
    case experimental
}

public struct TCModelCapabilities: Codable, Sendable {
    public let tasks: [String]?
    public let inputs: [String]?
    public let profiles: [String]?
    public let execution: [String]?
    public let audioOutput: Bool?
    public let audioRequired: Bool?
    public let referenceRoles: [String]?
    public let modes: [String]?
    public let maxReferenceImages: Int?
    public let recommendedWidth: Int?
    public let recommendedHeight: Int?
    public let recommendedFrames: Int?
    public let recommendedFPS: Int?
    public let recommendedSteps: Int?
    public let recommendedPersistent: Bool?

    enum CodingKeys: String, CodingKey {
        case tasks, inputs, profiles, execution, modes
        case audioOutput = "audio_output"
        case audioRequired = "audio_required"
        case referenceRoles = "reference_roles"
        case maxReferenceImages = "max_reference_images"
        case recommendedWidth = "recommended_width"
        case recommendedHeight = "recommended_height"
        case recommendedFrames = "recommended_frames"
        case recommendedFPS = "recommended_fps"
        case recommendedSteps = "recommended_steps"
        case recommendedPersistent = "recommended_persistent"
    }
}

public struct TCModelPlan: Codable, Identifiable, Sendable {
    public let id: String
    public let execution: TCExecutionMode
    public let quality: TCApproximationMode
    public let profile: String
    public let production: Bool
    public let description: String
}

public struct TCModelDescriptor: Codable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let engine: String
    public let version: String
    public let capabilities: TCModelCapabilities
    public let plans: [TCModelPlan]?
}

public struct TCModelsResponse: Codable, Sendable {
    public let data: [TCModelDescriptor]
}

public struct TCJobsResponse: Codable, Sendable {
    public let data: [TCJobRecord]
}

public struct TCExecutionPlanDescriptor: Codable, Identifiable, Sendable {
    public let id: String
    public let execution: TCExecutionMode
    public let quality: TCApproximationMode
    public let profile: String
    public let production: Bool
    public let priority: Int
    public let description: String
    public let requirements: [String: TCJSONValue]
    public let environment: [String: String]
    public let metadata: [String: TCJSONValue]
}

public struct TCPlanCandidate: Codable, Sendable {
    public let plan: TCExecutionPlanDescriptor
    public let failures: [String]
    public let available: Bool
}

public struct TCPlansResponse: Codable, Sendable {
    public let data: [TCPlanCandidate]
}

public struct TCSystemModel: Codable, Identifiable, Sendable {
    public let id: String
    public let engine: String
    public let available: Bool
}

public struct TCSystemReport: Codable, Sendable {
    public let framework: String
    public let platform: String
    public let machine: String
    public let python: String
    public let device: TCDeviceInfo?
    public let deviceProfiles: [String]?
    public let models: [TCSystemModel]

    enum CodingKeys: String, CodingKey {
        case framework, platform, machine, python, device, models
        case deviceProfiles = "device_profiles"
    }
}

public struct TCDeviceInfo: Codable, Sendable {
    public let architecture: String
    public let machineModel: String
    public let chip: String
    public let gpuCores: Int?
    public let memoryGiB: Double?
    public let osVersion: String
    public let fingerprint: String

    enum CodingKeys: String, CodingKey {
        case architecture, chip, fingerprint
        case machineModel = "machine_model"
        case gpuCores = "gpu_cores"
        case memoryGiB = "memory_gib"
        case osVersion = "os_version"
    }
}

public struct TCOutputSpec: Codable, Sendable {
    public var type: String
    public var path: String?
    public var width: Int
    public var height: Int
    public var frames: Int
    public var fps: Int
    public var audio: Bool

    public init(
        type: String,
        path: String? = nil,
        width: Int = 512,
        height: Int = 512,
        frames: Int? = nil,
        fps: Int = 24,
        audio: Bool? = nil
    ) {
        self.type = type
        self.path = path
        self.width = width
        self.height = height
        self.frames = frames ?? (type == "image" ? 1 : 22)
        self.fps = fps
        self.audio = audio ?? (type == "video")
    }
}

public struct TCInputAsset: Codable, Identifiable, Sendable {
    public let id: UUID
    public var type: String
    public var role: String
    public var path: String?
    public var text: String?
    public var audioPath: String?
    public var includeEmbeddedAudio: Bool
    public var strength: Double?
    public var frameIndex: Int?

    enum CodingKeys: String, CodingKey {
        case type, role, path, text
        case audioPath = "audio_path"
        case includeEmbeddedAudio = "include_embedded_audio"
        case strength
        case frameIndex = "frame_index"
    }

    public init(
        type: String,
        role: String,
        path: String? = nil,
        text: String? = nil,
        audioPath: String? = nil,
        includeEmbeddedAudio: Bool = true,
        strength: Double? = nil,
        frameIndex: Int? = nil
    ) {
        self.id = UUID()
        self.type = type
        self.role = role
        self.path = path
        self.text = text
        self.audioPath = audioPath
        self.includeEmbeddedAudio = includeEmbeddedAudio
        self.strength = strength
        self.frameIndex = frameIndex
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = UUID()
        type = try container.decode(String.self, forKey: .type)
        role = try container.decode(String.self, forKey: .role)
        path = try container.decodeIfPresent(String.self, forKey: .path)
        text = try container.decodeIfPresent(String.self, forKey: .text)
        audioPath = try container.decodeIfPresent(String.self, forKey: .audioPath)
        includeEmbeddedAudio = try container.decodeIfPresent(Bool.self, forKey: .includeEmbeddedAudio) ?? true
        strength = try container.decodeIfPresent(Double.self, forKey: .strength)
        frameIndex = try container.decodeIfPresent(Int.self, forKey: .frameIndex)
    }
}

public struct TCSamplingSpec: Codable, Sendable {
    public var seed: Int
    public var steps: Int?
    public var guidance: Double?

    public init(seed: Int = 42, steps: Int? = nil, guidance: Double? = nil) {
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
    }
}

public struct TCPolicySpec: Codable, Sendable {
    public var execution: TCExecutionMode
    public var profile: TCGenerationProfile
    public var approximation: TCApproximationMode
    public var persistent: Bool
    public var allowFallback: Bool

    enum CodingKeys: String, CodingKey {
        case execution, profile, approximation, persistent
        case allowFallback = "allow_fallback"
    }

    public init(
        execution: TCExecutionMode = .auto,
        profile: TCGenerationProfile = .quality,
        approximation: TCApproximationMode = .validated,
        persistent: Bool = false,
        allowFallback: Bool = true
    ) {
        self.execution = execution
        self.profile = profile
        self.approximation = approximation
        self.persistent = persistent
        self.allowFallback = allowFallback
    }
}

public struct TCGenerationRequest: Codable, Sendable {
    public var model: String
    public var task: String
    public var prompt: String
    public var mode: String
    public var inputs: [TCInputAsset]
    public var output: TCOutputSpec
    public var sampling: TCSamplingSpec
    public var policy: TCPolicySpec
    public var engineOptions: [String: TCJSONValue]

    enum CodingKeys: String, CodingKey {
        case model, task, prompt, mode, inputs, output, sampling, policy
        case engineOptions = "engine_options"
    }

    public init(
        model: String,
        task: String,
        prompt: String,
        mode: String = "auto",
        inputs: [TCInputAsset] = [],
        output: TCOutputSpec,
        sampling: TCSamplingSpec = .init(),
        policy: TCPolicySpec = .init(),
        engineOptions: [String: TCJSONValue] = [:]
    ) {
        self.model = model
        self.task = task
        self.prompt = prompt
        self.mode = mode
        self.inputs = inputs
        self.output = output
        self.sampling = sampling
        self.policy = policy
        self.engineOptions = engineOptions
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        task = try container.decode(String.self, forKey: .task)
        prompt = try container.decode(String.self, forKey: .prompt)
        mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "auto"
        inputs = try container.decodeIfPresent([TCInputAsset].self, forKey: .inputs) ?? []
        output = try container.decode(TCOutputSpec.self, forKey: .output)
        sampling = try container.decodeIfPresent(TCSamplingSpec.self, forKey: .sampling) ?? .init()
        policy = try container.decodeIfPresent(TCPolicySpec.self, forKey: .policy) ?? .init()
        engineOptions = try container.decodeIfPresent(
            [String: TCJSONValue].self,
            forKey: .engineOptions
        ) ?? [:]
    }
}

public struct TCJobRecord: Codable, Identifiable, Sendable {
    public let id: String
    public let state: String
    public let planID: String?
    public let progress: Double
    public let phase: String
    public let elapsedSeconds: Double?
    public let estimatedTotalSeconds: Double?
    public let estimatedRemainingSeconds: Double?
    public let outputPaths: [String]
    public let error: String?

    enum CodingKeys: String, CodingKey {
        case id, state, progress, phase, error
        case planID = "plan_id"
        case outputPaths = "output_paths"
        case elapsedSeconds = "elapsed_seconds"
        case estimatedTotalSeconds = "estimated_total_seconds"
        case estimatedRemainingSeconds = "estimated_remaining_seconds"
    }

    public var isTerminal: Bool {
        ["succeeded", "failed", "cancelled"].contains(state)
    }
}

public struct TCAPIError: Codable, Error, LocalizedError, Sendable {
    public let error: String
    public var errorDescription: String? { error }
}
