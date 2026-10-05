import Foundation
import CryptoKit
import CoreFoundation
import Darwin

/// A pinned local experiment. Intent is typed, execution is isolated per child,
/// and only the child's receipt can describe the route that actually ran.
enum RuntimeImageWorker {
    static let profileID = "private-w8a8-hadamard-512-v1"
    struct Options: Codable, Sendable, Equatable {
        let profile_id: String
        let descriptor_path: String
        let cache_dir: String
    }
    struct Receipt: Codable, Sendable, Equatable {
        let backend: String
        let selected_backend: String?
        let actual_execution: String
        let partial_fallback: Bool
        let data_path: String
        let partition_axis: String
        let ane_channels: Int
        let fallback_reason: String
        let hybrid_blocks: Int
        let gpu_blocks: Int
        let fallback_blocks: Int
        var label: String {
            if backend == "gpu" { return "GPU · Runtime 实验回退：\(fallback_reason)" }
            let suffix = partial_fallback ? " · \(fallback_blocks) 块回退 GPU：\(fallback_reason)" : ""
            return "GPU + Private ANE · Hadamard W8A8 数据路径（本机实验）\(suffix)"
        }
    }
    struct Prepared: Sendable {
        let reference: PublicImageWorker.Reference
        let input: Data
        let inputURL: URL
        let request: NativeRequestV2
        let options: Options
    }
    struct Verified: Sendable {
        let status: String
        let result: Data?
        let artifact: WorkerTerminalEnvelope.Artifact?
        let receipt: Receipt?
        let errorCode: String?
        let errorMessage: String?
    }
    private static func invalid(_ detail: String = "请求或执行回执不匹配") -> NativeFailure {
        NativeFailure(message: "runtime_worker_invalid: \(detail)。")
    }
    static func hardwareEligible(systemJSON: String = NativeEngine.system()) -> Bool {
        guard let system = (try? JSONSerialization.jsonObject(with: Data(systemJSON.utf8))) as? [String: Any],
              system["gpu"] as? String == "Apple M4 Pro",
              let bytes = system["physical_memory_bytes"] as? NSNumber,
              CFGetTypeID(bytes) != CFBooleanGetTypeID(), bytes.uint64Value == UInt64(48) << 30 else { return false }
        return true
    }
    static func physicalPath(_ path: String) -> String? {
        guard let resolved = realpath(path, nil) else { return nil }
        defer { free(resolved) }
        return String(cString: resolved)
    }
    static func programCache(store: URL) -> URL {
        URL(fileURLWithPath: physicalPath(store.path) ?? store.path, isDirectory: true)
            .appendingPathComponent("cache/runtime-ane/programs", isDirectory: true)
    }
    struct ProgramCacheBudget: Sendable {
        let maximumBytes: UInt64
        let maximumEntries: Int
        static let standard = Self(maximumBytes: 128 << 20, maximumEntries: 64)
    }
    struct ProgramCacheReport: Sendable {
        let beforeBytes: UInt64
        let afterBytes: UInt64
        let beforeEntries: Int
        let afterEntries: Int
        let evictedKeys: [String]
    }
    private struct ProgramCacheEntry {
        let key: String
        let directory: stat
        let model: stat?
        let weights: stat?
        let lock: stat
        let incomplete: Bool
        let bytes: UInt64
        let lastRead: TimeInterval
        let modified: TimeInterval
    }
    private static func cacheFailure(_ detail: String) -> NativeFailure {
        NativeFailure(message: "runtime_cache_invalid: \(detail)。未知内容会保留，请检查 App 的 Runtime 程序缓存。")
    }
    private static func cacheKey(_ name: String) -> Bool {
        name.utf8.count == 64 && name.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    private static func cacheStat(_ fd: Int32, _ name: String) throws -> stat {
        guard let value = try optionalCacheStat(fd, name) else { throw cacheFailure("缓存条目已消失：\(name)") }
        return value
    }
    private static func optionalCacheStat(_ fd: Int32, _ name: String) throws -> stat? {
        var value = stat()
        if fstatat(fd, name, &value, AT_SYMLINK_NOFOLLOW) != 0 {
            guard errno == ENOENT else { throw cacheFailure("无法读取缓存条目：\(name)") }
            return nil
        }
        guard value.st_uid == getuid() else {
            throw cacheFailure("无法确认缓存条目的所有者：\(name)")
        }
        return value
    }
    private static func sameCacheFile(_ lhs: stat, _ rhs: stat) -> Bool {
        lhs.st_dev == rhs.st_dev && lhs.st_ino == rhs.st_ino && lhs.st_mode == rhs.st_mode &&
            lhs.st_uid == rhs.st_uid && lhs.st_nlink == rhs.st_nlink && lhs.st_size == rhs.st_size &&
            lhs.st_mtimespec.tv_sec == rhs.st_mtimespec.tv_sec && lhs.st_mtimespec.tv_nsec == rhs.st_mtimespec.tv_nsec
    }
    private static func sameOptionalCacheFile(_ lhs: stat?, _ rhs: stat?) -> Bool {
        switch (lhs, rhs) {
        case (nil, nil): return true
        case let (left?, right?): return sameCacheFile(left, right)
        default: return false
        }
    }
    private static func regularCacheFile(_ value: stat) -> Bool {
        value.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) && value.st_nlink == 1 && value.st_size >= 0
    }
    private static func time(_ value: timespec) -> TimeInterval {
        Double(value.tv_sec) + Double(value.tv_nsec) / 1_000_000_000
    }
    /// Bounds App-owned program *source* files. The ANE service's compiled-model
    /// cache is independent and is never purged here. Native source_file reads
    /// both files on reuse; their access times approximate LRU (OS-coalesced
    /// ties use modification time then key). No weights are read for trimming.
    static func trimProgramCache(store: URL, workerIsIdle: Bool,
                                 budget: ProgramCacheBudget = .standard) throws -> ProgramCacheReport {
        guard workerIsIdle else { throw cacheFailure("工作进程尚未确认退出，禁止清理运行中的缓存") }
        guard budget.maximumEntries >= 0 else { throw cacheFailure("缓存条目上限无效") }
        let cache = programCache(store: store)
        var rootInfo = stat()
        if lstat(cache.path, &rootInfo) != 0 {
            guard errno == ENOENT else { throw cacheFailure("无法读取程序缓存目录") }
            return ProgramCacheReport(beforeBytes: 0, afterBytes: 0, beforeEntries: 0, afterEntries: 0, evictedKeys: [])
        }
        guard rootInfo.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR), rootInfo.st_uid == getuid(),
              physicalPath(cache.path) == cache.path else { throw cacheFailure("缓存目录包含符号链接或所有者不匹配") }
        let rootFD = open(cache.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard rootFD >= 0 else { throw cacheFailure("无法安全打开程序缓存目录") }
        defer { close(rootFD) }
        var openedRoot = stat()
        guard fstat(rootFD, &openedRoot) == 0, sameCacheFile(rootInfo, openedRoot) else { throw cacheFailure("缓存目录已改变") }
        let names = try FileManager.default.contentsOfDirectory(atPath: cache.path).sorted()
        var entries: [ProgramCacheEntry] = [], lockFiles: [String: stat] = [:]
        // Validate every object before deleting anything: unknown content must
        // never be silently removed to satisfy a nominal byte limit.
        for name in names {
            if name.hasSuffix(".lock"), cacheKey(String(name.dropLast(5))) {
                let info = try cacheStat(rootFD, name)
                guard regularCacheFile(info), info.st_size == 0 else { throw cacheFailure("缓存锁文件格式不匹配：\(name)") }
                lockFiles[String(name.dropLast(5))] = info
            } else if !cacheKey(name) {
                throw cacheFailure("发现未知缓存条目：\(name)")
            }
        }
        for name in names where cacheKey(name) {
            let info = try cacheStat(rootFD, name)
            guard info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR), let lockInfo = lockFiles[name] else {
                throw cacheFailure("程序目录或配套锁文件格式不匹配：\(name)")
            }
            let entryFD = openat(rootFD, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard entryFD >= 0 else { throw cacheFailure("无法安全打开缓存条目：\(name)") }
            defer { close(entryFD) }
            let children = try FileManager.default.contentsOfDirectory(atPath: cache.appendingPathComponent(name).path)
            guard Set(children).isSubset(of: Set(["model.mil", "weights.bin"])) else { throw cacheFailure("程序目录包含未知文件：\(name)") }
            let model = try optionalCacheStat(entryFD, "model.mil"), weights = try optionalCacheStat(entryFD, "weights.bin")
            let sources = [model, weights].compactMap { $0 }
            guard sources.allSatisfy(regularCacheFile) else {
                throw cacheFailure("程序源文件包含链接或格式不匹配：\(name)")
            }
            // Native creates the directory before writing its two sources. A
            // cancelled writer can leave only known files or an empty MIL.
            // Empty weights are valid for programs with no constants.
            let incomplete = model == nil || weights == nil || model?.st_size == 0
            let bytes = sources.reduce(UInt64(0)) { $0 + UInt64($1.st_size) }
            entries.append(ProgramCacheEntry(key: name, directory: info, model: model, weights: weights, lock: lockInfo,
                incomplete: incomplete, bytes: bytes,
                lastRead: sources.map { time($0.st_atimespec) }.max() ?? time(info.st_atimespec),
                modified: sources.map { time($0.st_mtimespec) }.max() ?? time(info.st_mtimespec)))
        }
        let beforeBytes = try entries.reduce(UInt64(0)) { sum, entry in
            let (value, overflow) = sum.addingReportingOverflow(entry.bytes)
            guard !overflow else { throw cacheFailure("缓存大小溢出") }; return value
        }
        var bytes = beforeBytes, count = entries.count, removed: [String] = []
        entries.sort {
            if $0.incomplete != $1.incomplete { return $0.incomplete }
            if $0.lastRead != $1.lastRead { return $0.lastRead < $1.lastRead }
            if $0.modified != $1.modified { return $0.modified < $1.modified }
            return $0.key < $1.key
        }
        func eraseLock(_ key: String, expected: stat) throws {
            let name = key + ".lock"
            let fd = openat(rootFD, name, O_RDWR | O_NOFOLLOW | O_CLOEXEC)
            guard fd >= 0 else { throw cacheFailure("缓存锁已改变：\(key)") }
            defer { flock(fd, LOCK_UN); close(fd) }
            var actual = stat()
            guard fstat(fd, &actual) == 0, sameCacheFile(expected, actual),
                  flock(fd, LOCK_EX | LOCK_NB) == 0 else { throw cacheFailure("缓存条目正在使用或已改变：\(key)") }
            guard unlinkat(rootFD, name, 0) == 0 else { throw cacheFailure("无法移除空闲缓存锁：\(key)") }
        }
        for entry in entries where entry.incomplete || bytes > budget.maximumBytes || count > budget.maximumEntries {
            let lockFD = openat(rootFD, entry.key + ".lock", O_RDWR | O_NOFOLLOW | O_CLOEXEC)
            guard lockFD >= 0 else { throw cacheFailure("缓存锁已改变：\(entry.key)") }
            defer { flock(lockFD, LOCK_UN); close(lockFD) }
            var lockInfo = stat()
            guard fstat(lockFD, &lockInfo) == 0, sameCacheFile(entry.lock, lockInfo),
                  flock(lockFD, LOCK_EX | LOCK_NB) == 0 else { throw cacheFailure("缓存条目正在使用：\(entry.key)") }
            guard sameCacheFile(entry.directory, try cacheStat(rootFD, entry.key)) else { throw cacheFailure("程序目录已改变：\(entry.key)") }
            let entryFD = openat(rootFD, entry.key, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard entryFD >= 0 else { throw cacheFailure("无法安全打开待清理程序目录") }
            defer { close(entryFD) }
            guard sameOptionalCacheFile(entry.model, try optionalCacheStat(entryFD, "model.mil")),
                  sameOptionalCacheFile(entry.weights, try optionalCacheStat(entryFD, "weights.bin")) else { throw cacheFailure("程序源文件已改变") }
            // Fixed basenames under anchored directory FDs: no recursive delete
            // and no symlink target or path outside the owned directory is used.
            if entry.model != nil, unlinkat(entryFD, "model.mil", 0) != 0 { throw cacheFailure("程序 MIL 清理未完成") }
            if entry.weights != nil, unlinkat(entryFD, "weights.bin", 0) != 0 { throw cacheFailure("程序常量清理未完成") }
            guard unlinkat(rootFD, entry.key, AT_REMOVEDIR) == 0,
                  unlinkat(rootFD, entry.key + ".lock", 0) == 0 else { throw cacheFailure("程序缓存清理未完成") }
            bytes -= entry.bytes; count -= 1; removed.append(entry.key); lockFiles[entry.key] = nil
        }
        // Bound zero-byte native lock remnants too, after an idle/exit proof.
        let remaining = Set(entries.map(\.key)).subtracting(removed)
        for (key, info) in lockFiles where !remaining.contains(key) { try eraseLock(key, expected: info) }
        return ProgramCacheReport(beforeBytes: beforeBytes, afterBytes: bytes,
                                  beforeEntries: entries.count, afterEntries: count, evictedKeys: removed)
    }
    static func options(model: String, store: URL) throws -> Options {
        guard ["z-image-turbo", "qwen-image-2.1"].contains(model) else { throw invalid("当前模型尚未验收 Runtime 实验") }
        let root = URL(fileURLWithPath: physicalPath(store.path) ?? store.path, isDirectory: true).appendingPathComponent("cache/runtime-ane", isDirectory: true)
        let descriptor = root.appendingPathComponent("descriptors/\(model)-1056-v1.json")
        let cache = programCache(store: store)
        for directory in [root, descriptor.deletingLastPathComponent(), cache] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            guard physicalPath(directory.path) == directory.path else { throw invalid("实验缓存目录不能是符号链接") }
        }
        let object = shape(model: model)
        if !FileManager.default.fileExists(atPath: descriptor.path) {
            let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
            do { try data.write(to: descriptor, options: .withoutOverwriting) }
            catch { if !FileManager.default.fileExists(atPath: descriptor.path) { throw error } }
        }
        try validateDescriptor(path: descriptor.path, model: model)
        return Options(profile_id: profileID, descriptor_path: descriptor.path, cache_dir: cache.path)
    }
    private static func shape(model: String) -> [String: Any] {
        ["schema_version": 1, "descriptor_version": 1, "backend": "private_runtime_shape", "kind": "swiglu",
         "rows": 1056, "hidden": model == "z-image-turbo" ? 3840 : 4096,
         "width": model == "z-image-turbo" ? 10240 : 12288, "tile_k": 2048, "tile_n": 1024,
         "layout": "out_in", "biases": false, "lora_inputs": true]
    }
    private static func absolute(_ path: String) -> Bool {
        !path.isEmpty && !path.contains("\0") && path.hasPrefix("/") &&
            path.split(separator: "/", omittingEmptySubsequences: false).dropFirst().allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    static func validateDescriptor(path: String, model: String) throws {
        guard absolute(path),
              try URL(fileURLWithPath: path).resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]).isRegularFile == true,
              physicalPath(path) == path,
              let object = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any],
              NSDictionary(dictionary: object).isEqual(to: shape(model: model)) else { throw invalid("实验描述文件的形状或版本不匹配") }
        for key in ["schema_version", "descriptor_version", "rows", "hidden", "width", "tile_k", "tile_n"] {
            guard let number = object[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { throw invalid("实验描述数值无效") }
        }
        for key in ["biases", "lora_inputs"] {
            guard let number = object[key] as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { throw invalid("实验描述开关无效") }
        }
    }
    static func validate(request: NativeRequestV2, options: Options) throws {
        guard options.profile_id == profileID, absolute(options.descriptor_path), absolute(options.cache_dir),
              ["z-image-turbo", "qwen-image-2.1"].contains(request.model),
              request.schema_version == 2, request.outputs.count == 1,
              request.outputs[0].kind == "image", request.outputs[0].width == 512, request.outputs[0].height == 512,
              request.outputs[0].frames == 1, request.outputs[0].fps == 24, !request.outputs[0].audio,
              request.execution.policy == "gpu_ane", request.execution.hybrid_mlp_mode == "runtime",
              request.execution.ane_manifest == options.descriptor_path,
              request.execution.allow_approximation == true, request.execution.profile == nil,
              request.execution.encoder_ane_manifest == nil, request.execution.streaming == nil,
              request.execution.quantized_cache == nil, request.execution.qwen21_w8a8 != true,
              request.execution.qwen21_dit_cache == nil || request.execution.qwen21_dit_cache == "off",
              request.parameters.dynamic_text, request.parameters.compile_gpu != true, request.parameters.noise_path == nil,
              request.dump_tensors == nil else { throw invalid("实验仅支持已验收的 512 参数，需关闭冻结量化、流式加载、DiT 缓存与额外编译") }
        let images = request.inputs.filter { $0.kind == "image" }
        guard request.inputs.count == images.count + 1,
              request.inputs.first?.kind == "text", request.inputs.first?.role == "prompt",
              (request.loras ?? []).allSatisfy({ $0.role == "transformer" && $0.strength.isFinite && (-8...8).contains($0.strength) }),
              (request.loras ?? []).count <= 8,
              ((request.loras ?? []).isEmpty ? request.lora_strategy == nil : request.lora_strategy == "inference_time") else { throw invalid("输入与 Runtime LoRA 策略不匹配") }
        if request.model == "z-image-turbo" {
            guard request.operation == "image.generate", images.isEmpty, request.sampling.steps == 8,
                  request.execution.residency == "resident" else { throw invalid("Z-Image 实验固定为 512×512、8 步、常驻文生图") }
        } else {
            guard request.execution.residency == "component_staged", images.count <= 2,
                  images.allSatisfy({ $0.role == "reference" }),
                  request.operation == (images.isEmpty ? "image.generate" : "image.edit"),
                  request.parameters.qwen21_reference_size == 1024 else { throw invalid("Qwen 实验需要分阶段加载与标准 1024 编码，最多 2 张参考图") }
            let loras = request.loras ?? []
            if loras.isEmpty {
                guard request.sampling.steps == 20 else { throw invalid("Qwen 基础模型实验固定为 20 步") }
            } else {
                guard loras.count == 1, URL(fileURLWithPath: loras[0].path).lastPathComponent == "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
                      loras[0].strength == 1, request.sampling.steps == 6 else { throw invalid("Qwen 实验仅验收 Viggle r128、6 步、强度 1") }
            }
        }
        try validateDescriptor(path: options.descriptor_path, model: request.model)
        guard physicalPath(options.cache_dir) == options.cache_dir,
              try URL(fileURLWithPath: options.cache_dir).resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else { throw invalid("实验缓存目录无效") }
    }
    static func environment(options: Options, model: String, inherited: [String: String] = ProcessInfo.processInfo.environment) throws -> [String: String] {
        guard options.profile_id == profileID, ["z-image-turbo", "qwen-image-2.1"].contains(model), absolute(options.cache_dir) else { throw invalid() }
        var result = inherited.filter { !$0.key.hasPrefix("TURBOCIDER_") }
        let values = ["ANE_BACKEND": "auto", "ALLOW_PRIVATE_ANE": "1", "PRIVATE_ANE_DATA_PATH": "w8a8",
                      "PRIVATE_ANE_GPU_IO": "1", "PRIVATE_ANE_CHANNELS": "4096", "RUNTIME_ANE_CHUNKS": "1",
                      "RUNTIME_ANE_PROFILE": "0", "PRIVATE_ANE_SCALE_CACHE": "1", "PRIVATE_ANE_PREFETCH": "0",
                      "PRIVATE_ANE_LAUNCH_FENCE": "0", "PRIVATE_ANE_A8_LOOKAHEAD": "0", "PRIVATE_ANE_STAGE_SPECIALIZE": "0",
                      "PRIVATE_ANE_CACHE_DIR": options.cache_dir]
        for (key, value) in values { result["TURBOCIDER_" + key] = value }
        if model == "qwen-image-2.1" {
            result["TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC"] = "1"
            result["TURBOCIDER_QWEN21_RUNTIME_PREPARE_EARLY"] = "0"
        }
        return result
    }
    static func requestDigest(request: [String: Any], options: [String: Any]) throws -> String {
        var bytes = Data("tc-runtime-worker-request-v1\n".utf8)
        bytes.append(try JSONSerialization.data(withJSONObject: ["native_request_v2": request, "runtime_options": options], options: [.sortedKeys, .withoutEscapingSlashes]))
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    static func encode(jobID: UUID, requestID: UUID, model: URL, request: NativeRequestV2, options: Options) throws -> Data {
        try validate(request: request, options: options)
        guard model.isFileURL, absolute(model.standardizedFileURL.path) else { throw invalid() }
        let native = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
        let typed = try JSONSerialization.jsonObject(with: JSONEncoder().encode(options)) as! [String: Any]
        let wire: [String: Any] = ["protocol_version": 1, "job_id": jobID.uuidString.lowercased(), "request_id": requestID.uuidString.lowercased(),
            "request_digest": try requestDigest(request: native, options: typed), "model_installation_ref": model.standardizedFileURL.path,
            "native_request_v2": native, "runtime_options": typed]
        let bytes = try JSONSerialization.data(withJSONObject: wire, options: [.sortedKeys, .withoutEscapingSlashes])
        guard bytes.count <= 1 << 20 else { throw invalid("请求过大") }
        return bytes
    }
    static func prepare(jobID: UUID, model: URL, legacy: NativeRequest, options: Options, store: URL) throws -> Prepared {
        guard legacy.prompt_enhance != true, legacy.prompt_enhance_edit_experimental != true else { throw invalid("实验暂不支持提示词增强") }
        var request = NativeRequestV2(legacy: legacy)
        // Public streaming deliberately omits residency. Runtime must retain it.
        request.execution.residency = legacy.residency
        if (request.loras ?? []).isEmpty { request.lora_strategy = nil }
        let requestID = UUID()
        let input = try encode(jobID: jobID, requestID: requestID, model: model, request: request, options: options)
        let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
        let identity = NativeEngine.runtimeBuildIdentity()
        guard !identity.isEmpty else { throw invalid("缺少运行库身份") }
        let reference = PublicImageWorker.Reference(requestID: requestID, requestDigest: wire["request_digest"] as! String,
            runtimeFingerprint: identity, stagedOutput: request.outputs[0].path)
        let directory = reference.directory(jobID: jobID, store: store)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let inputURL = directory.appendingPathComponent("input.json")
        try input.write(to: inputURL, options: .withoutOverwriting)
        return Prepared(reference: reference, input: input, inputURL: inputURL, request: request, options: options)
    }
    static func validateTerminal(_ data: Data, prepared: Prepared, exitCode: Int32) throws -> Verified {
        guard data.count <= 4 << 20,
              let wire = try JSONSerialization.jsonObject(with: prepared.input) as? [String: Any],
              let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = value["protocol_version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              value["actual_container"] as? String == "cli_worker",
              value["runtime_fingerprint"] as? String == prepared.reference.runtimeFingerprint,
              let status = value["status"] as? String else { throw invalid() }
        for key in ["job_id", "request_id", "request_digest"] {
            guard let expected = wire[key] as? String, value[key] as? String == expected else { throw invalid() }
        }
        let native = try JSONSerialization.jsonObject(with: JSONEncoder().encode(prepared.request)) as! [String: Any]
        let typed = try JSONSerialization.jsonObject(with: JSONEncoder().encode(prepared.options)) as! [String: Any]
        guard wire["request_digest"] as? String == (try requestDigest(request: native, options: typed)),
              let echo = value["runtime_options"] as? [String: Any], NSDictionary(dictionary: echo).isEqual(to: typed) else { throw invalid() }
        for key in ["resolution_digest", "record_digest", "layout_digest", "public_streaming_summary"] {
            guard value[key] is NSNull else { throw invalid() }
        }
        guard value["resolution"] == nil || value["resolution"] is NSNull else { throw invalid() }
        if status == "error" || status == "cancelled" {
            guard exitCode == (status == "cancelled" ? 2 : 1),
                  let error = value["error"] as? [String: Any], let code = error["code"] as? String, !code.isEmpty,
                  let message = error["message"] as? String, !message.isEmpty,
                  status != "cancelled" || code == "worker_cancelled" else { throw invalid() }
            for key in ["result", "runtime_receipt", "artifact"] { guard value[key] == nil || value[key] is NSNull else { throw invalid() } }
            return Verified(status: status, result: nil, artifact: nil, receipt: nil, errorCode: code, errorMessage: message)
        }
        guard status == "succeeded", exitCode == 0, value["error"] is NSNull,
              let rawResult = value["result"] as? [String: Any], let rawArtifact = value["artifact"] as? [String: Any],
              let rawReceipt = value["runtime_receipt"] as? [String: Any],
              Set(rawReceipt.keys) == Set(["backend", "data_path", "partition_axis", "ane_channels", "fallback_reason", "hybrid_blocks", "gpu_blocks", "fallback_blocks", "selected_backend", "actual_execution", "partial_fallback"]) else { throw invalid() }
        for key in ["ane_channels", "hybrid_blocks", "gpu_blocks", "fallback_blocks"] {
            guard let number = rawReceipt[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite, number.doubleValue >= 0, number.doubleValue.rounded() == number.doubleValue else { throw invalid() }
        }
        for key in ["schema_version", "width", "height", "steps", "seed"] {
            guard let number = rawResult[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue else { throw invalid() }
        }
        guard let warmup = rawResult["warmup"] as? NSNumber, CFGetTypeID(warmup) == CFBooleanGetTypeID(), !warmup.boolValue,
              (rawResult["schema_version"] as? NSNumber)?.intValue == 1 else { throw invalid() }
        guard rawResult["model"] as? String == prepared.request.model,
              rawResult["operation"] as? String == prepared.request.operation,
              rawResult["output"] as? String == prepared.request.outputs[0].path,
              (rawResult["width"] as? NSNumber)?.intValue == 512, (rawResult["height"] as? NSNumber)?.intValue == 512,
              (rawResult["steps"] as? NSNumber)?.intValue == prepared.request.sampling.steps,
              (rawResult["seed"] as? NSNumber)?.intValue == prepared.request.sampling.seed,
              rawResult["warmup"] as? Bool == false,
              rawResult["public_streaming"] == nil || rawResult["public_streaming"] is NSNull else { throw invalid() }
        let receipt = try JSONDecoder().decode(Receipt.self, from: JSONSerialization.data(withJSONObject: rawReceipt))
        let metrics = (rawResult["hybrid"] as? [String: Any])?["runtime_weight"] as? [String: Any]
        guard let metrics,
              (metrics["hybrid_blocks_session_total"] as? NSNumber)?.intValue == receipt.hybrid_blocks,
              (metrics["gpu_blocks_session_total"] as? NSNumber)?.intValue == receipt.gpu_blocks,
              (metrics["fallback_blocks_session_total"] as? NSNumber)?.intValue == receipt.fallback_blocks else { throw invalid() }
        guard let plan = rawResult["plan"] as? [String: Any],
              let contract = plan["runtime_weight_contract"] as? [String: Any],
              let hybrid = rawResult["hybrid"] as? [String: Any] else { throw invalid() }
        for key in ["ane_channels", "gpu_channels", "hybrid_blocks_session_total", "gpu_blocks_session_total", "fallback_blocks_session_total"] {
            guard let number = metrics[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite, number.doubleValue >= 0, number.doubleValue.rounded() == number.doubleValue else { throw invalid() }
        }
        guard let reason = metrics["backend_fallback_reason"] as? String, let failureReason = metrics["failure_reason"] as? String,
              let runtimeFailed = hybrid["runtime_failed"] as? NSNumber, CFGetTypeID(runtimeFailed) == CFBooleanGetTypeID() else { throw invalid() }
        guard receipt.fallback_reason == (failureReason.isEmpty ? reason : failureReason),
              contract["data_path"] as? String == metrics["data_path"] as? String,
              contract["partition_axis"] as? String == metrics["partition_axis"] as? String else { throw invalid() }
        let selected = contract["executor_backend"] as? String
        let failed = runtimeFailed.boolValue
        guard rawReceipt["partial_fallback"] is NSNumber,
              CFGetTypeID(rawReceipt["partial_fallback"] as! NSNumber) == CFBooleanGetTypeID(),
              receipt.selected_backend == selected,
              receipt.actual_execution == (receipt.hybrid_blocks > 0 ? "gpu_ane" : "gpu"),
              receipt.partial_fallback == (receipt.hybrid_blocks > 0 && (receipt.fallback_blocks > 0 || failed)) else { throw invalid() }
        if selected == "private_ane" {
            guard metrics["executor_backend"] as? String == selected,
                  contract["data_path"] as? String == "w8a8_hadamard",
                  contract["partition_axis"] as? String == "intermediate_channels",
                  (metrics["ane_channels"] as? NSNumber)?.intValue == 4096,
                  (metrics["gpu_channels"] as? NSNumber)?.intValue == (prepared.request.model == "z-image-turbo" ? 10240 : 12288) - 4096,
                  plan["execution"] as? String == "gpu_ane_experimental" else { throw invalid() }
        } else {
            guard contract["executor_backend"] is NSNull, metrics["executor_backend"] is NSNull,
                  rawReceipt["selected_backend"] is NSNull, receipt.hybrid_blocks == 0,
                  (metrics["ane_channels"] as? NSNumber)?.intValue == 0,
                  contract["data_path"] as? String == "",
                  ["", "rows"].contains(contract["partition_axis"] as? String ?? "invalid"),
                  plan["execution"] as? String == "gpu" else { throw invalid() }
        }
        if receipt.backend == "private_ane" {
            guard receipt.data_path == "w8a8_hadamard", receipt.partition_axis == "intermediate_channels", receipt.ane_channels == 4096,
                  receipt.hybrid_blocks > 0, metrics["executor_backend"] as? String == "private_ane",
                  metrics["data_path"] as? String == receipt.data_path,
                  metrics["partition_axis"] as? String == receipt.partition_axis,
                  (metrics["ane_channels"] as? NSNumber)?.intValue == 4096,
                  !receipt.partial_fallback || !receipt.fallback_reason.isEmpty else { throw invalid() }
        } else {
            guard receipt.backend == "gpu", receipt.data_path.isEmpty, receipt.partition_axis.isEmpty,
                  receipt.ane_channels == 0, receipt.hybrid_blocks == 0, !receipt.fallback_reason.isEmpty else { throw invalid() }
        }
        let artifact = try JSONDecoder().decode(WorkerTerminalEnvelope.Artifact.self, from: JSONSerialization.data(withJSONObject: rawArtifact))
        guard artifact.path == prepared.request.outputs[0].path, artifact.size > 0,
              artifact.sha256.utf8.count == 64, artifact.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw invalid() }
        return Verified(status: status, result: try JSONSerialization.data(withJSONObject: rawResult), artifact: artifact,
                        receipt: receipt, errorCode: nil, errorMessage: nil)
    }
}

/// Runtime progress is correlated independently; no streaming resolution exists.
final class RuntimeWorkerEventStream: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data(), discarding = false, failed = false
    private var sequence = 0, nativeSequence = -1
    private let jobID: String, reference: PublicImageWorker.Reference
    private let callback: @Sendable (NativeEvent) -> Void
    private static let prefix = Data("TC_EVENT\t".utf8)
    init(jobID: UUID, reference: PublicImageWorker.Reference, callback: @escaping @Sendable (NativeEvent) -> Void) {
        self.jobID = jobID.uuidString.lowercased(); self.reference = reference; self.callback = callback
    }
    func consume(_ data: Data) throws {
        lock.lock(); defer { lock.unlock() }
        do {
            guard !failed else { throw NativeFailure(message: "runtime_worker_event_invalid") }
            for byte in data {
                if byte == 10 {
                    if !discarding { try line(pending) }
                    pending.removeAll(keepingCapacity: true); discarding = false
                } else if !discarding {
                    pending.append(byte)
                    if pending.count > 128 * 1024 + Self.prefix.count {
                        guard !pending.starts(with: Self.prefix) else { throw NativeFailure(message: "runtime_worker_event_invalid") }
                        pending.removeAll(keepingCapacity: true); discarding = true
                    }
                }
            }
        } catch { failed = true; throw error }
    }
    private func line(_ data: Data) throws {
        guard data.starts(with: Self.prefix) else { return }
        guard let object = try JSONSerialization.jsonObject(with: Data(data.dropFirst(Self.prefix.count))) as? [String: Any],
              let version = object["event_schema_version"] as? NSNumber, CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              let next = object["sequence"] as? NSNumber, CFGetTypeID(next) != CFBooleanGetTypeID(), next.intValue > sequence,
              next.doubleValue == Double(next.intValue), object["kind"] as? String == "progress",
              object["job_id"] as? String == jobID, object["request_id"] as? String == reference.requestID.uuidString.lowercased(),
              object["request_digest"] as? String == reference.requestDigest,
              object["execution_container"] as? String == "cli_worker", object["runtime_fingerprint"] as? String == reference.runtimeFingerprint,
              let payload = object["payload"] as? [String: Any] else { throw NativeFailure(message: "runtime_worker_event_invalid") }
        let value = try JSONDecoder().decode(NativeEvent.self, from: JSONSerialization.data(withJSONObject: payload))
        guard value.sequence > nativeSequence, !value.phase.isEmpty, value.phase.utf8.count <= 128,
              value.completed >= 0, value.total >= 0, value.completed <= value.total,
              value.elapsed_seconds.isFinite, value.elapsed_seconds >= 0 else { throw NativeFailure(message: "runtime_worker_event_invalid") }
        nativeSequence = value.sequence; sequence = next.intValue; callback(value)
    }
    func finish() throws {
        lock.lock(); defer { lock.unlock() }
        guard !failed, !pending.starts(with: Self.prefix) else { throw NativeFailure(message: "runtime_worker_event_invalid") }
    }
}
