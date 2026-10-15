// Original benchmark client for the separately pinned FlashAttention package.
// The upstream library remains an optional research dependency, never a product dependency.
import Foundation
import Metal
import MetalPerformanceShaders
import FlashAttention

struct LabFailure: Error, CustomStringConvertible {
    let description: String
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw LabFailure(description: message) }
}

struct Generator {
    var state: UInt64
    mutating func value() -> Float16 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let code = Int((state >> 32) % 257) - 128
        return Float16(Float(code) / 1024)
    }
}

func execute() throws -> [String: Any] {
    let args = CommandLine.arguments
    try require(args.count == 5, "usage: MFAReplay M N K repeats-multiple-of-four")
    guard let m = Int(args[1]), let n = Int(args[2]), let k = Int(args[3]), let repeats = Int(args[4]) else {
        throw LabFailure(description: "integer geometry required")
    }
    try require([m, n, k].allSatisfy { 1...32768 ~= $0 }, "bounded geometry required")
    try require(4...32 ~= repeats && repeats % 4 == 0, "repeats must be 4..32 in ABBA groups")
    let device = MTLContext.global.device
    let queue = MTLContext.global.commandQueue
    let itemBytes = MemoryLayout<Float16>.stride
    let aBytesPerRow = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
    let bBytesPerRow = MPSMatrixDescriptor.rowBytes(fromColumns: k, dataType: .float16)
    let cBytesPerRow = MPSMatrixDescriptor.rowBytes(fromColumns: n, dataType: .float16)
    let aStride = aBytesPerRow / itemBytes
    let bStride = bBytesPerRow / itemBytes
    let cStride = cBytesPerRow / itemBytes
    let aBytes = m * aBytesPerRow, bBytes = n * bBytesPerRow, cBytes = m * cBytesPerRow
    try require(aBytes + bBytes + cBytes * 2 <= 512 * 1024 * 1024, "buffer admission exceeds 512 MiB")
    guard let a = device.makeBuffer(length: aBytes, options: .storageModeShared),
          let b = device.makeBuffer(length: bBytes, options: .storageModeShared),
          let baseline = device.makeBuffer(length: cBytes + 64, options: .storageModeShared),
          let candidate = device.makeBuffer(length: cBytes + 64, options: .storageModeShared) else {
        throw LabFailure(description: "shared buffer allocation failed")
    }
    let ap = a.contents().bindMemory(to: Float16.self, capacity: aBytes / itemBytes)
    let bp = b.contents().bindMemory(to: Float16.self, capacity: bBytes / itemBytes)
    // Define padded rows too; neither backend should see uninitialized storage.
    memset(a.contents(), 0, aBytes)
    memset(b.contents(), 0, bBytes)
    var weightRng = Generator(state: 71)
    for row in 0..<n {
        for column in 0..<k { bp[row * bStride + column] = weightRng.value() }
    }
    let am = MPSMatrix(buffer: a, descriptor: MPSMatrixDescriptor(rows: m, columns: k, rowBytes: aBytesPerRow, dataType: .float16))
    let bm = MPSMatrix(buffer: b, descriptor: MPSMatrixDescriptor(rows: n, columns: k, rowBytes: bBytesPerRow, dataType: .float16))
    let cm = MPSMatrix(buffer: baseline, descriptor: MPSMatrixDescriptor(rows: m, columns: n, rowBytes: cBytesPerRow, dataType: .float16))
    let mps = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
                                      resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)
    var descriptor = GEMMDescriptor()
    descriptor.matrixDimensions = (M: UInt32(m), N: UInt32(n), K: UInt32(k))
    descriptor.leadingDimensions = (A: UInt32(aStride), B: UInt32(bStride), C: UInt32(cStride))
    descriptor.transposeState = (A: false, B: true)
    descriptor.memoryPrecisions = (A: .FP16, B: .FP16, C: .FP16)
    descriptor.loadPreviousC = false
    let prepare = DispatchTime.now().uptimeNanoseconds
    GEMMKernel.register(descriptor: descriptor)
    guard let (kernel, pipeline) = GEMMKernel.pipelineCache[descriptor] else {
        throw LabFailure(description: "MFA pipeline missing")
    }
    let prepareSeconds = Double(DispatchTime.now().uptimeNanoseconds - prepare) * 1e-9
    let guardByte: UInt8 = 0x6d
    memset(baseline.contents().advanced(by: cBytes), Int32(guardByte), 64)
    memset(candidate.contents().advanced(by: cBytes), Int32(guardByte), 64)
    var samples: [[String: Any]] = []
    var maxAbs = [0.0, 0.0], errorEnergy = [0.0, 0.0], referenceEnergy = [0.0, 0.0]
    var previousGeneration = -1
    for trial in -4..<repeats {
        let variant = (trial + 4) % 4 == 1 || (trial + 4) % 4 == 2
        let arm = variant ? 1 : 0
        let generation = (trial + 4) / 2
        if generation != previousGeneration {
            var inputRng = Generator(state: UInt64(17 + generation))
            for row in 0..<m {
                for column in 0..<k { ap[row * aStride + column] = inputRng.value() }
            }
            previousGeneration = generation
        }
        let buffer = variant ? candidate : baseline
        let output = buffer.contents().bindMemory(to: Float16.self, capacity: cBytes / itemBytes)
        // Poison every trial outside the timed span so stale outputs cannot pass.
        for index in 0..<(cBytes / itemBytes) { output[index] = .nan }
        let started = DispatchTime.now().uptimeNanoseconds
        guard let command = queue.makeCommandBuffer() else { throw LabFailure(description: "command allocation failed") }
        if variant {
            guard let encoder = command.makeComputeCommandEncoder() else { throw LabFailure(description: "encoder allocation failed") }
            encoder.setComputePipelineState(pipeline)
            encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(candidate, offset: 0, index: 2)
            let grid = MTLSize(width: (n + Int(kernel.blockDimensions.N) - 1) / Int(kernel.blockDimensions.N),
                               height: (m + Int(kernel.blockDimensions.M) - 1) / Int(kernel.blockDimensions.M), depth: 1)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))
            encoder.endEncoding()
        } else {
            mps.encode(commandBuffer: command, leftMatrix: am, rightMatrix: bm, resultMatrix: cm)
        }
        command.commit()
        command.waitUntilCompleted()
        let wall = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        try require(command.status == .completed, "GPU command did not complete: \(String(describing: command.error))")
        for row in 0..<m {
            for column in 0..<n { try require(output[row * cStride + column].isFinite, "nonfinite logical output") }
        }
        let tail = buffer.contents().advanced(by: cBytes).bindMemory(to: UInt8.self, capacity: 64)
        try require((0..<64).allSatisfy { tail[$0] == guardByte }, "output tail guard overwritten")
        // Independent double dot oracle on bounded samples, not a whole-matrix accuracy claim.
        for index in 0..<32 {
            let row = index == 31 ? m - 1 : (index * 7919) % m
            let column = index == 31 ? n - 1 : (index * 3571) % n
            var reference = 0.0
            for inner in 0..<k { reference += Double(ap[row * aStride + inner]) * Double(bp[column * bStride + inner]) }
            let error = Double(output[row * cStride + column]) - reference
            maxAbs[arm] = max(maxAbs[arm], abs(error))
            errorEnergy[arm] += error * error
            referenceEnergy[arm] += reference * reference
        }
        if trial >= 0 {
            let gpuSpan = command.gpuEndTime - command.gpuStartTime
            samples.append(["variant": variant ? "mfa" : "mps", "pair": trial / 2,
                            "input_generation": generation, "wall_seconds": wall,
                            "gpu_parent_command_seconds": gpuSpan > 0 ? gpuSpan as Any : NSNull()])
        }
    }
    let relativeL2 = (0..<2).map { sqrt(errorEnergy[$0] / max(referenceEnergy[$0], 1e-30)) }
    try require(relativeL2.allSatisfy { $0 <= 0.02 }, "one arm exceeds the fixed 0.02 sampled relative L2 gate")
    return ["schema": "turbocider-mfa-mps-gemm-v1", "shape_mnk": [m,n,k], "device": device.name,
            "recorded_order": "ABBA", "warmup_trials": 4,
            "dtype": "fp16 inputs/outputs; backend-native accumulation", "source_weights": "synthetic deterministic FP16, not checkpoint values",
            "physical_element_strides": [aStride,bStride,cStride], "mfa_prepare_seconds": prepareSeconds,
            "mfa_block_dimensions": [Int(kernel.blockDimensions.M),Int(kernel.blockDimensions.N),Int(kernel.blockDimensions.K)],
            "sampled_oracle_arm_order": ["mps", "mfa"],
            "sampled_oracle_max_abs": maxAbs, "sampled_oracle_relative_l2": relativeL2,
            "oracle_samples_per_trial": 32, "relative_l2_gate": 0.02,
            "all_logical_outputs_finite": true, "tail_guards_intact": true, "samples": samples,
            "scope": "resident-weight single GEMM encode/submit/physical completion; setup/oracle/input fill excluded; not full FFN/request",
            "baseline_limit": "MPS is a library baseline, not proof of parity with TurboCider's optimized compiled MLX FFN",
            "production_qualified": false]
}

do {
    let report = try execute()
    let bytes = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(bytes)
    print("")
} catch {
    fputs("MFA integration failed: \(error)\n", stderr)
    exit(1)
}
