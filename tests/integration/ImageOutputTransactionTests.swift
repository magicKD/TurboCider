import Foundation
import CoreGraphics
import ImageIO
import CryptoKit

@main struct ImageOutputTransactionTests {
    static func check(_ value: Bool, _ message: String) throws {
        if !value { throw NativeFailure(message: message) }
    }
    static func main() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-image-tests-\(UUID())")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let fixture = root.appendingPathComponent("fixture.png")
        let context = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 0.3, green: 0.6, blue: 0.2, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let writer = CGImageDestinationCreateWithURL(fixture as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(writer, context.makeImage()!, nil)
        try check(CGImageDestinationFinalize(writer), "PNG fixture encoding failed")
        let png = try Data(contentsOf: fixture)
        let output = root.appendingPathComponent("final.png")
        let original = Data("existing image must survive rejected output".utf8)
        var request = NativeRequest(prompt: "fixture", output: output.path)
        request.model = "flux2-klein-4b"; request.operation = "image.generate"
        request.width = 64; request.height = 64; request.steps = 4; request.seed = 42
        func result(_ transaction: ImageOutputTransaction, _ mode: String = "valid") throws -> Data {
            var value: [String: Any] = ["schema_version": 1, "model": request.model,
                "operation": NativeRequestV2(legacy: request).operation, "output": transaction.stagedURL.path,
                "width": request.width, "height": request.height, "steps": request.steps,
                "seed": request.seed, "warmup": false]
            if mode == "seed" { value["seed"] = 43 }
            if mode == "old_path" { value["output"] = request.output }
            if mode == "warmup_type" { value["warmup"] = 0 }
            if mode == "schema" { value["schema_version"] = 2 }
            if mode == "empty_json" { value = [:] }
            return try JSONSerialization.data(withJSONObject: value)
        }
        for mode in ["seed", "old_path", "warmup_type", "schema", "empty_json", "empty_file", "missing_file", "invalid_png", "truncated_png", "bad_crc", "symlink", "dimensions"] {
            try original.write(to: output)
            var variant = request
            if mode == "dimensions" { variant.width = 128 }
            let transaction = try ImageOutputTransaction(request: variant)
            switch mode {
            case "missing_file": break
            case "empty_file": try Data().write(to: transaction.stagedURL)
            case "invalid_png": try Data("not PNG".utf8).write(to: transaction.stagedURL)
            case "truncated_png": try png.prefix(png.count / 2).write(to: transaction.stagedURL)
            case "bad_crc":
                var changed = png; changed[29] ^= 1
                try changed.write(to: transaction.stagedURL)
            case "symlink": try fm.createSymbolicLink(at: transaction.stagedURL, withDestinationURL: fixture)
            default: try png.write(to: transaction.stagedURL)
            }
            var rejected = false
            do {
                var payload = try result(transaction, mode)
                if mode == "dimensions" {
                    var value = try JSONSerialization.jsonObject(with: payload) as! [String: Any]
                    value["width"] = 128
                    payload = try JSONSerialization.data(withJSONObject: value)
                }
                _ = try transaction.prepare(payload)
                try transaction.publish()
            } catch { rejected = true }
            try check(rejected && (try Data(contentsOf: output)) == original, "Accepted invalid output: \(mode)")
        }
        // A destination created after validation must win atomically, even
        // when it is a dangling symlink rather than an existing regular file.
        try fm.removeItem(at: output)
        for kind in ["regular", "symlink", "dangling_symlink"] {
            let transaction = try ImageOutputTransaction(request: request)
            try png.write(to: transaction.stagedURL)
            _ = try transaction.prepare(result(transaction))
            if kind == "regular" { try original.write(to: output) }
            else {
                let target = kind == "symlink" ? fixture : root.appendingPathComponent("missing-target")
                try fm.createSymbolicLink(at: output, withDestinationURL: target)
            }
            var rejected = false
            do { try transaction.publish() }
            catch { rejected = error.localizedDescription.contains("image_publish_failed") }
            try check(rejected, "Publication replaced a concurrent destination: \(kind)")
            if kind == "regular" { try check(try Data(contentsOf: output) == original, "Existing destination changed") }
            else { _ = try fm.destinationOfSymbolicLink(atPath: output.path) }
            try check(try Data(contentsOf: fixture) == png, "Symlink target changed")
            try check(try Data(contentsOf: transaction.stagedURL) == png, "Rejected publication consumed staging")
            try fm.removeItem(at: output)
        }
        do {
            let transaction = try ImageOutputTransaction(request: request)
            try png.write(to: transaction.stagedURL)
            let receipt = try transaction.prepare(result(transaction))
            try check(!fm.fileExists(atPath: output.path), "Validation published before commit")
            try transaction.publish()
            let value = try JSONSerialization.jsonObject(with: receipt) as! [String: Any]
            let artifact = value["image_artifact"] as! [String: Any]
            let digest = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
            try check(value["output"] as? String == output.path && artifact["sha256"] as? String == digest,
                      "Receipt does not identify the published output")
            try check(try Data(contentsOf: output) == png, "Publication changed bytes")
            var rejected = false
            do { try transaction.publish() } catch { rejected = true }
            try check(rejected, "Double publication accepted")
        }
        try original.write(to: output)
        do {
            let transaction = try ImageOutputTransaction(request: request)
            try png.write(to: transaction.stagedURL)
            _ = try transaction.prepare(result(transaction))
            try png.write(to: transaction.stagedURL, options: .atomic)
            var rejected = false
            do { try transaction.publish() } catch { rejected = true }
            try check(rejected && (try Data(contentsOf: output)) == original, "Replaced staged inode was published")
        }
        do {
            let transaction = try ImageOutputTransaction(request: request)
            try png.write(to: transaction.stagedURL)
            _ = try transaction.prepare(result(transaction))
            let cancelled = await Task.detached { () -> Bool in
                withUnsafeCurrentTask { $0?.cancel() }
                do { try transaction.publish(); return false }
                catch is CancellationError { return true }
                catch { return false }
            }.value
            try check(cancelled && (try Data(contentsOf: output)) == original, "Cancelled publication replaced output")
        }
        do {
            let blocked = root.appendingPathComponent("blocked.png", isDirectory: true)
            try fm.createDirectory(at: blocked, withIntermediateDirectories: true)
            let marker = blocked.appendingPathComponent("keep")
            try original.write(to: marker)
            var value = request; value.output = blocked.path
            let transaction = try ImageOutputTransaction(request: value)
            try png.write(to: transaction.stagedURL)
            _ = try transaction.prepare(result(transaction))
            var rejected = false
            do { try transaction.publish() }
            catch { rejected = error.localizedDescription.contains("image_publish_failed") }
            try check(rejected && (try Data(contentsOf: marker)) == original, "Publication failure damaged destination")
        }
        for mode in ["staged", "published", "missing", "changed", "wrong_inode", "symlink", "bad_receipt"] {
            var recoveryRequest = request
            recoveryRequest.output = root.appendingPathComponent("recover-\(mode).png").path
            var transaction: ImageOutputTransaction? = try ImageOutputTransaction(request: recoveryRequest)
            let staged = transaction!.stagedURL
            try png.write(to: staged)
            var receipt = try transaction!.prepare(result(transaction!))
            transaction!.retainForRecovery()
            if ["published", "wrong_inode", "symlink"].contains(mode) { try transaction!.publish() }
            transaction = nil // Simulate the original owner no longer being present.
            let final = URL(fileURLWithPath: recoveryRequest.output)
            if mode == "missing" { try fm.removeItem(at: staged) }
            if mode == "changed" { try original.write(to: staged) }
            if mode == "wrong_inode" { try png.write(to: final, options: .atomic) }
            if mode == "symlink" {
                try fm.removeItem(at: final)
                try fm.createSymbolicLink(at: final, withDestinationURL: fixture)
            }
            if mode == "bad_receipt" {
                var value = try JSONSerialization.jsonObject(with: receipt) as! [String: Any]
                var artifact = value["image_artifact"] as! [String: Any]
                artifact["staged_output"] = fixture.path
                value["image_artifact"] = artifact
                receipt = try JSONSerialization.data(withJSONObject: value)
            }
            var recovered = false
            do { try ImageOutputTransaction.recover(request: recoveryRequest, result: receipt); recovered = true }
            catch { if ["staged", "published"].contains(mode) { throw error } }
            try check(recovered == ["staged", "published"].contains(mode), "Incorrect recovery outcome: \(mode)")
            if recovered {
                try check(try Data(contentsOf: final) == png, "Recovery changed pixels")
                try ImageOutputTransaction.recover(request: recoveryRequest, result: receipt) // Idempotent replay.
                try check(!fm.fileExists(atPath: staged.deletingLastPathComponent().path), "Recovery retained empty staging")
            } else {
                // Rejected history is retained for diagnostics; the verifier never
                // recursively deletes paths reconstructed from a receipt.
                try? fm.removeItem(at: staged.deletingLastPathComponent())
            }
        }
        let leftovers = try fm.contentsOfDirectory(atPath: root.path).filter { $0.hasPrefix(".tc-image-staging-") }
        try check(leftovers.isEmpty, "Staging directories leaked: \(leftovers)")
        print("PASS image transaction: request correlation, PNG decode, SHA receipt, atomic publication, rollback, cancellation, cleanup and receipt-bound recovery")
    }
}
