import Foundation

/// Opt-in real-model regression: tokenizer boundaries and GPU inference beyond 512.
@main struct ZImagePromptTests {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            throw NativeFailure(message: "z-image-prompt-tests MODEL OUTPUT")
        }
        let model = CommandLine.arguments[1]
        let output = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        func check(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
        let base = "A cinematic red fox walking through fresh snow, soft morning light."
        func prompt(_ count: Int) -> String { base + String(repeating: " snow", count: count - 21) }
        for count in [512, 513, 1024, 1025] {
            try check(try NativeEngine.zImageTokenCount(modelPath: model, prompt: prompt(count)) == count,
                      "Exact token count mismatch at \(count)")
        }
        do {
            _ = try NativeEngine.zImageTokenCount(modelPath: model, prompt: String(repeating: "a", count: 32769))
            throw NativeFailure(message: "Oversized byte input accepted")
        } catch { try check(error.localizedDescription.contains("32 KiB"), "Wrong byte limit error") }
        let engine = try NativeEngine(modelURL: URL(fileURLWithPath: model), modelID: "z-image-turbo")
        for (count, dynamic) in [(513, true), (1024, false)] {
            var request = NativeRequest(prompt: prompt(count), output: output.appendingPathComponent("gpu-\(count).png").path)
            request.model = "z-image-turbo"; request.steps = 8; request.dynamic_text = dynamic
            let data = try await engine.generate(request) { _ in }
            let result = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            try check(result["valid_text_tokens"] as? Int == count, "Generation truncated prompt")
            try check(FileManager.default.fileExists(atPath: request.output), "Missing generated image")
            try data.write(to: output.appendingPathComponent("gpu-\(count).json"))
            print("PASS GPU \(count) tokens, dynamic_text=\(dynamic)")
        }
        var oversized = NativeRequest(prompt: prompt(1025), output: output.appendingPathComponent("must-not-exist.png").path)
        oversized.model = "z-image-turbo"
        do {
            _ = try await engine.generate(oversized) { _ in }
            throw NativeFailure(message: "Overlimit generation silently accepted")
        } catch { try check(error.localizedDescription.contains("1024 tokens"), "Wrong generation limit error: \(error)") }
        try check(!FileManager.default.fileExists(atPath: oversized.output), "Overlimit generation created an image")
        print("PASS exact counter, byte limit, extended GPU context, no truncation with dynamic_text=false, 1025 rejection")
    }
}
