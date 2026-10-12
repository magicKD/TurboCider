import AppKit
import Foundation

/// NSItemProvider may keep its callback pending after the awaiting Task is
/// cancelled. Resolve our wait independently, and accept each result once.
enum ItemProviderDataLoader {
    @MainActor static func load(provider: NSItemProvider, typeIdentifier: String) async throws -> Data {
        let pending = PendingRead()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                guard pending.install(continuation) else { return }
                let progress = provider.loadDataRepresentation(forTypeIdentifier: typeIdentifier) { data, error in
                    if let data { pending.complete(.success(data)) }
                    else { pending.complete(.failure(error ?? NativeFailure(message: "无法读取拖入的图片。"))) }
                }
                pending.attach(progress)
            }
        } onCancel: {
            pending.cancel()
        }
    }

    private final class PendingRead: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Data, Error>?
        private var progress: Progress?
        private var result: Result<Data, Error>?
        private var cancelled = false

        func install(_ continuation: CheckedContinuation<Data, Error>) -> Bool {
            lock.lock()
            if let result {
                lock.unlock(); continuation.resume(with: result); return false
            }
            self.continuation = continuation
            lock.unlock(); return true
        }
        func attach(_ progress: Progress) {
            lock.lock()
            let shouldCancel = cancelled
            if result == nil { self.progress = progress }
            lock.unlock()
            // Cancellation can win before loadDataRepresentation returns its
            // Progress; a synchronous callback can also finish before attach.
            if shouldCancel { progress.cancel() }
        }
        func complete(_ result: Result<Data, Error>) {
            lock.lock()
            guard self.result == nil else { lock.unlock(); return }
            self.result = result
            let continuation = self.continuation
            self.continuation = nil; progress = nil
            lock.unlock()
            continuation?.resume(with: result)
        }
        func cancel() {
            lock.lock()
            guard result == nil else { lock.unlock(); return }
            cancelled = true
            result = .failure(CancellationError())
            let continuation = self.continuation, progress = self.progress
            self.continuation = nil; self.progress = nil
            lock.unlock()
            continuation?.resume(throwing: CancellationError())
            progress?.cancel()
        }
    }
}
