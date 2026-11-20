import Foundation

/// Frames newline-delimited worker events. The caller holds its stream lock
/// while consuming bytes, validating each payload and finishing the stream.
struct WorkerEventFraming {
    private static let prefix = Data("TC_EVENT\t".utf8)
    private static let maxLineBytes = 128 * 1024 + prefix.count

    private let invalidMessage: String
    private var pending = Data()
    private var discarding = false
    private var failed = false

    init(invalidMessage: String) {
        self.invalidMessage = invalidMessage
    }

    mutating func consume(_ data: Data, onEvent: (Data) throws -> Void) throws {
        guard !failed else { throw invalid() }
        do {
            for byte in data {
                if byte == 10 {
                    if !discarding, pending.starts(with: Self.prefix) {
                        try onEvent(Data(pending.dropFirst(Self.prefix.count)))
                    }
                    pending.removeAll(keepingCapacity: true)
                    discarding = false
                } else if !discarding {
                    pending.append(byte)
                    if pending.count > Self.maxLineBytes {
                        guard !pending.starts(with: Self.prefix) else { throw invalid() }
                        pending.removeAll(keepingCapacity: true)
                        discarding = true
                    }
                }
            }
        } catch {
            failed = true
            throw error
        }
    }

    func finish() throws {
        guard !failed, !pending.starts(with: Self.prefix) else { throw invalid() }
    }

    private func invalid() -> NativeFailure { NativeFailure(message: invalidMessage) }
}
