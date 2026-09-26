import Foundation
import Darwin

struct WorkerProcessIdentity: Codable, Sendable {
    let pid: Int32
    let processGroup: Int32
    let startSeconds: UInt64
    let startMicroseconds: UInt64
    let bootSession: String
    enum Observation { case present, exited, unknown }
    enum Failure: Error { case unavailable, invalid }
    private static func boot() throws -> String {
        var size = 0
        guard sysctlbyname("kern.bootsessionuuid", nil, &size, nil, 0) == 0, size > 1, size < 256 else { throw Failure.unavailable }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("kern.bootsessionuuid", &bytes, &size, nil, 0) == 0 else { throw Failure.unavailable }
        let value = String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        guard UUID(uuidString: value) != nil else { throw Failure.invalid }
        return value.lowercased()
    }
    static func capture(_ pid: Int32) throws -> Self {
        var info = proc_bsdinfo()
        guard pid > 1, proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, Int32(MemoryLayout.size(ofValue: info))) == MemoryLayout.size(ofValue: info),
              info.pbi_pid == UInt32(pid), info.pbi_pgid == UInt32(pid), info.pbi_ppid == UInt32(getpid()), info.pbi_start_tvsec > 0 else { throw Failure.unavailable }
        return Self(pid: pid, processGroup: pid, startSeconds: info.pbi_start_tvsec,
                    startMicroseconds: info.pbi_start_tvusec, bootSession: try boot())
    }
    /// Read-only restart check. Never signal a process by a persisted PID: a
    /// birth-time check followed by kill still has a PID-reuse race.
    func observe() -> Observation {
        guard pid > 1, processGroup == pid, startSeconds > 0, startMicroseconds < 1_000_000,
              UUID(uuidString: bootSession) != nil, let currentBoot = try? Self.boot() else { return .unknown }
        if currentBoot != bootSession.lowercased() { return .exited }
        var info = proc_bsdinfo()
        errno = 0
        let count = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, Int32(MemoryLayout.size(ofValue: info)))
        let error = errno
        if count == MemoryLayout.size(ofValue: info) {
            if info.pbi_start_tvsec == startSeconds && info.pbi_start_tvusec == startMicroseconds {
                return info.pbi_pgid == UInt32(processGroup) ? .present : .unknown
            }
            // Original leader has exited, but descendants may still own PGID.
            return kill(-processGroup, 0) == -1 && errno == ESRCH ? .exited : .unknown
        }
        if count == 0 && error == ESRCH {
            return kill(-processGroup, 0) == -1 && errno == ESRCH ? .exited : .unknown
        }
        return .unknown
    }
}

struct WorkerLaunchAdmission: Sendable {
    let journal: URL
    let jobID: UUID
    let requestID: UUID
    let requestDigest: String
    struct Record: Codable, Sendable {
        var schemaVersion = 1
        let jobID: UUID
        let requestID: UUID
        let requestDigest: String
        let process: WorkerProcessIdentity
    }
    /// Missing, malformed or mismatched journals are not proof of exit.
    func observe() -> WorkerProcessIdentity.Observation {
        do {
            let fd = open(journal.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
            guard fd >= 0 else { return .unknown }
            defer { close(fd) }
            var before = stat(), after = stat()
            guard fstat(fd, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
                  before.st_size > 0, before.st_size <= 16 * 1024, before.st_nlink == 1 else { return .unknown }
            var data = Data(count: Int(before.st_size))
            let complete = data.withUnsafeMutableBytes { bytes -> Bool in
                var offset = 0
                while offset < bytes.count {
                    let count = read(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                    if count < 0 && errno == EINTR { continue }
                    if count <= 0 { return false }
                    offset += count
                }
                return true
            }
            guard complete, fstat(fd, &after) == 0, before.st_size == after.st_size,
                  before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec, before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
                  before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec, before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else { return .unknown }
            let record = try JSONDecoder().decode(Record.self, from: data)
            guard record.schemaVersion == 1, record.jobID == jobID, record.requestID == requestID,
                  record.requestDigest == requestDigest else { return .unknown }
            return record.process.observe()
        } catch { return .unknown }
    }
    /// Publish an immutable identity before writing the worker's admission byte.
    /// A private job directory is required; stale journals are never overwritten.
    func persist(_ process: WorkerProcessIdentity) throws {
        guard journal.isFileURL, !journal.path.contains("\0"), requestDigest.utf8.count == 64,
              requestDigest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw WorkerProcessIdentity.Failure.invalid }
        let record = Record(jobID: jobID, requestID: requestID, requestDigest: requestDigest, process: process)
        let data = try JSONEncoder().encode(record)
        let temporary = journal.deletingLastPathComponent().appendingPathComponent(".launch-\(UUID())")
        let fd = open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { close(fd); unlink(temporary.path) }
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                let count = write(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                offset += count
            }
        }
        guard fsync(fd) == 0, renamex_np(temporary.path, journal.path, UInt32(RENAME_EXCL)) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        let directory = open(journal.deletingLastPathComponent().path, O_RDONLY | O_CLOEXEC | O_DIRECTORY)
        guard directory >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { close(directory) }
        guard fsync(directory) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }
}
