import Foundation

/// Selection follows the displayed order, keeping a stable anchor while Shift
/// extends or shrinks a range. Control/Command toggles individual items.
struct HistorySelection {
    private(set) var ids: Set<UUID> = []
    private(set) var anchor: UUID?

    mutating func select(_ id: UUID, orderedIDs: [UUID], toggle: Bool = false, range: Bool = false) {
        guard let end = orderedIDs.firstIndex(of: id) else { return }
        retain(Set(orderedIDs))
        if range, let anchor, let start = orderedIDs.firstIndex(of: anchor) {
            let interval = Set(orderedIDs[min(start, end)...max(start, end)])
            ids = toggle ? ids.union(interval) : interval
        } else {
            if toggle {
                if !ids.insert(id).inserted { ids.remove(id) }
            } else { ids = [id] }
            anchor = id
        }
    }

    mutating func selectAll(_ orderedIDs: [UUID]) {
        ids = Set(orderedIDs)
        anchor = orderedIDs.first
    }

    mutating func clear() { ids = []; anchor = nil }

    mutating func retain(_ available: Set<UUID>) {
        ids.formIntersection(available)
        if let anchor, !available.contains(anchor) { self.anchor = nil }
    }
}
