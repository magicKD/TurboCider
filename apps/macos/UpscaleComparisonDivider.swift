import SwiftUI

struct UpscaleComparisonDivider: View {
    static let coordinateSpace = "upscaleComparisonCanvas"
    let visibleRect: CGRect
    @Binding var fraction: CGFloat
    var originalTitle = "原图"

    var body: some View {
        let divider = visibleRect.minX + fraction * visibleRect.width
        let targetWidth = min(34, visibleRect.width)
        let targetX = min(visibleRect.maxX - targetWidth / 2, max(visibleRect.minX + targetWidth / 2, divider))
        ZStack(alignment: .topLeading) {
            if divider - visibleRect.minX >= 56 {
                badge(originalTitle).position(x: visibleRect.minX + 30, y: visibleRect.minY + 21)
            }
            if visibleRect.maxX - divider >= 50 {
                badge("超分").position(x: visibleRect.maxX - 30, y: visibleRect.minY + 21)
            }
            Rectangle().fill(.white).shadow(color: .black.opacity(0.8), radius: 1)
                .frame(width: 2, height: visibleRect.height)
                .position(x: divider, y: visibleRect.midY).allowsHitTesting(false)
            ZStack {
                Color.clear
                Image(systemName: "arrow.left.and.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.black)
                    .frame(width: 30, height: 30).background(.white, in: Circle())
                    .shadow(color: .black.opacity(0.45), radius: 3, y: 1)
            }
            .frame(width: targetWidth, height: visibleRect.height)
            .contentShape(Rectangle())
            .focusable().focusEffectDisabled()
            .onKeyPress(.leftArrow) { move(by: -0.05); return .handled }
            .onKeyPress(.rightArrow) { move(by: 0.05); return .handled }
            .accessibilityLabel("\(originalTitle)与超分分界线")
            .accessibilityValue("\(originalTitle) \(Int((fraction * 100).rounded()))%")
            .accessibilityHint("向左右拖动，或用方向键调整")
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment: move(by: 0.05)
                case .decrement: move(by: -0.05)
                @unknown default: break
                }
            }
            .accessibilityIdentifier("upscaleComparisonDivider")
            .highPriorityGesture(DragGesture(minimumDistance: 0, coordinateSpace: .named(Self.coordinateSpace))
                .onChanged { drag in
                    guard visibleRect.width > 0 else { return }
                    fraction = min(1, max(0, (drag.location.x - visibleRect.minX) / visibleRect.width))
                })
            .position(x: targetX, y: visibleRect.midY)
        }
    }

    private func move(by delta: CGFloat) { fraction = min(1, max(0, fraction + delta)) }

    private func badge(_ title: String) -> some View {
        Text(title).font(.caption.weight(.semibold)).foregroundStyle(.white)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.black.opacity(0.66), in: Capsule())
            .accessibilityHidden(true).allowsHitTesting(false)
    }
}
