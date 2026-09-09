import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let assets = root.appendingPathComponent("assets/branding")
let source = assets.appendingPathComponent("TurboDrop.png")
let sourceData = try Data(contentsOf: source)
guard let artwork = NSImage(data: sourceData) else { fatalError("Missing TurboDrop.png artwork") }
let encoded = sourceData.base64EncodedString()
let mark = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="TurboCider Turbo Drop">
<image width="512" height="512" href="data:image/png;base64,\(encoded)"/>
</svg>
"""
try mark.write(to: assets.appendingPathComponent("mark.svg"), atomically: true, encoding: .utf8)
let logo = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 280" role="img" aria-label="TurboCider — More from your Mac">
<rect width="920" height="280" rx="28" fill="#03152f"/>
<image x="16" y="16" width="248" height="248" href="data:image/png;base64,\(encoded)"/>
<text x="278" y="148" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="76" font-weight="750" fill="#ffffff">Turbo<tspan fill="#0cf0c5">Cider</tspan></text>
<text x="282" y="194" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="21" letter-spacing="2.5" fill="#b8c9de">MORE FROM YOUR MAC.</text>
</svg>
"""
try logo.write(to: assets.appendingPathComponent("logo.svg"), atomically: true, encoding: .utf8)

func png(_ size: Int, icon: Bool) -> Data {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.current?.imageInterpolation = .high
    let inset = icon ? CGFloat(size) * 0.055 : 0
    let frame = NSRect(x: inset, y: inset, width: CGFloat(size) - 2 * inset, height: CGFloat(size) - 2 * inset)
    NSBezierPath(roundedRect: frame, xRadius: frame.width * 0.22, yRadius: frame.height * 0.22).addClip()
    artwork.draw(in: frame, from: .zero, operation: .copy, fraction: 1)
    NSGraphicsContext.restoreGraphicsState()
    return bitmap.representation(using: .png, properties: [:])!
}
try png(256, icon: false).write(to: assets.appendingPathComponent("LogoMark.png"))
let iconset = assets.appendingPathComponent("AppIcon.iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = "icon_\(size)x\(size)\(scale == 2 ? "@2x" : "").png"
        try png(size * scale, icon: true).write(to: iconset.appendingPathComponent(name))
    }
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", assets.appendingPathComponent("AppIcon.icns").path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else { fatalError("iconutil failed") }
try FileManager.default.removeItem(at: iconset)
print("Generated Turbo Drop logo, sidebar mark and AppIcon.icns")
