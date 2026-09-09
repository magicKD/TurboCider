// Reproducible vector mark and App icon. Run from the repository root:
// swift tools/branding/generate.swift
import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let assets = root.appendingPathComponent("assets/branding")
try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)

// The supplied Squeeze concept: pressing forces, silicon core, leaf, cider drop.
// Coordinates use a 512-square artboard and only M/L/C/Z path commands.
let shapes: [(String, String)] = [
    ("#292D30", "M 133 104 C 123 97 117 104 117 119 L 117 136 C 166 190 166 248 117 303 L 117 320 C 117 335 123 341 136 334 C 229 291 229 151 133 104 Z"),
    ("#292D30", "M 379 104 C 389 97 395 104 395 119 L 395 136 C 346 190 346 248 395 303 L 395 320 C 395 335 389 341 376 334 C 283 291 283 151 379 104 Z"),
    ("#56863F", "M 244 163 C 248 125 274 100 303 100 C 304 131 283 158 244 163 Z"),
    ("#A8ADAD", "M 238 178 L 274 178 C 288 178 294 185 294 198 L 294 235 C 294 248 287 255 274 255 L 238 255 C 225 255 218 248 218 235 L 218 198 C 218 185 225 178 238 178 Z"),
    ("#E8AA2B", "M 256 270 C 242 292 220 318 220 340 C 220 388 292 388 292 340 C 292 318 270 292 256 270 Z"),
    ("#FFF9E9", "M 237 319 C 231 335 231 353 244 361 C 250 365 252 357 247 353 C 238 346 240 333 244 322 C 247 315 240 312 237 319 Z")
]
let paths = shapes.map { "<path fill=\"\($0.0)\" d=\"\($0.1)\"/>" }.joined(separator: "\n")
let mark = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="TurboCider: squeeze more from Apple silicon">
\(paths)
</svg>
"""
try mark.write(to: assets.appendingPathComponent("mark.svg"), atomically: true, encoding: .utf8)
let logo = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 280" role="img" aria-label="TurboCider — more from your Mac">
<rect width="920" height="280" rx="28" fill="#FCFCFA"/>
<g transform="translate(2 -2) scale(.55)">\(paths)</g>
<text x="270" y="147" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="76" font-weight="750" fill="#292D30">Turbo<tspan fill="#56863F">Cider</tspan></text>
<text x="275" y="192" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="21" letter-spacing="2.5" fill="#666C6B">MORE FROM YOUR MAC.</text>
</svg>
"""
try logo.write(to: assets.appendingPathComponent("logo.svg"), atomically: true, encoding: .utf8)

func color(_ hex: String) -> NSColor {
    let n = UInt32(hex.dropFirst(), radix: 16)!
    return NSColor(srgbRed: CGFloat((n >> 16) & 255) / 255,
                   green: CGFloat((n >> 8) & 255) / 255, blue: CGFloat(n & 255) / 255, alpha: 1)
}
func path(_ source: String) -> NSBezierPath {
    let tokens = source.split(separator: " ").map(String.init)
    var i = 0
    func point() -> NSPoint { let x = Double(tokens[i])!, y = Double(tokens[i + 1])!; i += 2; return NSPoint(x: x, y: y) }
    let p = NSBezierPath()
    while i < tokens.count {
        let command = tokens[i]; i += 1
        switch command {
        case "M": p.move(to: point())
        case "L": p.line(to: point())
        case "C": let a = point(), b = point(), c = point(); p.curve(to: c, controlPoint1: a, controlPoint2: b)
        case "Z": p.close()
        default: fatalError("Unsupported path command")
        }
    }
    return p
}
func png(_ size: Int, background: Bool) -> Data {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    let t = NSAffineTransform(); t.translateX(by: 0, yBy: CGFloat(size)); t.scaleX(by: CGFloat(size) / 512, yBy: -CGFloat(size) / 512); t.concat()
    if background {
        color("#FCFCFA").setFill()
        NSBezierPath(roundedRect: NSRect(x: 28, y: 28, width: 456, height: 456), xRadius: 102, yRadius: 102).fill()
    }
    for (fill, commands) in shapes { color(fill).setFill(); path(commands).fill() }
    NSGraphicsContext.restoreGraphicsState()
    return bitmap.representation(using: .png, properties: [:])!
}
try png(256, background: false).write(to: assets.appendingPathComponent("LogoMark.png"))
let iconset = assets.appendingPathComponent("AppIcon.iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = "icon_\(size)x\(size)\(scale == 2 ? "@2x" : "").png"
        try png(size * scale, background: true).write(to: iconset.appendingPathComponent(name))
    }
}
let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", assets.appendingPathComponent("AppIcon.icns").path]
try process.run(); process.waitUntilExit()
guard process.terminationStatus == 0 else { fatalError("iconutil failed") }
try FileManager.default.removeItem(at: iconset)
print("Generated vector logo, sidebar mark and AppIcon.icns")
