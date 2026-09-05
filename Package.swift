// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TurboCider",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "TurboCiderKit", targets: ["TurboCiderKit"]),
        .executable(name: "turbocider-swift", targets: ["TurboCiderSwiftCLI"]),
        .executable(name: "TurboCiderApp", targets: ["TurboCiderApp"]),
        .executable(name: "turbocider-swift-selftest", targets: ["TurboCiderKitSelfTest"]),
    ],
    targets: [
        .target(name: "TurboCiderKit"),
        .executableTarget(
            name: "TurboCiderSwiftCLI",
            dependencies: ["TurboCiderKit"]
        ),
        .executableTarget(
            name: "TurboCiderApp",
            dependencies: ["TurboCiderKit"]
        ),
        .executableTarget(
            name: "TurboCiderKitSelfTest",
            dependencies: ["TurboCiderKit"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
