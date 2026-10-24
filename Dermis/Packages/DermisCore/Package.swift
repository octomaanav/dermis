// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DermisCore",
    platforms: [.macOS("15.0")],
    products: [.library(name: "DermisCore", targets: ["DermisCore"])],
    targets: [
        .target(name: "DermisCore"),
        .testTarget(name: "DermisCoreTests", dependencies: ["DermisCore"]),
    ]
)

