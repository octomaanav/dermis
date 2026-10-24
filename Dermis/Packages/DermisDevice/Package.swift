// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DermisDevice",
    platforms: [.macOS("15.0")],
    products: [.library(name: "DermisDevice", targets: ["DermisDevice"])],
    dependencies: [.package(path: "../DermisCore")],
    targets: [
        .target(name: "DermisDevice", dependencies: ["DermisCore"]),
        .testTarget(name: "DermisDeviceTests", dependencies: ["DermisDevice", "DermisCore"]),
    ]
)

