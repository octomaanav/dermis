// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DermisRender",
    platforms: [.macOS("15.0")],
    products: [.library(name: "DermisRender", targets: ["DermisRender"])],
    dependencies: [.package(path: "../DermisCore")],
    targets: [.target(name: "DermisRender", dependencies: ["DermisCore"])]
)

