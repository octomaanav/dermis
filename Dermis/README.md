# Dermis Studio

Dermis Studio is a native macOS instrument application for the Dermis magnetic tactile-skin system. The repository contains the macOS app, local Swift packages, protocol/session documentation, and the validated ESP32-S3 firmware under `firmware/`.

## Generate the Xcode project

```bash
xcodegen generate
open DermisStudio.xcodeproj
```

The generated project is unsigned for local development. The source of truth is `project.yml`.

## Current toolchain issue on this Mac

The installed `/Applications/Xcode.app` reports a missing `libclang.dylib`, and the separate Command Line Tools installation is outdated/inconsistent. Repair or reinstall Xcode and its matching Command Line Tools before building. After repair:

```bash
xcodebuild -version
swift --version
swift test --package-path Packages/DermisCore
swift test --package-path Packages/DermisDevice
xcodebuild -project DermisStudio.xcodeproj -scheme DermisStudio -configuration Debug build
```

See `PLAN.md` for phased scope and `PROGRESS.md` for verified status.

