import SwiftUI

@main
struct DermisStudioApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .preferredColorScheme(.dark)
                .frame(minWidth: 1280, minHeight: 800)
        }
        .defaultSize(width: 1440, height: 900)
        .commands {
            CommandMenu("Device") {
                Button("Connect simulator") { model.connectSimulator() }
                    .keyboardShortcut("k", modifiers: [.command])
                Button("Disconnect") { model.disconnect() }
            }
            CommandMenu("Capture") {
                Button("Record") {}
                    .keyboardShortcut("r", modifiers: [])
                    .disabled(true)
                Button("Marker") {}
                    .keyboardShortcut("m", modifiers: [])
                    .disabled(true)
                Button("Tare") {}
                    .keyboardShortcut("t", modifiers: [])
                    .disabled(true)
            }
        }
    }
}

