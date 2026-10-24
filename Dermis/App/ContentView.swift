import DermisCore
import DermisDevice
import SwiftUI

struct ContentView: View {
    @Bindable var model: AppModel
    @State private var section: AppSection? = .live
    @State private var inspectorPresented = true

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $section) { item in
                Label(item.title, systemImage: item.symbol)
                    .tag(item)
            }
            .navigationSplitViewColumnWidth(min: 176, ideal: 196, max: 240)
        } detail: {
            liveWorkspace
                .inspector(isPresented: $inspectorPresented) {
                    PodInspector(model: model)
                        .inspectorColumnWidth(min: 260, ideal: 300, max: 360)
                }
        }
        .toolbar { toolbar }
        .background(DermisColor.canvas)
        .task { model.connectSimulator() }
    }

    private var liveWorkspace: some View {
        VStack(spacing: 0) {
            HandViewport(frame: model.frame, selectedChannel: $model.selectedChannel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider().overlay(DermisColor.hairline)
            PodLanes(frame: model.frame, selectedChannel: $model.selectedChannel)
                .frame(height: 236)
            statusBar
        }
        .background(DermisColor.canvas)
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItemGroup {
            HStack(spacing: 6) {
                Circle()
                    .fill(model.deviceState == .disconnected ? DermisColor.fault : DermisColor.ok)
                    .frame(width: 7, height: 7)
                Text(model.deviceState.rawValue)
            }
            .accessibilityElement(children: .combine)

            Button(model.deviceState == .disconnected ? "Connect" : "Disconnect") {
                model.deviceState == .disconnected ? model.connectSimulator() : model.disconnect()
            }

            Picker("Scenario", selection: $model.scenario) {
                ForEach(SimulatorScenario.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .frame(width: 150)

            Button("Record", systemImage: "record.circle") {}
                .disabled(true)
            Button("Marker", systemImage: "bookmark") {}
                .disabled(true)
            Button("Tare", systemImage: "scope") {}
                .disabled(true)
        }
        ToolbarItem(placement: .primaryAction) {
            Button("Inspector", systemImage: "sidebar.right") { inspectorPresented.toggle() }
                .keyboardShortcut("i", modifiers: [.option, .command])
        }
    }

    private var statusBar: some View {
        HStack(spacing: 16) {
            Text(model.deviceState == .simulated ? "SIMULATED" : model.deviceState.rawValue.uppercased())
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(model.deviceState == .simulated ? DermisColor.warn : DermisColor.text2)
            status("Rate", value: String(format: "%.0f Hz", model.frameRate))
            status("Dropped", value: "\(model.droppedFrames)")
            status("Latency", value: String(format: "%.1f ms", model.latencyMilliseconds))
            Spacer()
            Text("No calibration · field units")
                .foregroundStyle(DermisColor.text3)
        }
        .font(.system(size: 11).monospacedDigit())
        .padding(.horizontal, 12)
        .frame(height: 27)
        .background(DermisColor.surface1)
    }

    private func status(_ label: String, value: String) -> some View {
        HStack(spacing: 4) {
            Text(label).foregroundStyle(DermisColor.text3)
            Text(value).foregroundStyle(DermisColor.text2)
        }
    }
}

private enum AppSection: String, CaseIterable, Identifiable {
    case live, sessions, calibration, diagnostics, rig, twin, integrations
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var symbol: String {
        switch self {
        case .live: "waveform.path.ecg"
        case .sessions: "clock.arrow.circlepath"
        case .calibration: "scope"
        case .diagnostics: "stethoscope"
        case .rig: "move.3d"
        case .twin: "rectangle.split.2x1"
        case .integrations: "point.3.connected.trianglepath.dotted"
        }
    }
}

