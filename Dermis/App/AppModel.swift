import DermisCore
import DermisDevice
import Foundation
import Observation

@MainActor
@Observable
final class AppModel {
    enum DeviceState: String {
        case disconnected = "Disconnected"
        case simulated = "Simulator"
        case connected = "Connected"
    }

    var deviceState: DeviceState = .disconnected
    var frame: SensorFrame?
    var frameRate = 0.0
    var droppedFrames: UInt64 = 0
    var latencyMilliseconds = 0.0
    var scenario: SimulatorScenario = .resting
    var selectedChannel = 0

    private let simulator = DermisSimulator()
    private var simulationTask: Task<Void, Never>?

    func connectSimulator() {
        disconnect()
        deviceState = .simulated
        simulationTask = Task { [weak self] in
            var sequence: UInt32 = 0
            while !Task.isCancelled {
                guard let self else { return }
                // The simulator produces device-rate frames. Phase 0 presents the
                // latest snapshot at UI cadence; recording will consume every frame.
                for _ in 0..<6 {
                    self.frame = self.simulator.frame(sequence: sequence, scenario: self.scenario)
                    sequence &+= 1
                }
                self.frameRate = 357.7
                self.latencyMilliseconds = 2.8
                try? await Task.sleep(for: .milliseconds(16))
            }
        }
    }

    func disconnect() {
        simulationTask?.cancel()
        simulationTask = nil
        deviceState = .disconnected
        frame = nil
        frameRate = 0
    }
}

