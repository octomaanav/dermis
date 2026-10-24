import DermisCore
import Foundation

public enum SimulatorScenario: String, CaseIterable, Sendable {
    case resting = "Resting"
    case pinch = "Pinch"
    case powerGrasp = "Power grasp"
    case slip = "Slip"
    case injectedFault = "Injected fault"
}

public struct DermisSimulator: Sendable {
    public init() {}

    public func frame(sequence: UInt32, scenario: SimulatorScenario = .resting) -> SensorFrame {
        let t = Double(sequence) / 357.7
        let locations: [PodLocation] = [.index, .thumb, .middle, .thenarAPB]
        let pods = (0..<4).map { channel in
            let phase = t * 1.7 + Double(channel) * 0.8
            let active: Double = switch scenario {
            case .resting: 0
            case .pinch: channel < 2 ? max(0, sin(t * 2.2)) * 9_000 : 0
            case .powerGrasp: max(0, sin(t * 1.4)) * 7_000
            case .slip: channel == 0 ? sin(t * 90) * 800 : 0
            case .injectedFault: sequence % 500 == 0 ? -47_000 : 0
            }
            let x = sin(phase * 1.1) * 30 + active * 0.12
            let y = cos(phase * 0.9) * 30 + active * 0.08
            let z = 27_000 + sin(phase) * 60 + active
            return PodSample(
                channel: channel,
                location: locations[channel],
                field: .init(xMicrotesla: x, yMicrotesla: y, zMicrotesla: z),
                temperatureCelsius: 22 + sin(t / 80 + Double(channel)) * 0.3
            )
        }
        return SensorFrame(
            sequence: sequence,
            deviceTimeMicroseconds: UInt64(t * 1_000_000),
            hostTimeNanoseconds: DispatchTime.now().uptimeNanoseconds,
            pods: pods,
            simulated: true
        )
    }
}

