import DermisCore
import Foundation

public enum LegacyCSVError: Error, Equatable {
    case metadata
    case wrongColumnCount(Int)
    case invalidNumber(Int)
}

public struct LegacyCSVParser: Sendable {
    public init() {}

    public func parse(line: String, hostTimeNanoseconds: UInt64 = DispatchTime.now().uptimeNanoseconds) throws -> SensorFrame {
        let clean = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty, !clean.hasPrefix("#") else { throw LegacyCSVError.metadata }
        let columns = clean.split(separator: ",", omittingEmptySubsequences: false)
        guard columns.count == 18 else { throw LegacyCSVError.wrongColumnCount(columns.count) }

        guard let deviceTime = UInt64(columns[0]) else { throw LegacyCSVError.invalidNumber(0) }
        var pods: [PodSample] = []
        pods.reserveCapacity(4)
        for channel in 0..<4 {
            let base = 1 + channel * 4
            guard let x = Double(columns[base]) else { throw LegacyCSVError.invalidNumber(base) }
            guard let y = Double(columns[base + 1]) else { throw LegacyCSVError.invalidNumber(base + 1) }
            guard let z = Double(columns[base + 2]) else { throw LegacyCSVError.invalidNumber(base + 2) }
            guard let temperature = Double(columns[base + 3]) else { throw LegacyCSVError.invalidNumber(base + 3) }
            pods.append(PodSample(
                channel: channel,
                field: .init(xMicrotesla: x, yMicrotesla: y, zMicrotesla: z),
                temperatureCelsius: temperature
            ))
        }

        let flagText = String(columns[17])
        let flags: UInt32
        if flagText.hasPrefix("0x") || flagText.hasPrefix("0X") {
            flags = UInt32(flagText.dropFirst(2), radix: 16) ?? 0
        } else {
            flags = UInt32(flagText) ?? 0
        }

        if flags != 0 {
            pods = pods.map { pod in
                PodSample(
                    channel: pod.channel,
                    field: pod.field,
                    temperatureCelsius: pod.temperatureCelsius,
                    rawFlags: flags
                )
            }
        }
        return SensorFrame(sequence: nil, deviceTimeMicroseconds: deviceTime, hostTimeNanoseconds: hostTimeNanoseconds, pods: pods, simulated: false)
    }
}

