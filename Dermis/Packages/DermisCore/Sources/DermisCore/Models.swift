import Foundation

public struct FieldVector: Sendable, Equatable, Codable {
    public var xMicrotesla: Double
    public var yMicrotesla: Double
    public var zMicrotesla: Double

    public init(xMicrotesla: Double, yMicrotesla: Double, zMicrotesla: Double) {
        self.xMicrotesla = xMicrotesla
        self.yMicrotesla = yMicrotesla
        self.zMicrotesla = zMicrotesla
    }

    public var magnitudeMicrotesla: Double {
        sqrt(xMicrotesla * xMicrotesla + yMicrotesla * yMicrotesla + zMicrotesla * zMicrotesla)
    }

    public var magnitudeMillitesla: Double { magnitudeMicrotesla / 1_000 }
}

public enum PodLocation: String, CaseIterable, Sendable, Codable {
    case unassigned
    case thumb
    case index
    case middle
    case thenarAPB

    public var displayName: String {
        switch self {
        case .unassigned: "Unassigned"
        case .thumb: "Thumb"
        case .index: "Index"
        case .middle: "Middle"
        case .thenarAPB: "Thenar (APB)"
        }
    }
}

public struct PodSample: Sendable, Equatable, Codable, Identifiable {
    public var id: Int { channel }
    public let channel: Int
    public var location: PodLocation
    public let field: FieldVector
    public let temperatureCelsius: Double
    public let rawStatus: UInt8
    public let rawFlags: UInt32

    public init(
        channel: Int,
        location: PodLocation = .unassigned,
        field: FieldVector,
        temperatureCelsius: Double,
        rawStatus: UInt8 = 0,
        rawFlags: UInt32 = 0
    ) {
        self.channel = channel
        self.location = location
        self.field = field
        self.temperatureCelsius = temperatureCelsius
        self.rawStatus = rawStatus
        self.rawFlags = rawFlags
    }
}

public struct SensorFrame: Sendable, Equatable, Codable {
    public let sequence: UInt32?
    public let deviceTimeMicroseconds: UInt64
    public let hostTimeNanoseconds: UInt64
    public let pods: [PodSample]
    public let simulated: Bool

    public init(
        sequence: UInt32?,
        deviceTimeMicroseconds: UInt64,
        hostTimeNanoseconds: UInt64,
        pods: [PodSample],
        simulated: Bool
    ) {
        self.sequence = sequence
        self.deviceTimeMicroseconds = deviceTimeMicroseconds
        self.hostTimeNanoseconds = hostTimeNanoseconds
        self.pods = pods
        self.simulated = simulated
    }
}

public struct SampleQuality: OptionSet, Sendable, Codable {
    public let rawValue: UInt16

    public init(rawValue: UInt16) { self.rawValue = rawValue }

    public static let statusError = Self(rawValue: 1 << 0)
    public static let identicalAxes = Self(rawValue: 1 << 1)
    public static let nearLimit = Self(rawValue: 1 << 2)
    public static let saturated = Self(rawValue: 1 << 3)
    public static let noMagnet = Self(rawValue: 1 << 4)
    public static let stale = Self(rawValue: 1 << 5)
    public static let sequenceGap = Self(rawValue: 1 << 6)
    public static let firmwareFault = Self(rawValue: 1 << 7)
}

public enum QualityEvaluator {
    public static let defaultWarnMicrotesla = 45_000.0
    public static let defaultFaultMicrotesla = 50_000.0
    public static let defaultMissingMagnetMicrotesla = 2_000.0

    public static func evaluate(
        _ sample: PodSample,
        warnMicrotesla: Double = defaultWarnMicrotesla,
        faultMicrotesla: Double = defaultFaultMicrotesla,
        missingMagnetMicrotesla: Double = defaultMissingMagnetMicrotesla
    ) -> SampleQuality {
        var quality: SampleQuality = []
        if sample.rawStatus & 0x10 != 0 { quality.insert(.statusError) }
        if sample.rawFlags != 0 { quality.insert(.firmwareFault) }

        let field = sample.field
        if field.xMicrotesla == field.yMicrotesla,
           field.yMicrotesla == field.zMicrotesla {
            quality.insert(.identicalAxes)
        }

        let magnitude = field.magnitudeMicrotesla
        if magnitude >= faultMicrotesla { quality.insert(.saturated) }
        else if magnitude >= warnMicrotesla { quality.insert(.nearLimit) }
        if magnitude < missingMagnetMicrotesla { quality.insert(.noMagnet) }
        return quality
    }
}

