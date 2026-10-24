import Testing
@testable import DermisCore

@Test func flagsIdenticalAxesJunk() {
    let sample = PodSample(
        channel: 0,
        field: .init(xMicrotesla: -49_098.098, yMicrotesla: -49_098.098, zMicrotesla: -49_098.098),
        temperatureCelsius: 22
    )
    #expect(QualityEvaluator.evaluate(sample).contains(.identicalAxes))
}

@Test func enforcesAnalogLimits() {
    let warning = PodSample(channel: 0, field: .init(xMicrotesla: 46_000, yMicrotesla: 0, zMicrotesla: 0), temperatureCelsius: 22)
    let fault = PodSample(channel: 0, field: .init(xMicrotesla: 51_000, yMicrotesla: 0, zMicrotesla: 0), temperatureCelsius: 22)
    #expect(QualityEvaluator.evaluate(warning).contains(.nearLimit))
    #expect(QualityEvaluator.evaluate(fault).contains(.saturated))
}

