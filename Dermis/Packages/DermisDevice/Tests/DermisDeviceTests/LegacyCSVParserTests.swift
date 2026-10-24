import Testing
@testable import DermisDevice

@Test func parsesLegacyFrame() throws {
    let row = "1000,1,2,3,20,4,5,6,21,7,8,9,22,10,11,12,23,0x00000000"
    let frame = try LegacyCSVParser().parse(line: row, hostTimeNanoseconds: 99)
    #expect(frame.pods.count == 4)
    #expect(frame.pods[2].field.zMicrotesla == 9)
    #expect(frame.pods[3].temperatureCelsius == 23)
    #expect(frame.hostTimeNanoseconds == 99)
}

@Test func rejectsMetadataAndTruncatedRows() {
    #expect(throws: LegacyCSVError.metadata) { try LegacyCSVParser().parse(line: "#RATE,fps=357") }
    #expect(throws: LegacyCSVError.wrongColumnCount(3)) { try LegacyCSVParser().parse(line: "1,2,3") }
}

