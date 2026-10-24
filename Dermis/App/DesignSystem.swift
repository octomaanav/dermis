import SwiftUI

enum DermisColor {
    static let canvas = Color(hex: 0x0C0D0E)
    static let surface1 = Color(hex: 0x131416)
    static let surface2 = Color(hex: 0x191A1D)
    static let surface3 = Color(hex: 0x202226)
    static let hairline = Color(hex: 0x2B2D31)
    static let text1 = Color(hex: 0xECEAE6)
    static let text2 = Color(hex: 0xA6A5A1)
    static let text3 = Color(hex: 0x6F6F6C)
    static let ok = Color(hex: 0x6FAF8B)
    static let warn = Color(hex: 0xD9A54A)
    static let fault = Color(hex: 0xE06B5A)
    static let record = Color(hex: 0xE5484D)

    static let podSeries = [
        Color(hex: 0x6C9BD2),
        Color(hex: 0xD98A5F),
        Color(hex: 0x84B88A),
        Color(hex: 0xC58FB0),
    ]
}

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}

struct SectionLabel: View {
    let title: String

    var body: some View {
        Text(title.uppercased())
            .font(.system(size: 11, weight: .semibold))
            .tracking(0.66)
            .foregroundStyle(DermisColor.text3)
    }
}

