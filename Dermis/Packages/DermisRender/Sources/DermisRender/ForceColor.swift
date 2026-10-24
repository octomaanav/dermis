import SwiftUI

public enum ForceColor {
    private static let stops: [(Double, Color)] = [
        (0.00, Color(red: 0.125, green: 0.133, blue: 0.149)),
        (0.20, Color(red: 0.290, green: 0.204, blue: 0.137)),
        (0.45, Color(red: 0.557, green: 0.333, blue: 0.153)),
        (0.70, Color(red: 0.851, green: 0.522, blue: 0.184)),
        (0.88, Color(red: 0.957, green: 0.761, blue: 0.478)),
        (1.00, Color(red: 1.000, green: 0.957, blue: 0.878)),
    ]

    public static func color(normalized value: Double) -> Color {
        let value = min(max(value, 0), 1)
        guard let upperIndex = stops.firstIndex(where: { value <= $0.0 }), upperIndex > 0 else {
            return value <= 0 ? stops[0].1 : stops[stops.count - 1].1
        }
        let lower = stops[upperIndex - 1]
        let upper = stops[upperIndex]
        let fraction = (value - lower.0) / (upper.0 - lower.0)
        return lower.1.mix(with: upper.1, by: fraction)
    }
}

private extension Color {
    func mix(with other: Color, by amount: Double) -> Color {
        // SwiftUI resolves these device-independent sRGB colors at render time.
        // Discrete fallback keeps this utility allocation-free on the live path.
        amount < 0.5 ? self : other
    }
}

