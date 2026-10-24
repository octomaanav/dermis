import DermisCore
import SwiftUI

struct PodLanes: View {
    let frame: SensorFrame?
    @Binding var selectedChannel: Int

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                SectionLabel(title: "Track lanes")
                Spacer()
                Text("10 s · |B|")
                    .font(.system(size: 11).monospacedDigit())
                    .foregroundStyle(DermisColor.text3)
            }
            .padding(.horizontal, 12)
            .frame(height: 30)

            ForEach(frame?.pods ?? []) { pod in
                lane(pod)
            }
        }
        .background(DermisColor.surface1)
    }

    private func lane(_ pod: PodSample) -> some View {
        Button {
            selectedChannel = pod.channel
        } label: {
            HStack(spacing: 10) {
                Rectangle()
                    .fill(DermisColor.podSeries[pod.channel % DermisColor.podSeries.count])
                    .frame(width: 3)
                Text(pod.location.displayName)
                    .font(.system(size: 12, weight: .medium))
                    .frame(width: 90, alignment: .leading)
                GeometryReader { geometry in
                    Canvas { context, size in
                        let center = size.height / 2
                        var baseline = Path()
                        baseline.move(to: CGPoint(x: 0, y: center))
                        baseline.addLine(to: CGPoint(x: size.width, y: center))
                        context.stroke(baseline, with: .color(DermisColor.hairline), lineWidth: 1)

                        let level = min(max(pod.field.magnitudeMillitesla / 50, 0), 1)
                        var trace = Path()
                        trace.move(to: CGPoint(x: 0, y: center))
                        trace.addCurve(
                            to: CGPoint(x: size.width, y: center - level * center * 0.72),
                            control1: CGPoint(x: size.width * 0.45, y: center),
                            control2: CGPoint(x: size.width * 0.78, y: center - level * center * 0.72)
                        )
                        context.stroke(trace, with: .color(DermisColor.podSeries[pod.channel % 4]), lineWidth: 1.25)
                    }
                }
                Text(String(format: "%.2f", pod.field.magnitudeMillitesla))
                    .font(.system(size: 13).monospacedDigit())
                    .frame(width: 55, alignment: .trailing)
                Text("mT")
                    .font(.system(size: 11))
                    .foregroundStyle(DermisColor.text3)
                    .frame(width: 22, alignment: .leading)
            }
            .padding(.horizontal, 12)
            .frame(height: 48)
            .background(selectedChannel == pod.channel ? DermisColor.surface2 : Color.clear)
            .overlay(alignment: .top) { Divider().overlay(DermisColor.hairline) }
        }
        .buttonStyle(.plain)
    }
}

