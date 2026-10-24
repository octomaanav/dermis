import DermisCore
import SwiftUI

struct PodInspector: View {
    @Bindable var model: AppModel

    private var pod: PodSample? {
        model.frame?.pods.first(where: { $0.channel == model.selectedChannel })
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SectionLabel(title: "Selected pod")
                Text(pod?.location.displayName ?? "No pod")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(DermisColor.text1)

                readout("Field magnitude", value: pod.map { String(format: "%.2f", $0.field.magnitudeMillitesla) } ?? "—", unit: "mT")
                HStack { component("Bx", pod?.field.xMicrotesla); component("By", pod?.field.yMicrotesla); component("Bz", pod?.field.zMicrotesla) }

                Divider().overlay(DermisColor.hairline)
                SectionLabel(title: "Quality")
                qualityRows

                Divider().overlay(DermisColor.hairline)
                SectionLabel(title: "Calibration")
                Text("No force calibration. Field values remain in magnetic units.")
                    .font(.system(size: 12))
                    .foregroundStyle(DermisColor.text2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
        }
        .background(DermisColor.surface1)
    }

    private func readout(_ label: String, value: String, unit: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.system(size: 11)).foregroundStyle(DermisColor.text3)
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(value).font(.system(size: 22, weight: .medium).monospacedDigit())
                Text(unit).font(.system(size: 12)).foregroundStyle(DermisColor.text3)
            }
        }
    }

    private func component(_ label: String, _ value: Double?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.system(size: 11)).foregroundStyle(DermisColor.text3)
            Text(value.map { String(format: "%.0f", $0) } ?? "—")
                .font(.system(size: 13).monospacedDigit())
            Text("µT").font(.system(size: 11)).foregroundStyle(DermisColor.text3)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var qualityRows: some View {
        if let pod {
            let quality = QualityEvaluator.evaluate(pod)
            qualityRow("Stream", value: quality.isEmpty ? "Healthy" : "Flagged", good: quality.isEmpty)
            qualityRow("Calibration", value: "Uncalibrated", good: false)
            qualityRow("Temperature", value: String(format: "%.1f °C", pod.temperatureCelsius), good: true)
        } else {
            Text("No sample available.").foregroundStyle(DermisColor.text2)
        }
    }

    private func qualityRow(_ name: String, value: String, good: Bool) -> some View {
        HStack {
            Circle().fill(good ? DermisColor.ok : DermisColor.warn).frame(width: 7, height: 7)
            Text(name).foregroundStyle(DermisColor.text2)
            Spacer()
            Text(value).foregroundStyle(DermisColor.text1)
        }
        .font(.system(size: 12))
    }
}

