import DermisCore
import RealityKit
import SwiftUI

struct HandViewport: View {
    let frame: SensorFrame?
    @Binding var selectedChannel: Int

    var body: some View {
        ZStack {
            DermisColor.canvas
            RealityView { content in
                let root = makeSpikeHand()
                content.add(root)

                let camera = PerspectiveCamera()
                camera.camera.fieldOfViewInDegrees = 30
                camera.look(
                    at: SIMD3<Float>(0, 0.035, 0),
                    from: SIMD3<Float>(0.19, 0.17, 0.34),
                    relativeTo: nil
                )
                content.add(camera)

                let key = DirectionalLight()
                key.light.intensity = 2_200
                key.look(at: .zero, from: SIMD3<Float>(-0.2, 0.3, 0.25), relativeTo: nil)
                content.add(key)
            }
            .accessibilityHidden(true)

            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        SectionLabel(title: "Live field")
                        Text("RealityKit spike · procedural geometry")
                            .font(.system(size: 12))
                            .foregroundStyle(DermisColor.text2)
                    }
                    Spacer()
                    Label("SIMULATED", systemImage: "waveform")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(DermisColor.warn)
                }
                Spacer()
                HStack(alignment: .bottom) {
                    podCallouts
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("FIELD MAGNITUDE")
                            .foregroundStyle(DermisColor.text3)
                        Text("0–50 mT")
                            .foregroundStyle(DermisColor.text2)
                    }
                    .font(.system(size: 11).monospacedDigit())
                }
            }
            .padding(16)
        }
    }

    private var podCallouts: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(frame?.pods ?? []) { pod in
                Button {
                    selectedChannel = pod.channel
                } label: {
                    HStack(spacing: 8) {
                        Rectangle()
                            .fill(DermisColor.podSeries[pod.channel % DermisColor.podSeries.count])
                            .frame(width: 2, height: 24)
                        Text(pod.location.displayName)
                            .frame(width: 84, alignment: .leading)
                        Text(String(format: "%.2f", pod.field.magnitudeMillitesla))
                            .font(.system(size: 15).monospacedDigit())
                            .frame(width: 58, alignment: .trailing)
                        Text("mT").foregroundStyle(DermisColor.text3)
                    }
                    .foregroundStyle(DermisColor.text1)
                    .padding(.horizontal, 8)
                    .frame(height: 32)
                    .background(selectedChannel == pod.channel ? DermisColor.surface3 : DermisColor.surface1)
                    .overlay(Rectangle().stroke(DermisColor.hairline, lineWidth: 1))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(pod.location.displayName), \(pod.field.magnitudeMillitesla, specifier: "%.2f") millitesla")
            }
        }
    }
}

@MainActor
private func makeSpikeHand() -> Entity {
    let root = Entity()
    root.name = "RealityKitHandSpike"

    let glove = SimpleMaterial(color: .init(white: 0.55, alpha: 1), roughness: 0.78, isMetallic: false)
    let podMaterial = SimpleMaterial(color: .init(white: 0.10, alpha: 1), roughness: 0.62, isMetallic: false)

    let palm = ModelEntity(mesh: .generateBox(size: SIMD3<Float>(0.095, 0.018, 0.115), cornerRadius: 0.025), materials: [glove])
    palm.position = SIMD3<Float>(0, 0, 0)
    root.addChild(palm)

    let fingerX: [Float] = [-0.038, -0.013, 0.014, 0.039]
    let fingerLength: [Float] = [0.078, 0.102, 0.094, 0.073]
    for index in 0..<4 {
        let finger = makeRoundedSegment(
            length: fingerLength[index], radius: 0.0105, material: glove
        )
        finger.orientation = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(1, 0, 0))
        finger.position = SIMD3<Float>(fingerX[index], 0, -0.057 - fingerLength[index] / 2)
        root.addChild(finger)
    }

    let thumb = makeRoundedSegment(length: 0.072, radius: 0.012, material: glove)
    thumb.orientation = simd_quatf(angle: -0.72, axis: SIMD3<Float>(0, 1, 0)) * simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(1, 0, 0))
    thumb.position = SIMD3<Float>(-0.067, 0, -0.012)
    root.addChild(thumb)

    let podPositions: [SIMD3<Float>] = [
        SIMD3<Float>(-0.013, 0.013, -0.158),
        SIMD3<Float>(-0.095, 0.013, -0.040),
        SIMD3<Float>(0.014, 0.013, -0.149),
        SIMD3<Float>(-0.027, 0.013, 0.015),
    ]
    for (index, position) in podPositions.enumerated() {
        let pod = ModelEntity(mesh: .generateCylinder(height: 0.006, radius: 0.014), materials: [podMaterial])
        pod.name = "pod-\(index)"
        pod.position = position
        root.addChild(pod)
    }
    return root
}

@MainActor
private func makeRoundedSegment(
    length: Float,
    radius: Float,
    material: SimpleMaterial
) -> Entity {
    let segment = Entity()
    let bodyLength = max(length - radius * 2, radius * 0.5)
    let body = ModelEntity(
        mesh: .generateCylinder(height: bodyLength, radius: radius),
        materials: [material]
    )
    segment.addChild(body)

    let capMesh = MeshResource.generateSphere(radius: radius)
    let top = ModelEntity(mesh: capMesh, materials: [material])
    top.position.y = bodyLength / 2
    segment.addChild(top)

    let bottom = ModelEntity(mesh: capMesh, materials: [material])
    bottom.position.y = -bodyLength / 2
    segment.addChild(bottom)
    return segment
}

