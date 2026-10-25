import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { LatestState } from './module_bindings/types';

type Props = { state: LatestState | undefined; stale: boolean; task: string | undefined };

/** A rendering-only view of the subscribed cup/gripper state. It has no browser physics. */
export default function RobotScene({ state, stale, task }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef({ state, stale });
  latest.current = { state, stale };

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07111a');
    const camera = new THREE.PerspectiveCamera(40, 1, .01, 20);
    const assistHand = task === 'assist_hand';
    camera.position.set(assistHand ? .34 : .47, assistHand ? .34 : .34, assistHand ? .50 : .58);
    camera.lookAt(assistHand ? .07 : .0, assistHand ? .16 : .11, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    element.replaceChildren(renderer.domElement);
    scene.add(new THREE.HemisphereLight('#bdefff', '#071018', 2.1));
    const key = new THREE.DirectionalLight('#8ad8ff', 2.4); key.position.set(.3, .7, .45); scene.add(key);

    const table = new THREE.Mesh(new THREE.BoxGeometry(1.4, .05, 1), new THREE.MeshStandardMaterial({ color: '#142638', roughness: .85 }));
    table.position.y = -.04; scene.add(table);
    const catchPlane = new THREE.Mesh(new THREE.CylinderGeometry(.19, .19, .006, 40), new THREE.MeshStandardMaterial({ color: '#e59535', transparent: true, opacity: .25 }));
    catchPlane.position.set(0, -.008, 0); scene.add(catchPlane);

    // The worker publishes cup pose and gripper-joint opening, not a full robot-link pose.
    // This is consequently an honest parallel-jaw representation, anchored around the reported cup.
    const gripper = new THREE.Group(); gripper.visible = !assistHand; scene.add(gripper);
    const gripperMaterial = new THREE.MeshStandardMaterial({ color: '#6c91a9', metalness: .52, roughness: .31 });
    const palm = new THREE.Mesh(new THREE.BoxGeometry(.19, .05, .07), gripperMaterial);
    palm.position.y = .245; gripper.add(palm);
    const rail = new THREE.Mesh(new THREE.BoxGeometry(.045, .12, .048), gripperMaterial);
    const left = rail.clone(); const right = rail.clone(); left.position.y = right.position.y = .155; gripper.add(left, right);
    const padMaterial = new THREE.MeshStandardMaterial({ color: '#48d5e9', emissive: '#073c48', emissiveIntensity: .75, roughness: .3 });
    const leftPad = new THREE.Mesh(new THREE.BoxGeometry(.016, .07, .042), padMaterial); const rightPad = leftPad.clone();
    leftPad.position.set(.022, -.07, 0); rightPad.position.set(-.022, -.07, 0); left.add(leftPad); right.add(rightPad);
    const leftContact = new THREE.Mesh(new THREE.SphereGeometry(.012, 16, 12), new THREE.MeshStandardMaterial({ color: '#7efff4', emissive: '#1bcfbf', emissiveIntensity: .9 }));
    const rightContact = leftContact.clone(); leftContact.position.copy(leftPad.position); rightContact.position.copy(rightPad.position); left.add(leftContact); right.add(rightContact);

    const cup = new THREE.Group(); scene.add(cup);
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(.055, .048, .17, 32, 1, true), new THREE.MeshPhysicalMaterial({ color: '#bcecff', transparent: true, opacity: .38, roughness: .08, transmission: .35, side: THREE.DoubleSide }));
    glass.position.y = .085; cup.add(glass);
    const fill = new THREE.Mesh(new THREE.CylinderGeometry(.050, .046, .01, 32), new THREE.MeshStandardMaterial({ color: '#2ea8e9', transparent: true, opacity: .72 }));
    cup.add(fill);
    const rim = new THREE.Mesh(new THREE.TorusGeometry(.055, .004, 10, 32), new THREE.MeshStandardMaterial({ color: '#d8f5ff', metalness: .25 })); rim.rotation.x = Math.PI / 2; rim.position.y = .17; cup.add(rim);
    if (assistHand) {
      const handle = new THREE.Mesh(new THREE.TorusGeometry(.033, .006, 8, 24, Math.PI * 1.35), new THREE.MeshStandardMaterial({ color: '#d8f5ff', metalness: .25, roughness: .25 }));
      handle.rotation.y = Math.PI / 2; handle.position.set(0, .09, -.052); cup.add(handle);
    }

    // The assistive simulation publishes nine actual phalanx angles: thumb,
    // index, then middle. This is a three-finger state rendering, not a second
    // physics engine in the browser.
    const hand = new THREE.Group(); hand.visible = assistHand; hand.position.set(0, .22, 0); scene.add(hand);
    const handMaterial = new THREE.MeshStandardMaterial({ color: '#b67653', roughness: .62 });
    const sleeveMaterial = new THREE.MeshStandardMaterial({ color: '#24455c', roughness: .7 });
    const palmAssist = new THREE.Mesh(new THREE.BoxGeometry(.09, .026, .08), handMaterial); hand.add(palmAssist);
    const forearm = new THREE.Mesh(new THREE.CylinderGeometry(.026, .026, .11, 16), sleeveMaterial); forearm.rotation.z = Math.PI / 2; forearm.position.x = -.105; hand.add(forearm);
    const pivots: THREE.Group[] = [];
    const makeDigit = (base: THREE.Vector3, lengths: number[]) => {
      let parent = new THREE.Group(); parent.position.copy(base); hand.add(parent);
      lengths.forEach((length, index) => {
        const pivot = new THREE.Group(); parent.add(pivot); pivots.push(pivot);
        const segment = new THREE.Mesh(new THREE.CapsuleGeometry(.008 * (1 - index * .08), Math.max(.004, length - .016), 5, 10), handMaterial);
        segment.rotation.z = Math.PI / 2; segment.position.x = length / 2; pivot.add(segment);
        parent = pivot; parent.position.x = length;
      });
    };
    makeDigit(new THREE.Vector3(-.005, -.006, -.04), [.040, .030, .024]); // thumb
    makeDigit(new THREE.Vector3(.045, 0, -.019), [.046, .028, .022]); // index
    makeDigit(new THREE.Vector3(.045, 0, .019), [.046, .028, .022]); // middle

    const caption = document.createElement('div'); caption.className = 'scene-caption'; element.append(caption);
    let frame = 0;
    const targetCup = new THREE.Vector3(); const shownCup = new THREE.Vector3();
    const targetQuat = new THREE.Quaternion(); const shownQuat = new THREE.Quaternion();
    const resize = () => { const { width, height } = element.getBoundingClientRect(); camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix(); renderer.setSize(width, height, false); };
    const observer = new ResizeObserver(resize); observer.observe(element); resize();
    const draw = () => {
      const current = latest.current;
      if (current.state && !current.stale) {
        const s = current.state;
        // World data is in metres. assist_hand is z-up; the cup simulation has
        // its own presentation transform retained below.
        if (assistHand) {
          targetCup.set(s.cupPosition[0], s.cupPosition[2], -s.cupPosition[1]);
          targetQuat.set(s.cupOrientation[0], s.cupOrientation[2], -s.cupOrientation[1], s.cupOrientation[3]).normalize();
          pivots.forEach((pivot, index) => { pivot.rotation.z = -(s.robotJoints[index] ?? 0); });
        } else {
          targetCup.set((s.cupPosition[0] - .55) * 2.6, Math.max(0, s.cupPosition[2] - .05) * 2.6, s.cupPosition[1] * 2.6);
          targetQuat.set(s.cupOrientation[0], s.cupOrientation[2], -s.cupOrientation[1], s.cupOrientation[3]).normalize();
          const opening = Math.max(.055, Math.min(.18, ((Math.abs(s.robotJoints[7] ?? .03) + Math.abs(s.robotJoints[8] ?? .03)) * 2.5)));
          left.position.x = -opening / 2; right.position.x = opening / 2;
        }
        const leftLoad = Math.max(0, s.fingertipNormalLoads[0] ?? 0); const rightLoad = Math.max(0, s.fingertipNormalLoads[1] ?? 0);
        leftContact.scale.setScalar(.55 + Math.min(1.2, leftLoad / 3)); rightContact.scale.setScalar(.55 + Math.min(1.2, rightLoad / 3));
        (leftContact.material as THREE.MeshStandardMaterial).emissiveIntensity = .25 + Math.min(1.1, leftLoad / 2.5);
        (rightContact.material as THREE.MeshStandardMaterial).emissiveIntensity = .25 + Math.min(1.1, rightLoad / 2.5);
        fill.scale.y = Math.max(.04, Math.min(1, s.appliedDownwardLoad / 8)); fill.position.y = .01 + .075 * fill.scale.y;
        shownCup.lerp(targetCup, .18); shownQuat.slerp(targetQuat, .18); cup.position.copy(shownCup); cup.quaternion.copy(shownQuat);
        gripper.position.copy(shownCup); gripper.position.y += .005; gripper.position.z += .008;
        caption.textContent = assistHand
          ? `LIVE THREE-FINGER HAND · frame ${s.sequence} · all phalanx angles are subscribed state`
          : `LIVE SHARED SNAPSHOT · frame ${s.sequence} · contact pads show reported normal loads`;
      } else caption.textContent = current.stale ? 'STATE STALE · scene frozen at the last shared pose' : 'WAITING FOR A SHARED ROBOT SNAPSHOT';
      renderer.render(scene, camera); frame = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(frame); observer.disconnect(); renderer.dispose(); element.replaceChildren(); };
  }, [task]);
  return <div ref={host} className="robot-scene" aria-label="Subscribed robot and cup state visualization" />;
}
