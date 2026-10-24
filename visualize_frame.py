import json
import math
import pybullet as p
import pybullet_data
import time

# --- Step 1: Load and Parse the JSON Data ---
# Load the single frame of data provided by the hardware team
FRAME_PATHS = ['decoded_frame.json', 'firmware/v1/examples/decoded_frame.json']
for path in FRAME_PATHS:
    try:
        with open(path, 'r') as f:
            frame_data = json.load(f)
        break
    except FileNotFoundError:
        continue
else:
    print("Error: decoded_frame.json not found in: " + ", ".join(FRAME_PATHS))
    exit()

# Extract the XYZ accelerometer arrays for the index and thumb pods
# The IMU measures gravity (roughly 9.8 m/s^2 pointing down).
index_acc = frame_data['pods']['index_tip']['imu']['acc_mps2']
thumb_acc = frame_data['pods']['thumb_tip']['imu']['acc_mps2']

# --- Step 2: Calculate the Bend Angles ---
def calculate_bend(acc):
    """
    Calculates a normalized bend angle (0.0 to 1.0) from accelerometer data.
    When the hand is flat, gravity is mostly on the Z axis.
    As the finger bends down, gravity shifts onto the X and Y axes.
    """
    x, y, z = acc

    # Calculate the angle between the Z-axis and the gravity vector using atan2
    # This gives us the tilt in radians
    tilt_radians = math.atan2(math.sqrt(x**2 + y**2), abs(z))

    # Normalize the angle to a 0.0 (flat) to 1.0 (fully bent/90 degrees) scale
    # We cap it at 1.0 just in case it overshoots 90 degrees (pi/2)
    normalized_bend = min(1.0, max(0.0, tilt_radians / (math.pi / 2)))
    return normalized_bend

index_bend = calculate_bend(index_acc)
thumb_bend = calculate_bend(thumb_acc)

print(f"Data Extracted (from {path}):")
print(f"- Index Accelerometer: {index_acc}")
print(f"- Thumb Accelerometer: {thumb_acc}")
print(f"Calculated Angles:")
print(f"- Index Bend (0-1): {index_bend:.2f}")
print(f"- Thumb Bend (0-1): {thumb_bend:.2f}")

# --- Step 3: PyBullet Simulation ---
print("\nStarting PyBullet Simulator...")
# Connect to the PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load a floor plane
planeId = p.loadURDF("plane.urdf")

# Load a default robot arm with a gripper (Franka Panda), standing on the floor
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True)

# Bend the arm so the gripper hangs in front of the base, pointing down (all-zero joints point it straight up)
home_pose = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, math.pi / 4]
for joint, angle in enumerate(home_pose):
    p.resetJointState(robotId, joint, angle)
    p.setJointMotorControl2(robotId, joint, p.POSITION_CONTROL, targetPosition=angle, force=240)

# Find the joint indices for the two gripper fingers on this specific robot model
finger_joint_indices = [9, 10]

# The Panda gripper takes values roughly between 0.0 (closed) and 0.04 (fully open).
# Pinch = average of index and thumb (as in sim_env.py); 0 (flat hand) = open (0.04), 1 (bent) = closed (0.0)
grip = (index_bend + thumb_bend) / 2
open_finger_pos = 0.04
frame_finger_pos = 0.04 - (0.04 * grip)

print(f"Grip from this frame: {grip:.2f} -> finger position {frame_finger_pos:.4f} (fully open is 0.04)")

# Zoom the camera onto the gripper and hide the side panels
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.stepSimulation()
gripper_pos = p.getLinkState(robotId, 11, computeForwardKinematics=True)[4]
p.resetDebugVisualizerCamera(0.35, 60, -20, gripper_pos)
label = p.addUserDebugText("", [0, 0, 0])

# Alternate every 1.5 s between a flat hand (open) and this frame's pinch, so the change is visible
print("Simulation running. Close the PyBullet window to exit.")
start, phase = time.monotonic(), None
try:
    while p.isConnected():
        show_frame = int((time.monotonic() - start) / 1.5) % 2 == 1
        if show_frame != phase:
            phase = show_frame
            target = frame_finger_pos if show_frame else open_finger_pos
            text = f"frame: index {index_bend:.2f} thumb {thumb_bend:.2f} -> grip {grip:.2f}" if show_frame else "flat hand: open"
            for joint in finger_joint_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=robotId,
                    jointIndex=joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=50
                )
            label = p.addUserDebugText(text, [gripper_pos[0], gripper_pos[1], gripper_pos[2] + 0.12],
                                       [0.8, 0, 0], 1.3, replaceItemUniqueId=label)
        p.stepSimulation()
        time.sleep(1./240.)
except (p.error, KeyboardInterrupt):  # window closed or Ctrl-C
    pass
