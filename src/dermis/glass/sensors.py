"""SYNTHETIC glove frames from simulator contact, in the dermis.glass.schema layout.

Every frame is labelled: top-level flag 'SYNTHETIC', device_id 'SYNTHETIC-GLASS-SIM', and every pod carries
FORCE_PROVISIONAL like the real firmware. The magnetic model is INVENTED for demonstration (not a calibration
of any real pod): per pod, b_ut = baseline + gains * [vertical shear, horizontal shear, normal] + noise,
with baselines taken from a real device frame so magnitudes look familiar. force_n imitates the firmware:
the simulated normal force plus noise, clamped to [0, FORCE_MAX_N] with the matching flags. Above the clamp
the pod cannot report how hard it is pressed (FORCE_CLAMPED_HIGH), which is true of the real firmware too.

Pod <- simulated contact: thumb_tip / index_tip / middle_tip read the total contact on that digit (all
three phalanges); thenar_apb reads the thenar pad on the palm. The hand is fixed in the world, so the IMU
reads gravity plus noise only (it carries no information about the glass in this demo).
"""
from __future__ import annotations

import numpy as np

from .schema import CONFIG_ID, FORCE_MAX_N, POD_CH, POD_IDS, POD_ORDER, SCHEMA_VERSION

BASELINE_UT = {"index_tip": (619.97, -1612.41, 9672.26), "thumb_tip": (-533.47, -595.94, -5172.99),
               "middle_tip": (2835.54, -507.03, 7434.24), "thenar_apb": (389.29, 264.33, -6733.41)}
GAIN_UT_PER_N = (-38.0, 22.0, 160.0)        # (vertical shear, horizontal shear, normal) -> (bx, by, |bz| grows)
B_NOISE_UT, F_NOISE_N, PERIOD_US = 3.0, 0.05, 33_333
DEVICE_ID = "SYNTHETIC-GLASS-SIM"


class SyntheticGlove:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.frame = 0
        self.t_us = 1_000_000

    def make(self, contact: dict, acc_extra_z: float = 0.0) -> dict:
        """contact[pod] = {'normal': N, 'shear_z': N (vertical friction carried), 'shear_h': N}.
        acc_extra_z: SYNTHETIC vertical hand acceleration (m/s^2) added to the IMU's gravity reading."""
        r = self.rng
        self.frame += 1
        self.t_us += PERIOD_US
        pods = {}
        for i, name in enumerate(("index_tip", "thumb_tip", "middle_tip", "thenar_apb")):
            c = contact[name]
            base = np.array(BASELINE_UT[name])
            sgn = np.sign(base[2])
            raw = np.array([GAIN_UT_PER_N[0] * c["shear_z"], GAIN_UT_PER_N[1] * c["shear_h"], sgn * GAIN_UT_PER_N[2] * c["normal"]])
            b = base + raw + r.normal(0, B_NOISE_UT, 3)
            f = c["normal"] + r.normal(0, F_NOISE_N)
            flags = ["FORCE_PROVISIONAL"]
            if f <= 0.0:
                f = 0.0
                flags.append("FORCE_CLAMPED_LOW")
            elif f >= FORCE_MAX_N:
                f = FORCE_MAX_N
                flags.append("FORCE_CLAMPED_HIGH")
            pods[name] = {"pod_id": POD_IDS[name], "mag": {
                "ch": POD_CH[name], "b_ut": [round(float(v), 3) for v in b], "temp_c": round(26.0 + 0.5 * i + r.normal(0, 0.02), 4),
                "force_n": round(float(f), 6), "force_valid": True, "t_us": self.t_us + 2300 + 120 * i, "status": 3,
                "flags": flags, "valid": True}}
        acc = np.array([0.0, 0.0, 9.80665 + acc_extra_z]) + r.normal(0, 0.02, 3)
        imu_t = self.t_us + 2800
        return {"schema_version": SCHEMA_VERSION, "device_id": DEVICE_ID, "frame": self.frame, "t_device_us": self.t_us,
                "t_host_ns": None, "config_id": CONFIG_ID, "flags": ["SYNTHETIC"], "pods": pods,
                "imu": {"imu_id": "IMU-0001", "ch": 0, "acc_mps2": [round(float(v), 6) for v in acc],
                        "gyr_rads": [round(float(v), 6) for v in r.normal(0, 0.003, 3)], "temp_c": round(27.0 + r.normal(0, 0.02), 5),
                        "t_us": imu_t, "sensortime": int(imu_t / 39.0625), "flags": [], "valid": True}}


assert set(POD_ORDER) == set(BASELINE_UT)
