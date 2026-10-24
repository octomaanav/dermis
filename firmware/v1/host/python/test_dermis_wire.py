"""Tests for the Dermis Wire Schema v1.0 reference implementation.

    python -m unittest -v test_dermis_wire.py
"""
import json
import math
import os
import random
import unittest

import dermis_wire as dw

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", ".."))


class CrcAndCobs(unittest.TestCase):
    def test_crc_check_value(self):
        self.assertEqual(dw.crc32(b"123456789"), 0xCBF43926)

    def test_cobs_reference_examples(self):
        for a, b in dw.COBS_EXAMPLES:
            self.assertEqual(dw.cobs_encode(bytes.fromhex(a)).hex(), b, a[:16])
            self.assertEqual(dw.cobs_decode(bytes.fromhex(b)).hex(), a, b[:16])

    def test_cobs_roundtrip_random(self):
        rng = random.Random(1)
        for n in list(range(0, 600)) + [1016, 2032, 4096]:
            data = bytes(rng.choice([0, 0, 1, 7, 0xFF, rng.randrange(256)]) for _ in range(n))
            enc = dw.cobs_encode(data)
            self.assertNotIn(0, enc)
            self.assertLessEqual(len(enc), n + n // 254 + 1)
            self.assertEqual(dw.cobs_decode(enc), data)

    def test_cobs_long_nonzero_runs(self):
        for n in (253, 254, 255, 508, 509):
            data = bytes([0x5A]) * n
            self.assertEqual(dw.cobs_decode(dw.cobs_encode(data)), data)


class Packets(unittest.TestCase):
    def test_roundtrip_every_message(self):
        for name, wire in dw.vector_packets().items():
            self.assertEqual(wire[-1], 0)
            self.assertNotIn(0, wire[:-1])
            pkts = dw.StreamDecoder().feed(wire)
            self.assertEqual(len(pkts), 1, name)

    def test_corruption_is_rejected_and_stream_resyncs(self):
        wire = dw.vector_packets()["frame"]
        bad = bytearray(wire)
        bad[40] ^= 0x01                      # flip one bit inside the packet
        dec = dw.StreamDecoder()
        pkts = dec.feed(bytes(bad) + wire)
        self.assertEqual(len(pkts), 1)
        self.assertEqual(dec.errors["crc"] + dec.errors["cobs"] + dec.errors["length"], 1)

    def test_garbage_and_split_reads(self):
        rng = random.Random(2)
        wires = list(dw.vector_packets().values()) * 5
        stream = bytearray()
        for w in wires:
            stream += bytes(rng.randrange(1, 256) for _ in range(rng.randrange(0, 40)))  # junk, no 0x00
            stream += b"\x00" + w                                                        # junk ends at a delimiter
        dec = dw.StreamDecoder()
        got = []
        i = 0
        while i < len(stream):                     # feed in random-size pieces
            step = rng.randrange(1, 64)
            got += dec.feed(bytes(stream[i:i + step]))
            i += step
        self.assertEqual(len(got), len(wires))

    def test_wrong_major_version_rejected(self):
        wire = dw.encode_packet(dw.Msg.LOG, b"hi", 1, 1, version=0x20)
        dec = dw.StreamDecoder()
        self.assertEqual(dec.feed(wire), [])
        self.assertEqual(dec.errors["version"], 1)

    def test_newer_minor_version_accepted(self):
        wire = dw.encode_packet(dw.Msg.LOG, b"hi", 1, 1, version=0x13)
        self.assertEqual(len(dw.StreamDecoder().feed(wire)), 1)

    def test_canonical_hello_has_no_line_breaks(self):
        # A line-based text menu on the device must not see \n or \r in the first packet.
        wire = b"\x00" + dw.cmd_get_info(0)
        self.assertNotIn(0x0A, wire)
        self.assertNotIn(0x0D, wire)


class Frames(unittest.TestCase):
    def setUp(self):
        self.info = dw.DeviceInfo(dw.EXAMPLE_DEVICE_INFO)
        pkt = dw.StreamDecoder().feed(dw.vector_packets()["frame"])[0]
        self.pkt = pkt
        self.frame = dw.parse_frame(pkt)

    def test_sizes(self):
        self.assertEqual(len(self.pkt.payload), 16 + 4 * 12 + 4 * 24)
        self.assertEqual(len(dw.vector_packets()["frame"]), 182)

    def test_raw_fields(self):
        f = self.frame
        self.assertEqual((f.frame_counter, f.config_id, f.flags, f.acq_us, f.t_us), (1234, 3, 0x08, 1650, 12345678))
        self.assertEqual((f.mags[0].x, f.mags[0].y, f.mags[0].z), (52, -118, 6973))
        self.assertEqual(f.imus[1].acc, (-1210, 2950, 2700))
        self.assertEqual(f.imus[3].sensortime, 0xA1B2E8)

    def test_si_conversion(self):
        d = dw.decode_frame(self.frame, self.info)
        idx = d["pods"]["index_tip"]
        self.assertAlmostEqual(idx["mag"]["b_ut"][2], 6973 * 3.872, places=6)
        self.assertAlmostEqual(idx["mag"]["temp_c"], 25 + (46109 - 46244) / 45.2, places=6)
        self.assertEqual(idx["mag"]["t_us"], 12345678 + 1344 // 2)
        self.assertAlmostEqual(idx["imu"]["acc_mps2"][2], 4090 * 9.80665 / 4096, places=6)
        self.assertAlmostEqual(idx["imu"]["gyr_rads"][0], 12 * math.pi / 180 / 16.4, places=8)
        self.assertAlmostEqual(idx["imu"]["temp_c"], 23 + 552 / 512, places=6)
        self.assertEqual(idx["imu"]["t_us"], 12345678 + 212)
        thenar = d["pods"]["thenar_apb"]
        self.assertFalse(thenar["mag"]["valid"])
        self.assertIsNone(thenar["mag"]["b_ut"])
        self.assertEqual(thenar["mag"]["flags"], ["BAD_LENGTH"])
        self.assertEqual(d["pods"]["middle_tip"]["imu"]["flags"], ["STALE"])
        self.assertEqual(d["flags"], ["MARKER"])

    def test_config_mismatch_raises(self):
        info = dict(dw.EXAMPLE_DEVICE_INFO, config_id=4)
        with self.assertRaises(dw.ConfigMismatch):
            dw.decode_frame(self.frame, dw.DeviceInfo(info))

    def test_larger_records_from_a_future_minor_are_skipped_cleanly(self):
        f = self.frame
        payload = bytearray(dw.FRAME_HEADER.pack(f.frame_counter, f.config_id, f.flags, 0,
                                                 len(f.mags), 14, len(f.imus), 28, f.acq_us, 0))
        for m in f.mags:
            payload += dw.MAG_REC.pack(m.ch, m.status, m.flags, 0, m.x, m.y, m.z, m.t_raw) + b"\xAA\xBB"
        for r in f.imus:
            payload += dw.IMU_REC.pack(r.ch, r.flags, r.t_rel_us, *r.acc, *r.gyr, r.temp_raw, 0,
                                       r.sensortime) + b"\x01\x02\x03\x04"
        payload += b"\x99" * 5                                   # trailing extension block
        pkt = dw.Packet(dw.Msg.FRAME, 0x11, 1, f.t_us, bytes(payload))
        g = dw.parse_frame(pkt)
        self.assertEqual([vars(m) for m in g.mags], [vars(m) for m in f.mags])
        self.assertEqual([vars(r) for r in g.imus], [vars(r) for r in f.imus])

    def test_csv_row_has_every_column(self):
        cols = dw.csv_columns(self.info)
        row = dw.frame_to_row(self.frame, self.info)
        self.assertEqual(set(cols), set(row))
        self.assertEqual(len(cols), 5 + 4 * len(dw.POD_COLUMNS))


class ConfigAndMount(unittest.TestCase):
    def test_mount_rotation(self):
        self.assertEqual(dw.mount_matrix("+X+Y+Z"), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dw.mount_matrix("-Y+X+Z")
        for bad in ("+X+X+Z", "+X+Y-Z", "X+Y+Z+", "+X+Y+W"):
            with self.assertRaises(ValueError):
                dw.mount_matrix(bad)

    def test_set_config_applies_and_bumps_config_id(self):
        new = dw.apply_set_config(dw.EXAMPLE_DEVICE_INFO, dw.EXAMPLE_SET_CONFIG)
        self.assertEqual(new["config_id"], dw.EXAMPLE_DEVICE_INFO["config_id"] + 1)
        self.assertEqual(new["mag"][1]["mount"], "-Y+X+Z")
        self.assertAlmostEqual(new["imu"][1]["acc_scale_mps2_per_lsb"], 9.80665 / 2048, places=9)
        self.assertEqual(dw.EXAMPLE_DEVICE_INFO["config_id"], 3)        # original untouched

    def test_set_config_rejects_bad_input(self):
        for patch in ({}, {"tref": 1}, {"mag": [{"ch": 9, "osr": 1}]}, {"mag": [{"ch": 0, "scale_ut_per_lsb": [1, 1, 1]}]},
                      {"imu": [{"ch": 0, "mount": "+X+X+Z"}]}, {"pods": [{"id": "", "location": "elbow",
                                                                         "mag_ch": 0, "imu_ch": 0}]}):
            with self.assertRaises((ValueError, KeyError)):
                dw.apply_set_config(dw.EXAMPLE_DEVICE_INFO, patch)


class SimulatedHub(unittest.TestCase):
    def test_capture_decodes_with_expected_faults(self):
        data = dw.simulate_capture(seconds=2.0)
        stats = dw.dump([data[i:i + 1000] for i in range(0, len(data), 1000)], quiet=True)
        self.assertEqual(stats["stream_errors"], {k: 0 for k in stats["stream_errors"]})
        self.assertEqual(stats["lost_frames"], 1)
        self.assertEqual(stats["invalid_mag"], 1)
        self.assertGreater(stats["frames"], 690)

    def test_command_replies(self):
        sim = dw.Sim()
        dec = dw.StreamDecoder()

        def run(cmd):
            pkt = dw.StreamDecoder().feed(cmd)[0]
            return dec.feed(b"".join(sim.handle(pkt, 5_000_000)))

        r = run(dw.cmd_get_info(0))
        self.assertEqual([p.type for p in r], [dw.Msg.ACK, dw.Msg.DEVICE_INFO])
        r = run(dw.cmd_sync_ping(1, 777))
        self.assertEqual(dw.SYNC_PONG.unpack(r[0].payload), (777, 5_000_000))
        r = run(dw.cmd_start(2, 0))
        self.assertEqual(dw.ACK.unpack_from(r[0].payload)[2], dw.AckStatus.OK)
        r = run(dw.cmd_set_config(3, {"hand": "left"}))
        self.assertEqual(dw.ACK.unpack_from(r[0].payload)[2], dw.AckStatus.BAD_STATE)
        run(dw.cmd_stop(4))
        r = run(dw.cmd_set_config(5, {"hand": "left"}))
        self.assertEqual([p.type for p in r], [dw.Msg.ACK, dw.Msg.DEVICE_INFO, dw.Msg.EVENT])
        self.assertEqual(json.loads(r[1].payload)["config_id"], 4)
        r = run(dw.encode_packet(0x9F, b"", 6))
        self.assertEqual(dw.ACK.unpack_from(r[0].payload)[2], dw.AckStatus.UNKNOWN_TYPE)


class KitFiles(unittest.TestCase):
    """The files shipped in the kit must match what this code generates."""

    def _load(self, *parts):
        path = os.path.join(KIT, *parts)
        if not os.path.exists(path):
            self.skipTest(f"{path} not present")
        with open(path) as fh:
            return json.load(fh)

    def test_vectors_file_is_current(self):
        self.assertEqual(self._load("vectors", "test_vectors.json"), json.loads(json.dumps(dw.build_vectors())))

    def test_examples_validate_against_schemas(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("pip install jsonschema to run schema checks")
        pairs = [("device_info.json", "device_info.schema.json"), ("stats.json", "stats.schema.json"),
                 ("set_config.json", "set_config.schema.json"), ("decoded_frame.json", "decoded_frame.schema.json")]
        for example, schema in pairs:
            jsonschema.validate(self._load("examples", example), self._load("schemas", schema))


if __name__ == "__main__":
    unittest.main()
