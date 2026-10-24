from __future__ import annotations

import unittest

from protocol import HEADER, estimate_dropped, parse_frame, saturation_sensors


class ProtocolTests(unittest.TestCase):
    def test_header_has_expected_columns(self) -> None:
        self.assertEqual(len(HEADER), 18)
        self.assertEqual(HEADER[0], "t_us")
        self.assertEqual(HEADER[-1], "flags")

    def test_parse_frame(self) -> None:
        fields = ["1000000", *(str(value) for value in range(16)), "0x00000020"]
        frame = parse_frame(",".join(fields))
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.t_us, 1_000_000)
        self.assertEqual(frame.sensor(1), (4.0, 5.0, 6.0, 7.0))
        self.assertEqual(frame.flags, 0x20)
        self.assertEqual(saturation_sensors(frame.flags), [2])

    def test_parse_ignores_metadata_and_malformed_lines(self) -> None:
        self.assertIsNone(parse_frame("#RATE,fps=300"))
        self.assertIsNone(parse_frame("Dermis menu"))
        self.assertIsNone(parse_frame("1,2,3"))

    def test_drop_estimate(self) -> None:
        self.assertEqual(estimate_dropped([0, 2500, 5000, 10_000, 12_500]), 1)
        self.assertEqual(estimate_dropped([0, 2500]), 0)


if __name__ == "__main__":
    unittest.main()

