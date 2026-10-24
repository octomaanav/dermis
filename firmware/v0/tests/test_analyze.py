from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from analyze import load_csv, save_histogram, summarize
except ModuleNotFoundError as error:  # Parser/build tests still run before host deps are installed.
    raise unittest.SkipTest(f"optional analysis dependencies not installed: {error}")
from protocol import HEADER


class AnalyzeTests(unittest.TestCase):
    def test_load_summarize_and_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(HEADER)
                for frame in range(20):
                    sensor_values: list[float] = []
                    for sensor in range(4):
                        sensor_values.extend(
                            [
                                1000.0 * sensor + frame,
                                2000.0 * sensor + frame,
                                3000.0 * sensor + frame * 10,
                                25.0 + sensor,
                            ]
                        )
                    flags = 0x10 if frame == 19 else 0
                    writer.writerow([frame * 2500, *sensor_values, hex(flags)])

            timestamps, values, flags = load_csv(path)
            self.assertEqual(timestamps.shape, (20,))
            self.assertEqual(values.shape, (20, 16))
            report = "\n".join(summarize(timestamps, values, flags))
            self.assertIn("median_interval=2500.0us", report)
            self.assertIn("S1:", report)
            self.assertIn("saturation_frames=1", report)
            histogram = save_histogram(path, timestamps)
            self.assertTrue(histogram.exists())
            self.assertGreater(histogram.stat().st_size, 1000)

    def test_summarize_accepts_numpy_inputs(self) -> None:
        timestamps = np.asarray([0, 1000, 2000], dtype=np.int64)
        values = np.zeros((3, 16), dtype=float)
        flags = np.zeros(3, dtype=np.uint32)
        self.assertTrue(summarize(timestamps, values, flags))


if __name__ == "__main__":
    unittest.main()

