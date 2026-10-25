# Dermis force calibrator

Independent ESP32-S3 firmware for synchronously recording a 1 kg HX711 load
cell and four MLX90393 sensors. It does not modify the production firmware in
`firmware/v0/src`.

## Wiring

| Device | ESP32-S3 |
|---|---|
| HX711 DOUT / DT | GPIO42 |
| HX711 SCK | GPIO2 |
| HX711 VCC | 3.3 V |
| HX711 GND | GND |
| MLX shared SCLK | GPIO18 |
| MLX shared MISO | GPIO13 |
| MLX shared MOSI | GPIO14 |
| MLX S1/S2/S3/S4 CS | GPIO21/4/16/17 |

## Build and flash

```bash
./scripts/build.sh
./scripts/flash.sh --upload-port /dev/cu.usbmodemXXXX
./scripts/monitor.sh --port /dev/cu.usbmodemXXXX
```

## Load-cell calibration

In the serial monitor:

1. Remove all load and enter `tare 30`.
2. Apply a known mass without touching the fixture sideways.
3. For a 500 g reference, enter `cal 500 30`.
4. Remove the mass and enter `status`.
5. Validate using other known masses before calibrating the magnetic patches.

The firmware stores `zero_raw` and `counts_per_N` in ESP32 NVS. A negative
`counts_per_N` is valid and only indicates load-cell polarity.

## Synchronized capture

Install the host dependency and capture a loading/unloading trial:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/capture.py --port /dev/cu.usbmodemXXXX \
  --seconds 60 --output trial.csv
```

Each CSV row contains one HX711 reading, force in newtons, and X/Y/Z/temperature
from all four MLX sensors. Collect repeated increasing and decreasing force
steps with 3–5 second holds. Use the steady middle portion of each hold for the
magnetic-field-to-force model; reserve at least one complete cycle for
validation rather than fitting.

For a labeled table in the serial monitor, enter `stream`. Enter `stop` to
stop it. The host capture tool uses the separate `csv` command automatically.

## Fit each MLX force model

Put the load cell mechanically in series with the patch being calibrated, so
the load cell and that patch experience the same force. Run one sensor at a
time and follow the prompts:

```bash
.venv/bin/python tools/calibrate_mlx.py --port /dev/cu.usbmodemXXXX --sensor 1
.venv/bin/python tools/calibrate_mlx.py --port /dev/cu.usbmodemXXXX --sensor 2
.venv/bin/python tools/calibrate_mlx.py --port /dev/cu.usbmodemXXXX --sensor 3
.venv/bin/python tools/calibrate_mlx.py --port /dev/cu.usbmodemXXXX --sensor 4
```

Each run records a no-force baseline and asks for at least five slow loading
and unloading cycles. It saves both the raw synchronized samples and a
quadratic vector-field/temperature model under `calibration_data/`, with RMSE,
MAE, and R² reported in newtons. Do not accept a model with a narrow force
range, poor R², or error too large for the intended application.
