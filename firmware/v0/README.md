# Dermis v0 firmware

The active `src/main.cpp` is the ESP32-S3 firmware for four MLX90393
magnetometers and one BMI160. It creates the WPA2 Wi-Fi hotspot
`Dermis-Glove` (password `dermis-demo`) and serves JSON frames over WebSocket
at `ws://192.168.4.1:81/ws`, at 10 Hz. Join that hotspot from the Mac, then run
`tools/wifi_receive.py` to record the decoded frames and optionally forward
them to the local simulator. The ESP is the server, so it never needs the
Mac's IP address. Change the demo password in `src/main.cpp` before a public
deployment. The older interactive MLX/HX711
bring-up harness is retained under `mlx_firmware/`; the command-menu sections
below describe that archived harness.

This is a PlatformIO/Arduino bench harness for four MLX90393 magnetometers on one ESP32-S3 SPI bus. It provides probe, isolation, configuration, noise, streaming, rate-sweep, and temperature commands plus Python logging, plotting, and analysis tools.

## Wiring

Power every breakout from 3.3 V and connect all grounds. Leave `TRIG`, `INT`, `SCL`, and `SDA` unconnected. On an SPI-capable breakout, use its SPI `SCLK`, `MISO`, and `MOSI` labels; do not connect the I2C-only STEMMA/Qwiic signals in parallel with these wires.

| Signal | ESP32 pin | Notes |
|---|---:|---|
| SCLK | GPIO18 | shared by S1-S4 |
| MISO | GPIO13 | shared by all SPI sensors |
| MOSI | GPIO14 | shared by all SPI sensors |
| CS S1 | GPIO21 | dedicated |
| CS S2 | GPIO4 | dedicated |
| CS S3 | GPIO16 | dedicated |
| CS S4 | GPIO17 | dedicated |
| 3V3 | 3.3 V | never 5 V |
| GND | GND | common ground |

All pin assignments live in [`include/pins.h`](include/pins.h). The firmware drives all chip-select pins output-high **before** calling `SPI.begin()` so no device sees a floating protocol-select input.

> **ESP32-S3 USB note:** GPIO19/20 are reserved for native USB, leaving the native USB Serial/JTAG connector available for flashing and monitoring.

### HX711 load-cell wiring

| HX711 pin | ESP32-S3 pin |
|---|---:|
| DT / DOUT | GPIO1 |
| SCK | GPIO2 |
| VCC | 3.3 V only |
| GND | GND |

For the specified load cell, connect `E+ / E- / A- / A+` to `red / black /
white / green`. If applied force produces the opposite delta sign from the
desired convention, swap `A+` and `A-` or invert the sign during calibration.

Send `l` at the serial menu to enter load-cell mode. Remove all force during
the two-second tare. The firmware idles the MLX sensors, holds every CS high,
and stops SPI while the HX711 streams `t_us,raw,delta_counts`. Press any key to
exit; SPI is restarted and all four MLX sensors are reconfigured automatically.

## Sensor configuration and limits

The harness configures every axis for `GAIN_SEL=1` (called `MLX90393_GAIN_4X` by Adafruit), `RES=2`/18-bit selection, `OSR=0`, and `DIG_FILT=2`, with default `HALLCONF=0xC` and temperature oversampling `OSR2=0`.

| Axis | Scale | Digital range |
|---|---:|---:|
| X/Y | 2.403 µT/LSB | about ±78.74 mT |
| Z | 3.872 µT/LSB | about ±126.87 mT |

This gives digital headroom beyond ±60 mT. There is a separate physical limit: the MLX90393 datasheet specifies magnetic saturation onset around 50 mT for standard `ABA-0xx` parts (and 25 mT for `ABA-1xx`). No gain setting can remove analog saturation. The firmware therefore sets each sensor's saturation flag when any axis reaches 48 mT or a digital rail. Check the exact ordering code printed on the installed ICs before relying on measurements near 50 mT.

`OSR=0`, `DIG_FILT=2`, `HALLCONF=0xC` is the documented 1.84 ms typical TXYZ configuration with a 493 Hz sensor-side maximum. The firmware allows 2.15 ms before readout and triggers S1-S4 back-to-back, so conversions overlap. At 1 MHz, an expected practical acquisition ceiling is roughly 350-400 frames/s before serial formatting/USB behavior; the sensor conversion time, then 921600-baud CSV bandwidth, should dominate rather than four serialized conversion waits. Run `r` on the actual wiring for measured values.

## Build, flash, and monitor

The PlatformIO project uses `board = esp32-s3-devkitc-1`, the Arduino framework, SPI mode 3, a 1 MHz default runtime clock, and 921600-baud serial. The `f` menu command cycles the runtime clock through 1, 2, 4, 8, and 10 MHz; `r` tests 1, 2, 4, and 8 MHz and restores 1 MHz.

With the PlatformIO extension, use its Build and Upload buttons. From a shell:

```bash
./scripts/build.sh
./scripts/flash.sh /dev/cu.usbserial-0001
./scripts/monitor.sh /dev/cu.usbserial-0001
```

Equivalent direct commands are:

```bash
pio run
pio run --target upload --upload-port /dev/cu.usbserial-0001
pio device monitor --port /dev/cu.usbserial-0001 --baud 921600
```

List candidate ports with `./scripts/pio.sh device list`. The wrappers find `pio` on `PATH` or the normal `~/.platformio/penv/bin/pio` installation.

Run the clean firmware build and host parser tests together:

```bash
./scripts/test.sh
```

## Firmware menu

Open the monitor at 921600 baud and send one character:

| Command | Action |
|---|---|
| `p` | exit/reset/probe all devices, decode raw status bytes, write unique volatile scratch tags, and check MISO release |
| `c` | dump gain, resolution, OSR, OSR2, filter, and µT/LSB scale |
| `n` | collect 10 seconds and print per-axis mean/stddev/min/max and achieved rate |
| `s` | stream CSV until any key; `#RATE` metadata is emitted once per second |
| `r` | acquire for two seconds each at 1/2/4/8 MHz and report rate/errors |
| `t` | read raw temperature and the datasheet typical °C conversion |
| `l` | pause MLX acquisition, tare and stream HX711 raw/delta counts until any key |
| `f` | select the next runtime SPI frequency, up to the 10 MHz chip limit |
| `?` | print help |

Status decoding follows the datasheet byte layout: burst, wake-on-change, single-measurement, error, single-error-detected, reset, and data-length bits. Stream `flags` is a hexadecimal bit field:

| Bits | Meaning |
|---:|---|
| 0-3 | status/transaction error for S1-S4 |
| 4-7 | saturation or analog-saturation risk for S1-S4 |
| 8-11 | S1-S4 not found |
| 12-15 | corrected single EEPROM error (`SED`) for S1-S4 |

The isolation test writes only volatile free register `0x0A`; it never issues the EEPROM store command.

## Host tools

Create an isolated environment and install the pinned packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tools/requirements.txt
```

Log until Ctrl-C:

```bash
python tools/log.py --port /dev/cu.usbserial-0001 --out run.csv
```

The logger writes a normalized header, displays firmware `#RATE` lines, and estimates dropped frames from timestamp gaps when it exits.

Plot the press test live:

```bash
python tools/plot_live.py --port /dev/cu.usbserial-0001
```

The upper panel shows a clearly colored magnitude trace for every patch. The lower panel shows X/Y/Z for the selected sensor; press keys `1`-`4` to change it. The title turns red when a saturation flag appears.

Analyze a completed log:

```bash
python tools/analyze.py run.csv
```

This prints per-axis statistics, interval/jitter statistics, estimated drops, saturation counts, and baseline-versus-99.5th-percentile peak magnitude for each press channel. It also writes `run.jitter.png`.

## Step-by-step bring-up

1. Power down. Connect shared SCLK/MISO/MOSI and **S1 only**, including its dedicated CS. Power up, run `p`, then `n`. `p` will report S2-S4 not found at this stage; S1 and the MISO release check should pass.
2. Power down, add S2, power up, and rerun `p`. Repeat for S3 and S4. With all installed, the final probe summary must be `PASS` and each volatile tag must be unique.
3. Run `r`. Start testing at 1 MHz; only keep a higher runtime clock if it has zero errors on the assembled wiring.
4. Run `plot_live.py`. Press S1, S2, S3, and S4 in turn. Confirm only the matching magnitude trace responds and the title never reports saturation.
5. Leave every patch untouched for 60 seconds while running `log.py`, then stop it and run `analyze.py run.csv` for the drift/noise baseline.

## Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| Sensor `NOT FOUND` | Check MISO/MOSI orientation, power/ground, loose CS, and that CS is driven high at boot. Start with one sensor at 1 MHz. |
| All sensors read identical values | CS lines may be shorted together. The probe's unique scratch-tag isolation test should fail and identify the issue. |
| Status `error=1` | A command was rejected, data was read before ready, or an uncorrectable memory error occurred. Return to 1 MHz, check wiring, power-cycle, and rerun `p`. |
| Status `sed=1` | The device corrected a single EEPROM bit error. Record the device and replace it if the condition repeats. |
| Values pinned / saturation flags | Verify the configured gain/resolution with `c`; check the IC ordering code. A standard part is physically near saturation at 50 mT even though the digital range is wider. Increase magnet spacing if necessary. |
| Random dropouts | Shorten SPI wires, improve ground/decoupling, route away from noise, and lower the SPI clock with `f` or power-cycle to restore the 1 MHz default. |
| Corrupt serial / missing lines | Confirm exactly 921600 baud and use a USB cable/adapter that sustains the stream. `r` measures acquisition without CSV bandwidth. |

## Datasheet uncertainty

The field scale, status layout, commands, 10 MHz SPI maximum, conversion timing, and temperature conversion come from Melexis MLX90393 datasheet revision 012. Temperature in °C uses the *typical* 46244 LSB at 25 °C and 45.2 LSB/°C values, so it is useful for diagnostics, not precision thermometry. Without hardware attached here, actual frame rate, signal/noise, isolation behavior, and sensor ordering codes remain to be measured on the bench.
