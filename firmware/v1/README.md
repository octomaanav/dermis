# Dermis Wire Schema v1.0 — kit

The data contract between the Dermis hub (ESP32-S3 with 4× MLX90393 + 4× BMI160) and host software. Read `SPEC.md` first.

```
SPEC.md                      the specification
firmware/dermis_wire.h       device side: structs, flags, CRC-32, COBS, packet builder, command receiver
firmware/test_vectors.c      prints the header's bytes for the test vectors
host/python/dermis_wire.py   reference decoder/encoder, CSV export, fake hub (standard library only)
host/python/test_dermis_wire.py
schemas/                     JSON Schema 2020-12: device_info, set_config, stats, decoded_frame
examples/                    one valid example per schema
vectors/                     test_vectors.json, frame.bin, capture_2s.bin, capture_2s.csv
```

## Software team: start without hardware

```sh
cd host/python
python3 dermis_wire.py serve-pty                  # prints e.g. /dev/ttys012; open it like a serial port
python3 dermis_wire.py dump --file ../../vectors/capture_2s.bin --csv out.csv
python3 -m unittest -v test_dermis_wire.py        # 23 tests (pip install jsonschema for the schema checks)
```

The fake hub answers GET_INFO, START, STOP, SYNC_PING, SET_CONFIG, RESET_SENSOR and SELFTEST like the firmware will, and streams about 350 frames/s of plausible data: presses, shear, a slip burst, hand motion, noise. If your decoder reproduces `vectors/test_vectors.json` and `vectors/capture_2s.csv`, it will read the real hub.

## Firmware: check the byte layout

```sh
cd firmware
cc -std=c11 -Wall -Wextra -O2 -o dw_test test_vectors.c && ./dw_test > c_output.txt
python3 ../host/python/dermis_wire.py check-vectors c_output.txt     # expect "ok" for all 7
```

On the ESP32-S3, include `dermis_wire.h` from the PlatformIO project (it compiles as C or C++) and follow the firmware checklist in SPEC.md §11.

## Changing the schema

Edit SPEC.md, then the header and the Python reference together. Regenerate the vectors:

```sh
python3 host/python/dermis_wire.py vectors > vectors/test_vectors.json
```

Then rerun both checks. Minor versions only add things (SPEC.md §10).
