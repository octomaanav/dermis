# Dermis MLX90393 Array Bring-Up Test Report

**Test date:** October 22, 2026  
**Controller:** ESP32-S3  
**Sensors:** Four MLX90393 breakout boards on a shared SPI bus  
**Firmware:** Dermis PlatformIO/Arduino bring-up firmware

## Executive summary

All four MLX90393 sensors were detected, configured, and sampled successfully. At the final selected SPI clock of 1 MHz, the array completed a 10-second acquisition with 3,577 frames, zero communication errors, and zero saturation events. The achieved full-array rate was 357.7 frames per second.

The array was not reliable above 1 MHz with the assembled wiring. A clock sweep passed at 1 MHz, showed substantial errors at 2 MHz, and failed continuously at 4 and 8 MHz. The recommended operating SPI clock is therefore 1 MHz.

## Hardware findings and corrections

Initial communication returned either `0x00` or `0xFF` for every transaction. The following hardware and platform issues were identified and corrected:

1. The controller was identified as an ESP32-S3 with 32 MB octal flash and 16 MB octal PSRAM. The generic quad-flash PlatformIO profile caused a boot-time flash initialization assertion. The project was updated for octal flash/PSRAM and native USB CDC.
2. SPI MISO was moved from GPIO19 to GPIO14 because GPIO19 is used by the ESP32-S3 native USB interface.
3. The MLX90393 breakout revision has reversed `MOSI` and `MISO` silkscreen labels. The breakout pin labeled `MISO` was connected to ESP32 MOSI (GPIO13), and the pin labeled `MOSI` was connected to ESP32 MISO (GPIO14).
4. The CS trace jumper on each breakout was cut to release CS from its default I2C-mode connection. Each sensor could then be selected independently by its ESP32 GPIO.

Final wiring:

| Signal | ESP32-S3 pin | Connection |
|---|---:|---|
| SCLK | GPIO18 | Shared by S1-S4 |
| MOSI | GPIO13 | Shared; breakout pad labeled `MISO` |
| MISO | GPIO14 | Shared; breakout pad labeled `MOSI` |
| S1 CS | GPIO21 | Dedicated |
| S2 CS | GPIO4 | Dedicated |
| S3 CS | GPIO16 | Dedicated |
| S4 CS | GPIO17 | Dedicated |
| Power | 3.3 V | Shared |
| Ground | GND | Shared |

## Detection and bus-isolation results

All four devices returned valid reset and register-read responses. Each device also retained a unique volatile scratch-register tag while the other devices remained deselected.

| Sensor | CS pin | Detection | Scratch tag | Isolation |
|---|---:|---|---:|---|
| S1 | GPIO21 | Found | `0xA501` | Pass |
| S2 | GPIO4 | Found | `0xA502` | Pass |
| S3 | GPIO16 | Found | `0xA503` | Pass |
| S4 | GPIO17 | Found | `0xA504` | Pass |

The shared MISO line followed both the ESP32's weak pull-up and pull-down while all CS lines were inactive. This confirms that the sensors release MISO correctly and that no device was continuously driving the shared bus.

## Sensor configuration

All sensors returned the same expected configuration:

| Parameter | Setting |
|---|---|
| Gain | 4x (`GAIN_SEL=1`) |
| X/Y/Z resolution | 18-bit selection (`RES=2`) |
| Oversampling | `OSR=0` |
| Temperature oversampling | `OSR2=0` |
| Digital filter | `DIG_FILT=2` |
| X/Y scale | 2.403 uT/LSB |
| Z scale | 3.872 uT/LSB |
| X/Y digital range | approximately +/-78.74 mT |
| Z digital range | approximately +/-126.87 mT |

Configuration register readback passed on all four sensors.

## Temperature readings

| Sensor | Raw value | Calculated temperature |
|---|---:|---:|
| S1 | 46,109 | 22.01 °C |
| S2 | 46,107 | 21.97 °C |
| S3 | 46,056 | 20.84 °C |
| S4 | 46,027 | 20.20 °C |

All temperature commands returned valid status bytes. These temperatures use the datasheet's typical conversion constants and are intended as diagnostics rather than calibrated temperature measurements.

## Final 10-second field and noise test

The final run was performed at 1 MHz after the clock sweep restored the runtime clock. It collected 3,577 complete four-sensor frames in 10 seconds.

| Sensor/axis | Mean (uT) | Standard deviation (uT) | Minimum (uT) | Maximum (uT) |
|---|---:|---:|---:|---:|
| S1 X | -21.794 | 6.629 | -43.254 | 0.000 |
| S1 Y | -26.712 | 5.015 | -43.254 | -9.612 |
| S1 Z | -57.514 | 11.661 | -92.928 | -27.104 |
| S2 X | -48.215 | 7.288 | -69.687 | -28.836 |
| S2 Y | 13.232 | 5.038 | -2.403 | 33.642 |
| S2 Z | -59.921 | 7.683 | -89.056 | -34.848 |
| S3 X | -11.463 | 7.609 | -33.642 | 9.612 |
| S3 Y | -45.487 | 4.422 | -60.075 | -31.239 |
| S3 Z | -50.066 | 7.660 | -81.312 | -27.104 |
| S4 X | -29.198 | 6.346 | -50.463 | -9.612 |
| S4 Y | -19.194 | 4.775 | -33.642 | -4.806 |
| S4 Z | -60.456 | 7.323 | -85.184 | -34.848 |

Final acquisition summary:

| Metric | Result |
|---|---:|
| Duration | 10 seconds |
| Complete frames | 3,577 |
| Full-array rate | 357.7 frames/s |
| Communication/status errors | 0 |
| Saturation events | 0 |
| Overall result | Pass |

The means represent the ambient magnetic field plus offsets and nearby magnetic material present during the bench test. They are not calibration constants. The standard deviations include sensor noise, bench vibration, electromagnetic interference, and any movement of nearby magnetic material.

## SPI clock-rate sweep

| SPI clock | Full-array rate | Frames | Errors | Result |
|---:|---:|---:|---:|---|
| 1 MHz | 364.6 fps | 730 | 0 | Pass |
| 2 MHz | 396.2 fps | 793 | 770 | Fail |
| 4 MHz | 414.1 fps | 829 | 829 | Fail |
| 8 MHz | 423.7 fps | 848 | 848 | Fail |

Increasing the SPI clock provided only a modest frame-rate improvement because sensor conversion time dominates the acquisition period. The assembled bus had insufficient signal integrity or timing margin above 1 MHz. Operation above 1 MHz is not recommended without improving wiring length, routing, grounding, and signal integrity and then repeating the sweep.

## Observed transient

During one noise run immediately following temperature measurements, S1 returned fixed negative-rail values on all axes:

- X/Y: -49,098.098 uT
- Z: -79,112.703 uT

The condition cleared after subsequent test sequencing, and the final 1 MHz run completed with no errors or saturation. This appears to be a transient conversion-state or command-sequencing condition rather than persistent sensor damage. It should nevertheless be investigated before relying on unattended operation. Recommended firmware hardening is to discard the first full XYZ conversion after a temperature-only command, reset/reconfigure a sensor after a rail-valued frame, and log per-sensor status when a rail value occurs.

## Conclusions and recommendations

The four-sensor SPI topology and individual chip-select wiring are functional. All sensors pass detection, register configuration, temperature conversion, bus isolation, and sustained full-array acquisition at 1 MHz.

For continued development:

1. Keep the production SPI clock at 1 MHz.
2. Preserve GPIO14 as MISO so native USB remains available.
3. Keep the corrected breakout-label mapping documented during assembly.
4. Add automatic recovery and first-sample rejection after temperature-only operations.
5. Perform stationary baseline captures away from magnets and current-carrying conductors before calibration.
6. Re-run the noise and rate tests after any wiring, cable-length, PCB, or power-supply change.
7. Do not treat the reported field means as calibrated measurements until hard-iron, soft-iron, axis-alignment, and sensor-to-sensor calibration have been completed.

