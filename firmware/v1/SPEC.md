# Dermis Wire Schema v1.0

Status: draft for review · October 22, 2026
Owner: Sujal (hub firmware) · Readers: everyone building Dermis host software

This is the contract between the Dermis hub (ESP32-S3) and host software: the bytes the hub sends over USB, and the commands it accepts. Version 1.0 carries four pods. Each pod has one **MLX90393** magnetometer (touch) and one **BMI160** IMU (motion):

| Pod | Location key | MLX90393 | BMI160 |
|---|---|---|---|
| Index fingertip | `index_tip` | mag ch 0 (CS GPIO21) | imu ch 0 |
| Thumb fingertip | `thumb_tip` | mag ch 1 (CS GPIO4) | imu ch 1 |
| Middle fingertip | `middle_tip` | mag ch 2 (CS GPIO16) | imu ch 2 |
| Thenar eminence, over abductor pollicis brevis | `thenar_apb` | mag ch 3 (CS GPIO17) | imu ch 3 |

The channel-to-location mapping above is the default; the real one is whatever DEVICE_INFO says. Nothing in the schema is fixed at 4 + 4: counts come from DEVICE_INFO and from every frame, so a fifth IMU on the back of the hand, a robot-hand hub or a multi-taxel tile needs no format change.

Machine-readable versions of everything here: `firmware/dermis_wire.h` (device side), `host/python/dermis_wire.py` (reference decoder, encoder, fake hub), `schemas/*.json` (JSON Schema 2020-12), `vectors/test_vectors.json` (exact bytes both sides must produce).

---

## 1. At a glance

| | |
|---|---|
| Transport | USB CDC byte stream from the ESP32-S3's native USB. The format also works unchanged over UART, TCP or BLE. |
| Framing | COBS; every packet ends with one 0x00 byte; CRC-32 per packet |
| Byte order | Little-endian, packed (no padding), two's-complement signed integers |
| Sensor values | Raw integer counts. Scale factors to SI units are in DEVICE_INFO. |
| Device clock | `esp_timer_get_time()`: microseconds since boot, u64 |
| Data packet | One FRAME per acquisition tick: all MLX90393 records + all BMI160 records |
| Frame size | 180 bytes decoded, 182 bytes on the wire (4 + 4 sensors) |
| Frame rate | About 350 frames/s with the current MLX90393 settings (conversion-bound) |
| Bandwidth | About 64 KB/s |
| Max packet | 8,192 bytes decoded (payload ≤ 8,172) |

Why raw counts: they're lossless and compact, the firmware stays simple, and scaling lives in one place (DEVICE_INFO), so a range change can't silently corrupt data.

---

## 2. Packets

Every packet, in both directions, has this layout before encoding:

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | u8 | `msg_type` | §3 |
| 1 | u8 | `version` | `0x10` = v1.0 (major in the high nibble, minor in the low nibble) |
| 2 | u16 | `payload_len` | Payload bytes |
| 4 | u32 | `seq` | Per-direction packet counter: +1 for every packet sent, wraps at 2³² |
| 8 | u64 | `t_us` | Device → host: device clock (µs) at a moment defined per message. Host → device: 0. |
| 16 | — | payload | `payload_len` bytes |
| 16 + N | u32 | `crc32` | CRC over bytes 0 … 15 + N |

**CRC:** CRC-32/ISO-HDLC, the same function as `zlib.crc32` and Python's `binascii.crc32` (reflected polynomial 0xEDB88320, initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF). Check value: `"123456789"` → `0xCBF43926`.

**Framing:** COBS-encode the whole packet (header + payload + CRC), then append a single 0x00. Encoded data never contains 0x00, so a receiver finds packet boundaries by splitting on 0x00 and recovers from any error at the next delimiter. When a 254-byte COBS block ends exactly at the end of the input, no extra 0x01 code byte is added. (This only matters for byte-exact comparisons; both reference encoders do the same, and every decoder accepts both forms.)

**A receiver discards a packet, counts it, and carries on** when:
- COBS decoding fails
- it's shorter than 20 bytes
- 16 + `payload_len` + 4 ≠ decoded length
- the CRC doesn't match
- the major version isn't 1

Empty packets (two 0x00 in a row) are skipped silently. A host may send a lone 0x00 at any time to flush the device's receive buffer.

---

## 3. Message types

**Device → host**

| Type | Name | Payload | Sent |
|---|---|---|---|
| 0x01 | DEVICE_INFO | JSON (§5) | after GET_INFO and after every applied SET_CONFIG |
| 0x02 | STATS | JSON (§6.4) | once per second while streaming |
| 0x03 | EVENT | binary + text (§6.3) | boot, stream start/stop, faults, marker button |
| 0x04 | SYNC_PONG | binary (§7) | in reply to SYNC_PING |
| 0x05 | ACK | binary + optional text (§6.2) | in reply to every command except SYNC_PING |
| 0x06 | LOG | UTF-8 text | debug text; replaces `Serial.print` once in binary mode |
| 0x10 | FRAME | binary (§4) | every acquisition tick while streaming |
| 0x11 | IMU_BURST | reserved (v1.1) | high-rate IMU FIFO data |
| 0x12 | FORCE_REF | reserved | load-cell ground truth |
| 0x13 | ENV | reserved | ambient temperature and humidity |

**Host → device:** 0x80–0x88, §6.1.

Receivers ignore (and count) message types they don't know.

---

## 4. FRAME (0x10)

The header's `t_us` is the **trigger time**: when the firmware sends the Start-Measurement command to the first MLX90393. All MLX90393s are triggered back to back (within about 100 µs) and convert at the same time; the IMUs are read while they convert.

Payload = frame header (16 bytes), then `n_mag` mag records, then `n_imu` IMU records. With 4 + 4 sensors that's 16 + 48 + 96 = 160 bytes.

### 4.1 Frame header (16 bytes)

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | u32 | `frame_counter` | +1 every acquisition tick since boot, **even when the frame isn't sent**. A gap means frames were lost. |
| 4 | u16 | `config_id` | Configuration this frame was made with; matches DEVICE_INFO |
| 6 | u8 | `frame_flags` | §4.4 |
| 7 | u8 | reserved | 0 |
| 8 | u8 | `n_mag` | Number of mag records (4) |
| 9 | u8 | `mag_rec_size` | Bytes per mag record (12 in v1.0) |
| 10 | u8 | `n_imu` | Number of IMU records (4) |
| 11 | u8 | `imu_rec_size` | Bytes per IMU record (24 in v1.0) |
| 12 | u16 | `acq_us` | µs from the trigger until the last sensor read finished |
| 14 | u16 | reserved | 0 |

### 4.2 Mag record — MLX90393 (12 bytes)

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | u8 | `ch` | Mag channel, 0-based |
| 1 | u8 | `mlx_status` | Status byte from the Read-Measurement reply, unmodified |
| 2 | u8 | `flags` | §4.4 |
| 3 | u8 | reserved | 0 |
| 4 | i16 | `x` | Signed counts, chip X axis |
| 6 | i16 | `y` | Chip Y |
| 8 | i16 | `z` | Chip Z |
| 10 | u16 | `t_raw` | Raw temperature word |

**Counts are signed, with 0 = 0 µT.** The MLX90393 returns RES 2 and RES 3 data unsigned with an offset; the firmware removes it, so host code never needs to know RES:
- RES 0/1: reinterpret the word as int16
- RES 2: word − 32768
- RES 3: word − 16384 (confirm against the datasheet's output-format table)

`dw_mlx_counts()` in the header does this for TCMP_EN = 0.

**`mlx_status` bits (datasheet):**
- 7 BURST_MODE · 6 WOC_MODE · 5 SM_MODE · 4 ERROR · 3 SED · 2 RS
- 1–0 D: the chip sends 2·D + 2 data bytes. Reading z, y, x and T needs 8 bytes, so a good reply has D = 3. Anything else is `BAD_LENGTH`. This catches the junk frame seen in bring-up (−20,432 LSB on all three axes after a temperature-only measurement).

### 4.3 IMU record — BMI160 (24 bytes)

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | u8 | `ch` | IMU channel, 0-based |
| 1 | u8 | `flags` | §4.4 |
| 2 | i16 | `t_rel_us` | When this IMU's data read started, relative to the frame trigger |
| 4 | i16 ×3 | `acc_x`, `acc_y`, `acc_z` | Accelerometer counts, chip axes |
| 10 | i16 ×3 | `gyr_x`, `gyr_y`, `gyr_z` | Gyroscope counts, chip axes |
| 16 | i16 | `temp_raw` | TEMPERATURE register; 0x8000 = invalid |
| 18 | u16 | reserved | 0 |
| 20 | u32 | `sensortime` | SENSORTIME from the same burst (low 24 bits, 39.0625 µs per tick); 0 if not read |

Firmware tip: one 15-byte burst read starting at register 0x0C returns gyro (0x0C–0x11), accel (0x12–0x17) and SENSORTIME (0x18–0x1A) from the same sample. The registers hold gyro first; the record stores accel first.

### 4.4 Flags

**`frame_flags`**

| Bit | Name | Meaning |
|---|---|---|
| 0x01 | CONFIG_CHANGED | First frame after a configuration change |
| 0x02 | OVERRUN | This tick took longer than the target period |
| 0x04 | TX_DROP | At least one earlier frame was produced but not sent (USB buffer full). `frame_counter` shows how many. |
| 0x08 | MARKER | The hub's marker button was pressed since the previous frame |

**Mag record `flags`**

| Bit | Name | Meaning | Values usable? |
|---|---|---|---|
| 0x01 | STALE | No new conversion this tick; values repeated | Yes, but old |
| 0x02 | BAD_LENGTH | Status D ≠ requested length | **No** |
| 0x04 | CHIP_ERROR | Status ERROR bit set | **No** |
| 0x08 | RAIL | An axis reached `rail_lsb` (saturated) | Bound only |
| 0x10 | COMM_FAIL | SPI transaction failed; fields are 0 | **No** |
| 0x20 | TEMP_STALE | `t_raw` repeated from an earlier tick | Yes |
| 0x40 | RECOVERED | Chip was reset or reconfigured since the previous frame; expect a step | Yes |

**IMU record `flags`**

| Bit | Name | Meaning | Values usable? |
|---|---|---|---|
| 0x01 | STALE | No new sample since the previous frame | Yes, but old |
| 0x02 | COMM_FAIL | Bus transaction failed; fields are 0 | **No** |
| 0x04 | ACC_RAIL | An accel axis at −32768 or 32767 | Bound only |
| 0x08 | GYR_RAIL | A gyro axis at −32768 or 32767 | Bound only |
| 0x10 | TEMP_STALE | `temp_raw` not refreshed this tick | Yes |
| 0x20 | RECOVERED | Sensor re-initialized since the previous frame | Yes |

A record is **invalid** when `flags & 0x16` (mag) or `flags & 0x02` (IMU) is non-zero. Keep invalid records in raw recordings; show them as missing (null/NaN) in SI data. Never interpolate over them silently.

### 4.5 Units and times

With the matching DEVICE_INFO entry (`m` = `mag[ch]`, `i` = `imu[ch]`):

| Quantity | Formula | Unit |
|---|---|---|
| Field, per axis | `count × m.scale_ut_per_lsb[axis]` | µT |
| MLX temperature | `25 + (t_raw − m.tref) / m.temp_lsb_per_c` | °C |
| Acceleration | `count × i.acc_scale_mps2_per_lsb` (= 9.80665 / LSB-per-g) | m/s² |
| Angular rate | `count × i.gyr_scale_rads_per_lsb` (= (π/180) / LSB-per-°/s) | rad/s |
| IMU temperature | `i.temp_offset_c + temp_raw × i.temp_c_per_lsb` (= 23 + raw/512) | °C |
| Mag sample time | `t_us + m.conv_time_us / 2` (middle of the conversion window) | µs, device clock |
| IMU read time | `t_us + t_rel_us`; the data is the newest sample at that moment, at most 1/ODR old | µs, device clock |

BMI160 sensitivities: accel ±2/4/8/16 g → 16384/8192/4096/2048 LSB/g; gyro ±125/250/500/1000/2000 °/s → 262.4/131.2/65.6/32.8/16.4 LSB per °/s. The accelerometer measures specific force: it reads about +9.81 m/s² on the axis pointing up when still.

### 4.6 Reference frame

`vectors/test_vectors.json` → `frame` is a complete example with its decoded values: index, thumb and middle healthy; the thenar MLX record carries the bring-up junk with `BAD_LENGTH`; the middle IMU is `STALE`; the frame has `MARKER` set. Its 182 wire bytes:

```
041010a0022a0101044e61bc0101010103d204010203020807040c0418720601010203010234098aff3d1b1db4
010301042eff4008c11b1bb4020320020f0addffc21ae8b30301020730b030b030b001010102d4022306c4fffa
0f0c04f8ff030328020104c3b2a102010caf0146fb860b8c0a6aff5f0228031c020104d0b2a10602018a021406
d3ff051005020205fcff24020104dcb2a102030a65039001c4f084031e04eaff120330020104e8b2a105ebdfa677
00
```

---

## 5. DEVICE_INFO (0x01)

A compact UTF-8 JSON object (schema: `schemas/device_info.schema.json`; full example: `examples/device_info.json`, about 2.9 KB). It describes the hub, its pods and every channel, including all scale factors. Hosts must ignore keys they don't know.

| Key | Meaning |
|---|---|
| `proto` | `"1.0"` |
| `fw`, `board`, `build` | Firmware version, board name, optional build stamp |
| `device_id` | Stable per hub (from the eFuse MAC), e.g. `dermis-3c84279f2a10` |
| `role` | `glove`, `robot` or `bench` |
| `hand` | `left`, `right` or `none` |
| `config_id` | +1 on every applied SET_CONFIG; persists across reboots. Hosts can cache DEVICE_INFO by (`device_id`, `config_id`). |
| `rate_hz` | Target frame rate; 0 = as fast as the sensors allow |
| `clock` | `{"source": "esp_timer", "unit": "us"}` |
| `pods[]` | `id` (pod serial or ""), `location`, `mag_ch`, `imu_ch` (either may be null) |
| `mag[]` | Per MLX90393: `ch`, `part`, `bus`, `cs_gpio`, `gain_sel`, `hallconf`, `res_xyz`, `osr`, `osr2`, `dig_filt`, `tcmp_en`, `scale_ut_per_lsb[3]`, `rail_lsb[3]`, `tref`, `temp_lsb_per_c`, `conv_time_us`, `mount` |
| `imu[]` | Per BMI160: `ch`, `part`, `bus`, `addr` or `cs_gpio`, `acc_range_g`, `acc_odr_hz`, `gyr_range_dps`, `gyr_odr_hz`, `acc_scale_mps2_per_lsb`, `gyr_scale_rads_per_lsb`, `temp_offset_c`, `temp_c_per_lsb`, `sensortime_us_per_tick`, `mount` |

**Location keys:** `thumb_tip`, `index_tip`, `middle_tip`, `ring_tip`, `little_tip`, `thenar_apb`, `hypothenar`, `palm_center`, `hand_dorsum`, `wrist`, `bench`, `unassigned`.

**Current values** (from the Oct 22 bring-up), shared by all four MLX90393s:
- gain_sel 1, HALLCONF 0xC, RES 2, OSR 0, DIG_FILT 2
- scale 2.403 µT/LSB (X, Y) and 3.872 µT/LSB (Z)
- `conv_time_us` = 1344, from the datasheet formula: (2 + 2^DIG_FILT) × 2^OSR × 64 µs per axis × 3, plus 2^OSR2 × 192 µs for temperature
- `rail_lsb` 22000 per axis. This is my reading of the datasheet's RES 2 output range; confirm it.

Proposed IMU defaults: ±8 g and ±2000 °/s at 800 Hz ODR. That gives a fresh sample every frame at 350 Hz and leaves headroom for knuckle taps. The IMU `bus`/`addr` values in the example are placeholders until the IMU wiring is final.

---

## 6. Commands, replies, events, stats

### 6.1 Commands (host → device)

| Type | Name | Payload | Device replies |
|---|---|---|---|
| 0x80 | GET_INFO | — | ACK, then DEVICE_INFO |
| 0x81 | START | u16 `rate_hz` (0 = max), u16 reserved | ACK, EVENT STREAM_STARTED, then FRAMEs + STATS. Idempotent; a new rate applies immediately. |
| 0x82 | STOP | — | ACK, EVENT STREAM_STOPPED |
| 0x83 | SET_CONFIG | JSON (§6.5) | ACK. On success also DEVICE_INFO (new `config_id`) and EVENT CONFIG_CHANGED. |
| 0x84 | SYNC_PING | u64 `host_time` (opaque to the device) | SYNC_PONG only (no ACK) |
| 0x85 | RESET_SENSOR | u8 `kind` (0 mag, 1 IMU, 2 both), u8 `ch` (0xFF = all) | ACK; next record(s) carry RECOVERED |
| 0x86 | SELFTEST | u8 `test` (0 probe, 1 CS isolation, 2 one-second noise) | ACK, then EVENT SELFTEST_RESULT with a JSON result. Only while stopped. |
| 0x87 | SET_MODE | u8 `mode` (0 text menu, 1 binary) | ACK, then switches |
| 0x88 | REBOOT | — | ACK, then restart (the host sees EVENT BOOT and `seq` back at 0) |

### 6.2 ACK (0x05)

Payload: u32 `cmd_seq` (the command's header `seq`), u8 `cmd_type`, u8 `status`, u16 reserved, then optional UTF-8 text explaining a failure.

| Status | Name | Meaning |
|---|---|---|
| 0 | OK | Done |
| 1 | UNKNOWN_TYPE | Unknown command |
| 2 | BAD_PAYLOAD | Wrong size or invalid JSON/values; nothing changed |
| 3 | BAD_STATE | Not allowed now (e.g. SET_CONFIG while streaming) |
| 4 | FAILED | Valid request, but the hardware didn't cooperate |
| 5 | VERSION | Unsupported version |

### 6.3 EVENT (0x03)

Payload: u16 `code`, u8 `severity` (0 info, 1 warning, 2 error), u8 `ch` (0xFF = none), then UTF-8 text (may be empty). The header's `t_us` is when it happened.

| Code | Name | Text |
|---|---|---|
| 0x0001 | BOOT | Reset reason |
| 0x0002 | STREAM_STARTED | e.g. `rate_hz=350` |
| 0x0003 | STREAM_STOPPED | — |
| 0x0004 | CONFIG_CHANGED | New `config_id` |
| 0x0005 | MARKER | Marker button (the frame flag carries the same moment) |
| 0x0010 / 0x0011 | MAG_RESET / MAG_FAULT | Detail |
| 0x0020 / 0x0021 | IMU_RESET / IMU_FAULT | Detail |
| 0x0030 | TX_OVERFLOW | Frames dropped since the last report |
| 0x0031 | RATE_NOT_MET | Achieved rate |
| 0x0040 | SELFTEST_RESULT | JSON |

### 6.4 STATS (0x02)

JSON, once per second while streaming (schema: `schemas/stats.schema.json`).
- Required: `uptime_ms`, `frames_produced`, `frames_sent`, `tx_dropped`, `rate_hz`
- Optional: `acq_us_max`, `period_us_max`, `mag_err[]`, `imu_err[]`, `rx_bad_packets`, `heap_free`, `usb_tx_free_min`, `chip_temp_c`

### 6.5 SET_CONFIG

A JSON object with only the writable keys (schema: `schemas/set_config.schema.json`):
- `rate_hz`, `role`, `hand`
- `pods`: replaces the whole list
- `mag[]` and `imu[]`: matched by `ch` and merged. Writable: mag `gain_sel`, `hallconf`, `res_xyz`, `osr`, `osr2`, `dig_filt`, `mount`; IMU `acc_range_g`, `acc_odr_hz`, `gyr_range_dps`, `gyr_odr_hz`, `mount`.

Rules for the device:
- Accept it only while stopped (otherwise BAD_STATE).
- Apply all of it or none of it.
- Recompute scale factors.
- Persist the config in NVS.
- Increment `config_id`.
- Reply ACK + DEVICE_INFO.

The software's "identify pods" and "axis mapping" steps write `pods` and `mount` this way.

### 6.6 Session

```
host: open port, send 0x00 + GET_INFO              device: ACK, DEVICE_INFO
host: SYNC_PING × 5 (then 1 per second)            device: SYNC_PONG each
host: START(0)                                     device: ACK, EVENT STREAM_STARTED
                                                   device: FRAME ×~350/s, STATS 1/s, EVENTs
host: STOP                                         device: ACK, EVENT STREAM_STOPPED
```

If no DEVICE_INFO arrives within 500 ms, resend GET_INFO (up to 3 times). The device stops streaming when the USB host disconnects; the host sends START again after reconnecting.

### 6.7 Text menu and binary mode

The firmware may keep the bring-up text menu. It boots in text mode and switches to binary mode on the first valid binary packet (or SET_MODE 1); from then on all text goes out as LOG packets.

To make that safe, the text menu must be **line-based** (command + Enter), not single-keystroke. The first packet a host sends, `00 03 80 10 01 01 01 01 01 01 01 01 01 01 01 01 01 05 ee 39 24 f4 00` (0x00 + GET_INFO with seq 0), contains no 0x0A or 0x0D, so it can't trigger a text command.

---

## 7. Time and clock sync

All device times are `esp_timer` microseconds since boot. To put frames on the host clock (for video, pose and robot data), the host runs NTP-style pings:

1. Host sends SYNC_PING with `host_time = h1` (its own monotonic clock, any unit; the device echoes it).
2. Device records `t_rx_us` on receipt and replies at once with SYNC_PONG: payload `host_time` (echo) + `t_rx_us` (u64); header `t_us` = transmit time `t_tx_us`.
3. Host receives it at `h2`. Round trip = (h2 − h1) − (t_tx − t_rx). Offset ≈ ((t_rx − h1) + (t_tx − h2)) / 2, in common units.
4. Keep the lowest-round-trip samples from the last ~60 s and fit `host = a + b × device` (b absorbs crystal drift, typically within ±50 ppm). Map each frame's `t_us` through the fit.

USB adds about 1 ms of jitter per direction, so the minimum-round-trip filter matters. `esp_timer` doesn't jump while the device is running; a reboot shows up as EVENT BOOT and `seq` restarting at 0.

---

## 8. Axes and pod frames

Every record is in its chip's own axes, as printed on the breakout (X and Y in the board plane, Z out of the component side). Each `mag[]` and `imu[]` entry has a `mount` string that rotates chip axes into the **pod frame**:

| Pod axis | Fingertip pods | `thenar_apb` | `hand_dorsum` (optional 5th IMU) |
|---|---|---|---|
| **n** (normal) | Out of the finger pad, toward the contact | Out of the palm | Out of the back of the hand |
| **d** (distal) | Toward the fingertip along the distal phalanx | Along the first metacarpal toward the thumb | Toward the middle-finger knuckle |
| **l** (lateral) | n × d (right-handed: d × l = n) | n × d | n × d |

`mount` lists which signed chip axis points along d, l and n, in that order. `"+X+Y+Z"` means the chip X is distal, Y is lateral and Z is normal. `"-Y+X+Z"` is the same board turned 90°. A mount must be a rotation: each letter exactly once and no mirror image (`mount_matrix()` in the Python reference checks this).

Mount each pod's BMI160 with the **same orientation** as its MLX90393 if you can. Then one rotation serves both, and the IMU's orientation can be used directly to subtract Earth's field from the magnetometer.

Recommendation: wire the fifth BMI160 from the pack on the back of the hand as `hand_dorsum` (imu ch 4, `mag_ch` null). It gives the hand a reference frame, so finger orientation can be computed relative to the hand. The schema handles it with no changes.

---

## 9. Decoded form (for host software)

After decoding a FRAME with its DEVICE_INFO, host code should work with one shape (schema: `schemas/decoded_frame.schema.json`; example: `examples/decoded_frame.json`):
- frame-level: `device_id`, `frame`, `t_device_us`, `t_host_ns` (null until clock sync), `config_id`, `flags`
- `pods`, keyed by location; each has `mag` (`b_ut[3]`, `temp_c`, `t_us`, `status`, `flags`, `valid`) and `imu` (`acc_mps2[3]`, `gyr_rads[3]`, `temp_c`, `t_us`, `sensortime`, `flags`, `valid`)
- values stay in chip axes; invalid values are null

Use this for internal APIs, WebSocket streams and logs.

**Column names for CSV / Parquet / NumPy exports:**
- frame columns: `t_device_us, t_host_ns, frame, config_id, frame_flags`
- then, for each pod `<loc>`: `<loc>_bx_ut, <loc>_by_ut, <loc>_bz_ut, <loc>_mag_temp_c, <loc>_mag_flags, <loc>_ax_mps2, <loc>_ay_mps2, <loc>_az_mps2, <loc>_gx_rads, <loc>_gy_rads, <loc>_gz_rads, <loc>_imu_temp_c, <loc>_imu_t_us, <loc>_imu_flags`

That's 61 columns for 4 pods. Flags are raw integers; invalid values are empty. `vectors/capture_2s.csv` shows the exact output.

---

## 10. Versioning and compatibility

- `version` = major.minor. Receivers reject a different **major**. A newer **minor** must still decode.
- Minor versions only add things:
  - new message types (ignored by old receivers)
  - new JSON keys (ignored)
  - new fields at the **end** of a record (old receivers step by `mag_rec_size` / `imu_rec_size` and never by `sizeof`)
  - extension bytes after the last record (ignored)
- Reserved fields are sent as 0 and ignored on receipt.
- Anything that changes the meaning or position of an existing field is a new major version.
- `config_id` in every frame ties data to the DEVICE_INFO that explains it. A host that sees an unknown `config_id` sends GET_INFO and holds those frames until the new info arrives, rather than converting with stale scales.

---

## 11. Firmware checklist

- [ ] All multi-byte fields little-endian; structs packed; reserved fields 0 (`dermis_wire.h` handles this).
- [ ] `frame_counter` increments every tick even when a frame isn't sent; `seq` increments per packet actually sent.
- [ ] Never block on USB. If the TX buffer can't take a whole packet, drop the frame, count it and set TX_DROP on the next frame sent.
- [ ] Trigger all MLX90393s, read the IMUs while they convert, then read the MLX90393s; `t_us` = trigger time.
- [ ] Check every MLX reply: ERROR bit → CHIP_ERROR; D ≠ 3 → BAD_LENGTH; SPI failure → COMM_FAIL with zeroed fields.
- [ ] Convert MLX words to signed counts (`dw_mlx_counts`); set RAIL against `rail_lsb`.
- [ ] Set RECOVERED after any sensor reset or re-init; set STALE / TEMP_STALE when values are repeated.
- [ ] DEVICE_INFO scale factors always match the active configuration; bump and persist `config_id` on every change.
- [ ] Stop streaming when the USB host disconnects.
- [ ] STATS once per second while streaming.
- [ ] Run `test_vectors.c` on the host, or the same calls on the ESP32, and compare against the vectors before trusting the byte layout.

## 12. Host checklist

- [ ] Split the stream on 0x00; COBS-decode; check length, CRC and major version; count every rejection.
- [ ] Ignore unknown message types and JSON keys; step through records by the sizes in the frame header.
- [ ] Hold frames whose `config_id` doesn't match the current DEVICE_INFO; request GET_INFO.
- [ ] Detect lost frames from `frame_counter` gaps, separately from transport errors.
- [ ] Keep raw counts in recordings; derive SI values with the DEVICE_INFO saved alongside.
- [ ] Treat invalid records as missing; show RAIL as saturated; apply your own analog soft limits (the MLX90393 goes nonlinear above about 45–50 mT without setting any flag).
- [ ] Run clock sync continuously and store the fit with each recording.
- [ ] Don't assume 4 + 4 sensors or a fixed rate; use the counts and timestamps.

## 13. Test kit

| File | What it's for |
|---|---|
| `vectors/test_vectors.json` | Exact wire bytes + decoded values for 8 packets, CRC check value, COBS examples. Any implementation in any language should reproduce these. |
| `vectors/frame.bin` | The reference FRAME as raw wire bytes |
| `vectors/capture_2s.bin` | Two seconds of simulated hub output (≈700 frames, STATS, EVENTs, one dropped frame, one BAD_LENGTH record, one marker) |
| `vectors/capture_2s.csv` | The same capture decoded with the reference decoder; compare your output against it |
| `host/python/dermis_wire.py serve-pty` | A fake hub on a virtual serial port (macOS/Linux). Point the app at the printed `/dev/ttys…` path and develop without hardware. |
| `firmware/test_vectors.c` | Prints the C encoder's bytes; `dermis_wire.py check-vectors` compares them |
