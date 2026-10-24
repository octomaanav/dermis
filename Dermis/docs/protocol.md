# Dermis Wire Protocol v1

## Framing

Packets are COBS encoded and terminated by `0x00`. The decoded packet is little-endian:

```text
u8 type | u8 flags | u16 payload_length | u32 sequence | u64 device_time_us |
payload[payload_length] | u32 crc32
```

CRC-32 covers the header and payload, excluding the CRC field. The polynomial is IEEE 802.3 (`0xEDB88320`, reflected), initial and final XOR `0xFFFFFFFF`.

## Compatibility

`HELLO` contains `protocol_major` and `protocol_minor`. Hosts reject a different major version with a plain compatibility error. Unknown packet types and fields from a compatible minor version are logged and ignored.

## Device-to-host types

| Value | Type | Payload |
|---:|---|---|
| `0x01` | HELLO | UTF-8 JSON device description |
| `0x02` | MAG_FRAME | Repeated channel samples |
| `0x03` | IMU_FRAME | Reserved |
| `0x04` | FORCE_REF | Reserved |
| `0x05` | ENV | Reserved |
| `0x06` | EVENT | Structured event |
| `0x07` | LOG | UTF-8 diagnostic text |
| `0x08` | ACK | Command ID and optional data |
| `0x09` | NACK | Command ID, reason code, text |
| `0x0A` | SYNC_PONG | Host timestamp and device timestamp |

Each `MAG_FRAME` channel contains channel ID, raw MLX status, signed X/Y/Z counts as `i32`, raw temperature as `u16`, and flags. Flags include response-length mismatch, digital rail, timeout, and stale temperature. Raw values are always preserved.

## Host-to-device types

`GET_INFO`, `START(rate_hz)`, `STOP`, `SET_CONFIG(JSON)`, `RESET_SENSOR(channel)`, `SELFTEST(kind)`, `SET_POD_ID(channel, id)`, and `SYNC_PING(host_time)`.

## Legacy input

The current firmware remains supported. Command `s` emits CSV rows:

```text
t_us,s1x_uT,s1y_uT,s1z_uT,s1t_C,...,s4z_uT,s4t_C,flags
```

Lines beginning with `#` are metadata or rate reports. The adapter must tolerate partial reads, CRLF, malformed rows, and reconnects.

