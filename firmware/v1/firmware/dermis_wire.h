/*
 * dermis_wire.h — Dermis Wire Schema v1.0, device side.
 *
 * One header, no dependencies: message types, flags, packed structs,
 * CRC-32, COBS, a packet builder and a byte-at-a-time command receiver.
 * C11 or C++11 and later. Little-endian targets only (the ESP32-S3 is).
 *
 * Spec: SPEC.md. Test: cc -std=c11 -Wall -Wextra -o dw_test test_vectors.c && ./dw_test
 *
 * Typical use on the hub:
 *
 *   static uint8_t scratch[DW_MAX_PACKET], wire[DW_MAX_WIRE], payload[512];
 *   size_t plen = dw_build_frame_payload(payload, sizeof payload, frame_counter, config_id,
 *                                        frame_flags, acq_us, mag, 4, imu, 4);
 *   size_t n = dw_build_packet(DW_MSG_FRAME, tx_seq++, t_trigger_us, payload, plen, scratch, wire);
 *   if (usb_tx_free() >= n) usb_write(wire, n); else { tx_dropped++; next_frame_flags |= DW_FRAME_TX_DROP; }
 */
#ifndef DERMIS_WIRE_H
#define DERMIS_WIRE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- sizes */

#define DW_PROTO_MAJOR   1u
#define DW_PROTO_MINOR   0u
#define DW_PROTO_VERSION ((DW_PROTO_MAJOR << 4) | DW_PROTO_MINOR)   /* 0x10 */

#define DW_HEADER_SIZE   16u
#define DW_CRC_SIZE      4u
#define DW_MAX_PACKET    8192u                                       /* header + payload + CRC */
#define DW_MAX_PAYLOAD   (DW_MAX_PACKET - DW_HEADER_SIZE - DW_CRC_SIZE)
#define DW_COBS_MAX(n)   ((n) + ((n) / 254u) + 1u)                   /* worst-case encoded size */
#define DW_MAX_WIRE      (DW_COBS_MAX(DW_MAX_PACKET) + 1u)           /* + 0x00 delimiter */

/* ---------------------------------------------------------------- message types */

enum {
    /* device -> host */
    DW_MSG_DEVICE_INFO = 0x01,   /* JSON */
    DW_MSG_STATS       = 0x02,   /* JSON, 1 Hz while streaming */
    DW_MSG_EVENT       = 0x03,   /* dw_event_t + UTF-8 text */
    DW_MSG_SYNC_PONG   = 0x04,   /* dw_sync_pong_t */
    DW_MSG_ACK         = 0x05,   /* dw_ack_t + optional UTF-8 text */
    DW_MSG_LOG         = 0x06,   /* UTF-8 text */
    DW_MSG_FRAME       = 0x10,   /* dw_frame_header_t + records */
    DW_MSG_IMU_BURST   = 0x11,   /* reserved (v1.1) */
    DW_MSG_FORCE_REF   = 0x12,   /* reserved */
    DW_MSG_ENV         = 0x13,   /* reserved */
    /* host -> device */
    DW_CMD_GET_INFO     = 0x80,
    DW_CMD_START        = 0x81,  /* dw_cmd_start_t */
    DW_CMD_STOP         = 0x82,
    DW_CMD_SET_CONFIG   = 0x83,  /* JSON */
    DW_CMD_SYNC_PING    = 0x84,  /* dw_cmd_sync_ping_t */
    DW_CMD_RESET_SENSOR = 0x85,  /* dw_cmd_reset_t */
    DW_CMD_SELFTEST     = 0x86,  /* uint8_t test */
    DW_CMD_SET_MODE     = 0x87,  /* uint8_t mode */
    DW_CMD_REBOOT       = 0x88
};

/* ---------------------------------------------------------------- flags */

/* dw_frame_header_t.frame_flags */
#define DW_FRAME_CONFIG_CHANGED 0x01u   /* first frame after a config change */
#define DW_FRAME_OVERRUN        0x02u   /* this tick took longer than the target period */
#define DW_FRAME_TX_DROP        0x04u   /* >=1 earlier frame was produced but not sent */
#define DW_FRAME_MARKER         0x08u   /* marker button pressed since the previous frame */

/* dw_mag_rec_t.flags */
#define DW_MAG_STALE        0x01u   /* no new conversion; values repeated */
#define DW_MAG_BAD_LENGTH   0x02u   /* status D1:D0 != requested length -> invalid */
#define DW_MAG_CHIP_ERROR   0x04u   /* status ERROR bit set -> invalid */
#define DW_MAG_RAIL         0x08u   /* |axis| >= rail_lsb -> saturated */
#define DW_MAG_COMM_FAIL    0x10u   /* SPI transaction failed -> invalid, fields 0 */
#define DW_MAG_TEMP_STALE   0x20u   /* t_raw repeated from an earlier frame */
#define DW_MAG_RECOVERED    0x40u   /* chip was reset/reconfigured since the previous frame */
#define DW_MAG_INVALID_MASK (DW_MAG_BAD_LENGTH | DW_MAG_CHIP_ERROR | DW_MAG_COMM_FAIL)

/* dw_imu_rec_t.flags */
#define DW_IMU_STALE        0x01u   /* no new sample since the previous frame */
#define DW_IMU_COMM_FAIL    0x02u   /* bus transaction failed -> invalid, fields 0 */
#define DW_IMU_ACC_RAIL     0x04u   /* an accel axis at -32768 or 32767 */
#define DW_IMU_GYR_RAIL     0x08u   /* a gyro axis at -32768 or 32767 */
#define DW_IMU_TEMP_STALE   0x10u   /* temp_raw not refreshed this frame */
#define DW_IMU_RECOVERED    0x20u
#define DW_IMU_INVALID_MASK (DW_IMU_COMM_FAIL)

/* MLX90393 status byte (returned with every command reply) */
#define DW_MLX_ST_BURST_MODE 0x80u
#define DW_MLX_ST_WOC_MODE   0x40u
#define DW_MLX_ST_SM_MODE    0x20u
#define DW_MLX_ST_ERROR      0x10u
#define DW_MLX_ST_SED        0x08u
#define DW_MLX_ST_RS         0x04u
#define DW_MLX_ST_D_MASK     0x03u
/* Data bytes that follow the status byte after RM/RR: 2*D + 2. Reading z,y,x,T needs 8 (D = 3). */
#define DW_MLX_DATA_BYTES(status) (2u * ((status) & DW_MLX_ST_D_MASK) + 2u)

/* ACK status */
enum {
    DW_ACK_OK           = 0,
    DW_ACK_UNKNOWN_TYPE = 1,
    DW_ACK_BAD_PAYLOAD  = 2,
    DW_ACK_BAD_STATE    = 3,
    DW_ACK_FAILED       = 4,
    DW_ACK_VERSION      = 5
};

/* EVENT codes */
enum {
    DW_EV_BOOT            = 0x0001,  /* text: reset reason */
    DW_EV_STREAM_STARTED  = 0x0002,
    DW_EV_STREAM_STOPPED  = 0x0003,
    DW_EV_CONFIG_CHANGED  = 0x0004,  /* text: new config_id */
    DW_EV_MARKER          = 0x0005,  /* marker button */
    DW_EV_MAG_RESET       = 0x0010,
    DW_EV_MAG_FAULT       = 0x0011,
    DW_EV_IMU_RESET       = 0x0020,
    DW_EV_IMU_FAULT       = 0x0021,
    DW_EV_TX_OVERFLOW     = 0x0030,  /* text: frames dropped */
    DW_EV_RATE_NOT_MET    = 0x0031,
    DW_EV_SELFTEST_RESULT = 0x0040   /* text: JSON result */
};
enum { DW_SEV_INFO = 0, DW_SEV_WARNING = 1, DW_SEV_ERROR = 2 };
#define DW_CH_NONE 0xFFu

enum { DW_RESET_MAG = 0, DW_RESET_IMU = 1, DW_RESET_BOTH = 2 };
enum { DW_SELFTEST_PROBE = 0, DW_SELFTEST_ISOLATION = 1, DW_SELFTEST_NOISE = 2 };
enum { DW_MODE_TEXT = 0, DW_MODE_BINARY = 1 };

/* ---------------------------------------------------------------- packed structs */

#if defined(__GNUC__) || defined(__clang__)
#  define DW_PACKED __attribute__((packed))
#else
#  error "dermis_wire.h: define DW_PACKED for this compiler"
#endif

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#  if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#    error "dermis_wire.h assumes a little-endian target"
#  endif
#endif

typedef struct DW_PACKED {
    uint8_t  type;          /* DW_MSG_* / DW_CMD_* */
    uint8_t  version;       /* DW_PROTO_VERSION */
    uint16_t payload_len;
    uint32_t seq;           /* per-direction packet counter */
    uint64_t t_us;          /* esp_timer us; FRAME: trigger time; host->device: 0 */
} dw_header_t;

typedef struct DW_PACKED {
    uint32_t frame_counter; /* +1 every acquisition tick, sent or not */
    uint16_t config_id;
    uint8_t  frame_flags;   /* DW_FRAME_* */
    uint8_t  reserved0;
    uint8_t  n_mag;
    uint8_t  mag_rec_size;  /* sizeof(dw_mag_rec_t) = 12 */
    uint8_t  n_imu;
    uint8_t  imu_rec_size;  /* sizeof(dw_imu_rec_t) = 24 */
    uint16_t acq_us;        /* trigger -> last sensor read done */
    uint16_t reserved1;
} dw_frame_header_t;

typedef struct DW_PACKED {
    uint8_t  ch;
    uint8_t  mlx_status;    /* status byte from the RM reply, unmodified */
    uint8_t  flags;         /* DW_MAG_* */
    uint8_t  reserved;
    int16_t  x, y, z;       /* signed counts, 0 = 0 uT (see dw_mlx_counts) */
    uint16_t t_raw;         /* raw temperature word */
} dw_mag_rec_t;

typedef struct DW_PACKED {
    uint8_t  ch;
    uint8_t  flags;         /* DW_IMU_* */
    int16_t  t_rel_us;      /* read start relative to the frame trigger */
    int16_t  acc[3];        /* chip axes */
    int16_t  gyr[3];
    int16_t  temp_raw;      /* BMI160 TEMPERATURE, 0x8000 = invalid */
    uint16_t reserved;
    uint32_t sensortime;    /* BMI160 SENSORTIME (24 bits), 0 if not read */
} dw_imu_rec_t;

typedef struct DW_PACKED { uint32_t cmd_seq; uint8_t cmd_type; uint8_t status; uint16_t reserved; } dw_ack_t;
typedef struct DW_PACKED { uint16_t code; uint8_t severity; uint8_t ch; } dw_event_t;
typedef struct DW_PACKED { uint64_t host_time; uint64_t t_rx_us; } dw_sync_pong_t;
typedef struct DW_PACKED { uint16_t rate_hz; uint16_t reserved; } dw_cmd_start_t;
typedef struct DW_PACKED { uint64_t host_time; } dw_cmd_sync_ping_t;
typedef struct DW_PACKED { uint8_t kind; uint8_t ch; } dw_cmd_reset_t;

#ifdef __cplusplus
#  define DW_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#  define DW_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
DW_STATIC_ASSERT(sizeof(dw_header_t) == 16, "dw_header_t must be 16 bytes");
DW_STATIC_ASSERT(sizeof(dw_frame_header_t) == 16, "dw_frame_header_t must be 16 bytes");
DW_STATIC_ASSERT(sizeof(dw_mag_rec_t) == 12, "dw_mag_rec_t must be 12 bytes");
DW_STATIC_ASSERT(sizeof(dw_imu_rec_t) == 24, "dw_imu_rec_t must be 24 bytes");
DW_STATIC_ASSERT(sizeof(dw_ack_t) == 8, "dw_ack_t must be 8 bytes");
DW_STATIC_ASSERT(sizeof(dw_event_t) == 4, "dw_event_t must be 4 bytes");
DW_STATIC_ASSERT(sizeof(dw_sync_pong_t) == 16, "dw_sync_pong_t must be 16 bytes");
DW_STATIC_ASSERT(sizeof(dw_cmd_start_t) == 4, "dw_cmd_start_t must be 4 bytes");
DW_STATIC_ASSERT(sizeof(dw_cmd_sync_ping_t) == 8, "dw_cmd_sync_ping_t must be 8 bytes");
DW_STATIC_ASSERT(sizeof(dw_cmd_reset_t) == 2, "dw_cmd_reset_t must be 2 bytes");

/* ---------------------------------------------------------------- CRC-32 */

/* CRC-32/ISO-HDLC, identical to zlib crc32(): dw_crc32("123456789", 9) == 0xCBF43926.
 * On ESP-IDF, esp_rom_crc32_le(0, p, n) should give the same value; check it against
 * the value above once at boot before switching to it. */
static inline uint32_t dw_crc32(const uint8_t *p, size_t n)
{
    uint32_t crc = 0xFFFFFFFFu;
    while (n--) {
        crc ^= *p++;
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

/* ---------------------------------------------------------------- COBS */

/* Encodes n bytes (no delimiter). out needs DW_COBS_MAX(n) bytes. Returns the encoded length.
 * A block that fills to 254 bytes exactly at the end gets no extra 0x01 (same as the Python reference). */
static inline size_t dw_cobs_encode(const uint8_t *in, size_t n, uint8_t *out)
{
    uint8_t *dst = out;
    uint8_t *codep = dst++;
    uint8_t code = 1;
    for (size_t i = 0; i < n; i++) {
        const uint8_t b = in[i];
        if (b) {
            *dst++ = b;
            code++;
        }
        if (!b || code == 0xFF) {
            *codep = code;
            code = 1;
            codep = dst;
            if (!b || i + 1 < n)
                dst++;
        }
    }
    *codep = code;          /* may land one past the returned length; DW_COBS_MAX leaves room */
    return (size_t)(dst - out);
}

/* Decodes n encoded bytes (delimiter not included). Returns the decoded length, or 0 on error. */
static inline size_t dw_cobs_decode(const uint8_t *in, size_t n, uint8_t *out, size_t cap)
{
    size_t i = 0, o = 0;
    while (i < n) {
        const uint8_t code = in[i++];
        if (code == 0)
            return 0;
        for (uint8_t k = 1; k < code; k++) {
            if (i >= n || o >= cap || in[i] == 0)
                return 0;
            out[o++] = in[i++];
        }
        if (code != 0xFF && i < n) {
            if (o >= cap)
                return 0;
            out[o++] = 0;
        }
    }
    return o;
}

/* ---------------------------------------------------------------- packets */

/* Builds one wire packet into out: COBS(header + payload + CRC) followed by 0x00.
 * scratch needs DW_HEADER_SIZE + payload_len + DW_CRC_SIZE bytes; out needs
 * DW_COBS_MAX(that) + 1 bytes (DW_MAX_PACKET / DW_MAX_WIRE always suffice).
 * Returns the number of bytes to send, or 0 if the payload is too large. */
static inline size_t dw_build_packet(uint8_t type, uint32_t seq, uint64_t t_us,
                                     const void *payload, size_t payload_len,
                                     uint8_t *scratch, uint8_t *out)
{
    if (payload_len > DW_MAX_PAYLOAD)
        return 0;
    dw_header_t h;
    h.type = type;
    h.version = (uint8_t)DW_PROTO_VERSION;
    h.payload_len = (uint16_t)payload_len;
    h.seq = seq;
    h.t_us = t_us;
    memcpy(scratch, &h, sizeof h);
    if (payload_len)
        memcpy(scratch + DW_HEADER_SIZE, payload, payload_len);
    const uint32_t crc = dw_crc32(scratch, DW_HEADER_SIZE + payload_len);
    memcpy(scratch + DW_HEADER_SIZE + payload_len, &crc, DW_CRC_SIZE);
    size_t n = dw_cobs_encode(scratch, DW_HEADER_SIZE + payload_len + DW_CRC_SIZE, out);
    out[n++] = 0x00;
    return n;
}

/* Assembles a FRAME payload (frame header + mag records + IMU records).
 * Returns the payload length, or 0 if it doesn't fit in cap. */
static inline size_t dw_build_frame_payload(uint8_t *buf, size_t cap,
                                            uint32_t frame_counter, uint16_t config_id,
                                            uint8_t frame_flags, uint16_t acq_us,
                                            const dw_mag_rec_t *mag, uint8_t n_mag,
                                            const dw_imu_rec_t *imu, uint8_t n_imu)
{
    const size_t len = sizeof(dw_frame_header_t) + (size_t)n_mag * sizeof(dw_mag_rec_t)
                     + (size_t)n_imu * sizeof(dw_imu_rec_t);
    if (len > cap || len > DW_MAX_PAYLOAD)
        return 0;
    dw_frame_header_t fh;
    memset(&fh, 0, sizeof fh);
    fh.frame_counter = frame_counter;
    fh.config_id = config_id;
    fh.frame_flags = frame_flags;
    fh.n_mag = n_mag;
    fh.mag_rec_size = (uint8_t)sizeof(dw_mag_rec_t);
    fh.n_imu = n_imu;
    fh.imu_rec_size = (uint8_t)sizeof(dw_imu_rec_t);
    fh.acq_us = acq_us;
    memcpy(buf, &fh, sizeof fh);
    if (n_mag)
        memcpy(buf + sizeof fh, mag, (size_t)n_mag * sizeof(dw_mag_rec_t));
    if (n_imu)
        memcpy(buf + sizeof fh + (size_t)n_mag * sizeof(dw_mag_rec_t), imu,
               (size_t)n_imu * sizeof(dw_imu_rec_t));
    return len;
}

/* A validated packet; payload points into the caller's decode buffer. */
typedef struct {
    dw_header_t hdr;
    const uint8_t *payload;
} dw_packet_t;

/* Validates one COBS-decoded packet. Returns 0 if good, negative otherwise. */
static inline int dw_check_packet(const uint8_t *p, size_t n, dw_packet_t *pkt)
{
    if (n < DW_HEADER_SIZE + DW_CRC_SIZE)
        return -1;                                   /* too short */
    dw_header_t h;
    memcpy(&h, p, sizeof h);
    if ((size_t)DW_HEADER_SIZE + h.payload_len + DW_CRC_SIZE != n)
        return -2;                                   /* length mismatch */
    uint32_t crc;
    memcpy(&crc, p + DW_HEADER_SIZE + h.payload_len, DW_CRC_SIZE);
    if (crc != dw_crc32(p, DW_HEADER_SIZE + h.payload_len))
        return -3;                                   /* CRC */
    if ((h.version >> 4) != DW_PROTO_MAJOR)
        return -4;                                   /* major version */
    if (pkt) {
        pkt->hdr = h;
        pkt->payload = p + DW_HEADER_SIZE;
    }
    return 0;
}

/* ---------------------------------------------------------------- command receiver */

#ifndef DW_RX_CAP
#define DW_RX_CAP 1536u      /* largest encoded command accepted (SET_CONFIG JSON) */
#endif

typedef struct {
    uint8_t  enc[DW_RX_CAP];
    size_t   len;
    uint8_t  overflow;       /* discard until the next 0x00 */
    uint32_t n_ok, n_bad;
} dw_rx_t;

static inline void dw_rx_init(dw_rx_t *rx) { memset(rx, 0, sizeof *rx); }

/* Feed one received byte. Returns 1 when a valid packet was decoded into pkt_buf
 * (pkt->hdr filled, pkt->payload points into pkt_buf); otherwise 0. */
static inline int dw_rx_push(dw_rx_t *rx, uint8_t byte, uint8_t *pkt_buf, size_t pkt_cap, dw_packet_t *pkt)
{
    if (byte != 0) {
        if (rx->len < DW_RX_CAP)
            rx->enc[rx->len++] = byte;
        else
            rx->overflow = 1;
        return 0;
    }
    const size_t n = rx->len;
    const uint8_t ovf = rx->overflow;
    rx->len = 0;
    rx->overflow = 0;
    if (n == 0)
        return 0;                                    /* back-to-back delimiters */
    if (ovf) {
        rx->n_bad++;
        return 0;
    }
    const size_t m = dw_cobs_decode(rx->enc, n, pkt_buf, pkt_cap);
    if (m == 0 || dw_check_packet(pkt_buf, m, pkt) != 0) {
        rx->n_bad++;
        return 0;
    }
    rx->n_ok++;
    return 1;
}

/* ---------------------------------------------------------------- sensor helpers */

/* MLX90393 axis word -> signed counts with 0 = 0 uT, for TCMP_EN = 0 (the Dermis default).
 * RES 0/1 are two's complement already; RES 2 and 3 are offset. Confirm against the
 * datasheet's output-format table before changing RES or enabling TCMP_EN. */
static inline int16_t dw_mlx_counts(uint16_t raw, uint8_t res)
{
    switch (res) {
    case 2:  return (int16_t)((int32_t)raw - 32768);
    case 3:  return (int16_t)((int32_t)raw - 16384);
    default: return (int16_t)raw;
    }
}

/* Sets DW_MAG_RAIL if any axis reaches the configured rail (DEVICE_INFO mag[].rail_lsb). */
static inline uint8_t dw_mag_rail_flag(const dw_mag_rec_t *m, const int16_t rail[3])
{
    const int16_t v[3] = { m->x, m->y, m->z };
    for (int i = 0; i < 3; i++) {
        const int32_t a = v[i] < 0 ? -(int32_t)v[i] : (int32_t)v[i];
        if (a >= rail[i])
            return DW_MAG_RAIL;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* DERMIS_WIRE_H */
