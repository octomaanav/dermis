/*
 * Host-side check that dermis_wire.h produces the reference bytes.
 *
 *   cc -std=c11 -Wall -Wextra -O2 -o dw_test test_vectors.c
 *   ./dw_test > c_output.txt
 *   python3 ../host/python/dermis_wire.py check-vectors c_output.txt
 *
 * Prints one "name hex" line per vector; exits non-zero if a self-check fails.
 */
#include <stdio.h>
#include "dermis_wire.h"

static uint8_t scratch[DW_MAX_PACKET];
static uint8_t wire[DW_MAX_WIRE];
static uint8_t payload[DW_MAX_PAYLOAD];
static uint8_t decoded[DW_MAX_PACKET];

static void print_hex(const char *name, const uint8_t *b, size_t n)
{
    printf("%s ", name);
    for (size_t i = 0; i < n; i++)
        printf("%02x", b[i]);
    printf("\n");
}

static int fail(const char *what)
{
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

int main(void)
{
    size_t n;

    if (dw_crc32((const uint8_t *)"123456789", 9) != 0xCBF43926u)
        return fail("CRC-32 check value");

    /* host -> device */
    n = dw_build_packet(DW_CMD_GET_INFO, 0, 0, NULL, 0, scratch, wire);
    print_hex("get_info", wire, n);

    dw_cmd_start_t start = { 0, 0 };
    n = dw_build_packet(DW_CMD_START, 1, 0, &start, sizeof start, scratch, wire);
    print_hex("start", wire, n);

    dw_cmd_sync_ping_t ping = { 0x0123456789ABCDEFull };
    n = dw_build_packet(DW_CMD_SYNC_PING, 2, 0, &ping, sizeof ping, scratch, wire);
    print_hex("sync_ping", wire, n);

    /* device -> host: the reference FRAME */
    const dw_mag_rec_t mag[4] = {
        { 0, 0x03, 0,                 0,     52,   -118,   6973, 46109 },
        { 1, 0x03, 0,                 0,   -210,     64,   7105, 46107 },
        { 2, 0x03, DW_MAG_TEMP_STALE, 0,     15,    -35,   6850, 46056 },
        { 3, 0x01, DW_MAG_BAD_LENGTH, 0, -20432, -20432, -20432,     0 },
    };
    const dw_imu_rec_t imu[4] = {
        { 0, 0,            212, {    35,   -60, 4090 }, {   12,  -8,  3 }, 552, 0, 0x00A1B2C3u },
        { 1, 0,            431, { -1210,  2950, 2700 }, { -150,  95, 40 }, 540, 0, 0x00A1B2D0u },
        { 2, DW_IMU_STALE, 650, {    20,   -45, 4101 }, {    5,   2, -4 }, 548, 0, 0x00A1B2DCu },
        { 3, 0,            869, {   400, -3900,  900 }, {   30, -22, 18 }, 560, 0, 0x00A1B2E8u },
    };
    const size_t plen = dw_build_frame_payload(payload, sizeof payload, 1234, 3, DW_FRAME_MARKER, 1650,
                                               mag, 4, imu, 4);
    if (plen != 160)
        return fail("frame payload length");
    n = dw_build_packet(DW_MSG_FRAME, 42, 12345678ull, payload, plen, scratch, wire);
    if (n != 182)
        return fail("frame wire length");
    print_hex("frame", wire, n);

    /* the receiver must accept it and reject a corrupted copy */
    {
        dw_rx_t rx;
        dw_packet_t pkt;
        int got = 0;
        dw_rx_init(&rx);
        /* DW_RX_CAP is sized for commands; use a big local receiver for the frame */
        for (size_t i = 0; i < n; i++)
            got += dw_rx_push(&rx, wire[i], decoded, sizeof decoded, &pkt);
        if (got != 1 || pkt.hdr.type != DW_MSG_FRAME || pkt.hdr.payload_len != 160)
            return fail("receiver did not accept the frame");
        if (memcmp(pkt.payload, payload, plen) != 0)
            return fail("receiver payload differs");
        wire[30] ^= 0x01;
        got = 0;
        for (size_t i = 0; i < n; i++)
            got += dw_rx_push(&rx, wire[i], decoded, sizeof decoded, &pkt);
        if (got != 0 || rx.n_bad != 1)
            return fail("receiver accepted a corrupted frame");
    }

    dw_ack_t ack = { 0, DW_CMD_GET_INFO, DW_ACK_OK, 0 };
    n = dw_build_packet(DW_MSG_ACK, 7, 12000000ull, &ack, sizeof ack, scratch, wire);
    print_hex("ack", wire, n);

    dw_sync_pong_t pong = { 0x0123456789ABCDEFull, 12000120ull };
    n = dw_build_packet(DW_MSG_SYNC_PONG, 8, 12000150ull, &pong, sizeof pong, scratch, wire);
    print_hex("sync_pong", wire, n);

    uint8_t ev[sizeof(dw_event_t) + 6];
    const dw_event_t e = { DW_EV_MARKER, DW_SEV_INFO, DW_CH_NONE };
    memcpy(ev, &e, sizeof e);
    memcpy(ev + sizeof e, "button", 6);
    n = dw_build_packet(DW_MSG_EVENT, 9, 12100000ull, ev, sizeof ev, scratch, wire);
    print_hex("event_marker", wire, n);

    /* COBS edge: 254 non-zero bytes encode to 255 bytes with no trailing code byte */
    {
        uint8_t in[254], enc[DW_COBS_MAX(254)], dec[254];
        for (int i = 0; i < 254; i++)
            in[i] = (uint8_t)(i + 1);
        const size_t m = dw_cobs_encode(in, 254, enc);
        if (m != 255 || enc[0] != 0xFF)
            return fail("COBS 254-byte block");
        if (dw_cobs_decode(enc, m, dec, sizeof dec) != 254 || memcmp(in, dec, 254) != 0)
            return fail("COBS 254-byte round trip");
    }

    /* MLX helpers */
    if (dw_mlx_counts(32768u + 6973u, 2) != 6973 || dw_mlx_counts(65535u, 0) != -1)
        return fail("dw_mlx_counts");
    {
        const int16_t rail[3] = { 22000, 22000, 22000 };
        const dw_mag_rec_t sat = { 0, 0x03, 0, 0, 10, -22000, 5, 0 };
        if (dw_mag_rail_flag(&sat, rail) != DW_MAG_RAIL || dw_mag_rail_flag(&mag[0], rail) != 0)
            return fail("dw_mag_rail_flag");
    }

    fprintf(stderr, "self-checks passed\n");
    return 0;
}
