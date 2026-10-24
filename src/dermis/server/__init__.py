"""TCP telemetry server and the armed TCP teleoperation path (see docs/tcp_teleop.md).

    tcp_server.py   asyncio NDJSON receiver -> validation -> TelemetryPipeline -> LatestSlot
    ndjson.py       byte-stream framing (partial / combined messages, UTF-8, size limit)
    validation.py   JSON Schema check against firmware/v1/schemas/decoded_frame.schema.json
    dev_sender.py   development TCP client: fixture | replay | SYNTHETIC, with fault injection
    teleop/         EXPERIMENTAL mapping, operator calibration, arm/disarm controller, sim loop, check
"""
