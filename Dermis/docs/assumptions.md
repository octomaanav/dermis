# Assumptions

- The first release targets macOS 15 and uses Swift 6 strict concurrency.
- Developer ID builds are initially unsandboxed; a future sandboxed build is documented separately.
- The current four channels have no durable pod-location mapping. The UI labels them S1–S4 until Identify Pods is completed.
- Newton values are never displayed without an active valid calibration. Phase 0 displays field values in mT.
- The simulator is the default device on first launch and is visibly marked `SIMULATED`.
- The current legacy stream is 921600 baud CSV and remains supported alongside Wire Protocol v1.
- SPI remains at 1 MHz until a physical bus change passes the rate sweep.
- The existing firmware folder is authoritative for the current hub implementation.

