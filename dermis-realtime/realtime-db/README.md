# Dermis shared state

SpacetimeDB coordination backend and a minimal React observer for Dermis. It stores a bounded current snapshot per session, shared annotations, and job status. It deliberately contains no physics, model inference, training, glove control, or public ingestion endpoint.

Database: `dermis-realtime-db`. The checked-in SDK and CLI target SpacetimeDB `2.13.*` / CLI `2.13.0`.

## Run locally

In separate terminals:

```bash
spacetime start
cd dermis-realtime/realtime-db
npm run spacetime:publish:local
npm run spacetime:generate
npm run dev
```

Set `VITE_SPACETIMEDB_HOST=ws://127.0.0.1:3000` and `VITE_SPACETIMEDB_DB_NAME=dermis-realtime-db` only when the defaults are not suitable. The browser stores its identity token in `localStorage`; Node tools store their token in `.dermis-tokens/`, which is ignored by git. Use the same secure token file for `create_session` and the process that publishes its state.

## Public Maincloud + static dashboard deployment

For a judge-facing deployment, host the database on Maincloud and the built `dist/` directory on any static HTTPS host. The database and the browser use public URLs; the local Python worker makes an outbound connection to Maincloud, so no device needs the Mac's LAN IP or an open inbound port.

```bash
# One time: authenticate the CLI with the Maincloud account that owns the database.
spacetime login

# Publish/update the existing public Dermis database.
spacetime publish dermis-realtime-db --module-path spacetimedb --server maincloud --yes

# Build the browser app with the public database endpoint baked in.
VITE_SPACETIMEDB_HOST=https://maincloud.spacetimedb.com \
VITE_SPACETIMEDB_DB_NAME=dermis-realtime-db \
npm run build
```

Deploy `dist/` to a static host such as Netlify, Vercel, or Cloudflare Pages. The included `public/_redirects` is copied to `dist/_redirects` by Vite and makes Netlify-style static hosts serve `index.html` for `/room/SESSION_ID` QR links. Configure the equivalent SPA fallback if using another host.

When the local worker should publish to Maincloud, use a fresh cloud token file; local-server tokens cannot authenticate there:

```bash
export SPACETIMEDB_HOST=https://maincloud.spacetimedb.com
export SPACETIMEDB_DB_NAME=dermis-realtime-db

npm run room:assist -- --token-file .dermis-tokens/assist-maincloud.token
npm run worker:assist -- --session-id SESSION_ID --token-file .dermis-tokens/assist-maincloud.token
```

The Maincloud web dashboard is for database inspection, queries, logs, and usage. Share the deployed React dashboard—not the Maincloud administration URL—with judges.

## Synthetic publisher

```bash
npm run publisher -- --duration 30
```

It prints `SESSION_ID=...`, creates a `SYNTHETIC` / `MANUAL` session, publishes predictable 10 Hz telemetry, and transitions an `EVALUATION` job from queued to running to succeeded. Its summaries explicitly state that no model ran.

## Neural-policy cup replay

Use the existing `fragile_cup` neural-policy worker trace for an honest, data-rich dashboard demo. This publisher sends recorded PyBullet/checkpoint states at 10 Hz with source `REPLAY` and mode `AUTONOMOUS`; it never claims inference is running inside SpacetimeDB.

```bash
# Generate an actual trace from the existing trained cup policy once, if needed.
python -m dermis cup worker \
  --checkpoint runs/cup/autonomous_demo/cup_autonomous_demo_v1 \
  --seed 2000000 --out /tmp/cup-neural-trace.jsonl

# Loop it so judges joining at any point see a changing, labelled replay.
npm run publisher:cup-trace -- \
  --trace /tmp/cup-neural-trace.jsonl --loop --hz 10
```

## Collaborative judge demo

Open the viewer on two devices pointed at the same SpacetimeDB server. Start a publisher and select its session on both screens. The observer can add an annotation or press **Request autonomous cup trial**; both actions are persistent shared records delivered by subscriptions.

For a same-Wi-Fi local demo, leave `VITE_SPACETIMEDB_HOST` unset (the supplied `.env.local` does this), run Vite on all interfaces, then open the displayed LAN URL on both devices. The viewer derives its WebSocket URL from that hostname, while `spacetime start` listens on all interfaces by default. Allow the ports through macOS Firewall if prompted.

```bash
npm run dev -- --host 0.0.0.0
# Open http://YOUR-MAC-LAN-IP:5173 on both devices.
```

The task button is deliberately not a backdoor into the simulator. It creates one `AUTONOMOUS_CUP_TRIAL` request through a reducer. Only the session publisher identity can atomically claim it, and only that claimant can advance it through `QUEUED → CLAIMED → RUNNING → SUCCEEDED|FAILED`. Task events provide a shared history.

Run the local task worker with the same publisher token. It executes only the explicit local runner you provide; the runner receives `SESSION_ID REQUEST_ID`, controls the simulator locally, and must continue publishing telemetry through the existing bridge. A zero exit code means only that the runner completed—not that the cup trial succeeded.

```bash
npm run worker -- \
  --session-id SESSION_ID \
  --token-file .dermis-tokens/synthetic-publisher.token \
  --command /absolute/path/to/local-simulator-runner
```

The viewer's engineering panel reports actual currently connected SDK clients, its locally observed snapshot rate, latest sequence, and server-receipt age. It intentionally does not claim end-to-end latency.

## Actual local neural-policy challenge room

This is the genuine two-device request/state/result path. It uses the fixed compatible checkpoint `runs/cup/autonomous_demo/cup_autonomous_demo_v1`; requests cannot provide a shell command, checkpoint path, or arbitrary load value. The local worker claims the existing allowlisted `AUTONOMOUS_CUP_TRIAL`, runs Python/PyBullet, then publishes the resulting state at 15 Hz through the supported TypeScript SDK.

```bash
# Terminal 1: local database (once per machine)
spacetime start

# Terminal 2: frontend, reachable to devices on the same Wi-Fi
cd dermis-realtime/realtime-db
npm run dev -- --host 0.0.0.0

# Terminal 3: create a shared learned-policy room. Copy SESSION_ID.
npm run room:cup -- --token-file .dermis-tokens/cup-room.token

# Terminal 4: the authorized local worker. It must use the same token.
npm run worker:cup -- --session-id SESSION_ID --token-file .dermis-tokens/cup-room.token
```

Open `http://MAC_CURRENT_WIFI_IP:5173` on the desktop. Its QR code links a judge phone to `/room/SESSION_ID`; the phone can request one cup trial and mark a timestamped moment. Both devices must use the same non-isolated Wi-Fi network. The worker prefers `DERMIS_PYTHON` when set, otherwise this repository's `.venv/bin/python`.

The verification run used an independent judge identity and produced request `verify-cup-challenge-2`: it was claimed once, published 231 `PYBULLET_LOCAL_SIMULATION` snapshots, and completed `SUCCESS`. The earlier replay publisher remains a distinct, visibly labelled recorded-playback demo—not a live worker.

## Raw data samples for visualizers

`raw_data_samples` is a public historical archive for visualizers. It is deliberately separate from the single-row `latest_state` table: live views subscribe to `latest_state`; review, charting, or replay apps subscribe to `raw_data_samples` and filter on `session_id`.

Each row is one ordered, immutable `GZIP_NDJSON_UTF8` batch. It includes the sequence/time range and `sample_count` for selective loading, plus a compressed `payload` byte array. Publishers may write at most 256 samples, 256 KiB compressed, and 4 MiB uncompressed per batch. The database validates that chunks append in sequence/time order and only a session's authorized publisher may create them.

The synthetic publisher now writes ten frames per archive row. Each decompressed NDJSON line follows this stable contract:

```json
{"schema_version":1,"sequence":42,"source_time_us":4200000,"clock_domain":"SYNTHETIC_MONOTONIC","robot_joints":[0.0,0.2],"cup_position":[0.45,0,0.18],"cup_orientation":[0,0,0,1],"applied_downward_load":3.2,"fingertip_normal_loads":[0.8,1.1,0.7],"cup_relative_displacement":0.002,"outcome":"IN_PROGRESS"}
```

For example, a browser visualizer can inflate one subscribed row without a server round trip:

```ts
async function decodeRawBatch(row: { payload: number[] }) {
  const compressed = Uint8Array.from(row.payload);
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const text = await new Response(stream).text();
  return text.trim().split('\n').map(line => JSON.parse(line));
}
```

Raw data is an archive, not a queue: a visualizer should sort rows by `chunk_sequence`, use their sequence ranges to request only the needed chunks, and never use it for low-latency control. For very long recordings, export completed compressed batches to durable object storage rather than treating a realtime database as unlimited cold storage.

## Three-finger assistive-hand room

The three-finger hand is a different PyBullet task from `fragile_cup`; do not use the cup worker to represent it. The separate bridge imports the existing learned assistive-hand checkpoint, runs its local PyBullet simulation, and shares the measured nine phalanx angles, mug pose, three contact loads, applied load, displacement, and outcome. It never gives a browser motor control.

```bash
# Terminal 3: create the correct three-finger room. Copy SESSION_ID.
npm run room:assist -- --token-file .dermis-tokens/assist-room.token

# Terminal 4: authorized local physics + learned-policy worker.
npm run worker:assist -- --session-id SESSION_ID --token-file .dermis-tokens/assist-room.token
```

On either subscribed browser, choose that **Three-finger learned policy** room and press **Request three-finger grasp**. The dashboard uses the session task to render the three digits rather than the cup task's two-finger Panda gripper. The task reducer rejects a cup request in an assistive-hand session (and vice versa), preventing another source mismatch.

## Python stdin bridge

The bridge is an authenticated Node TypeScript SDK client; it does not require or assume a Python SDK.

```bash
npm run bridge:stdin -- \
  --session-id SESSION_ID \
  --token-file .dermis-tokens/synthetic-publisher.token
```

It accepts one UTF-8 NDJSON snapshot per line on stdin, validates it, holds at most one unsent snapshot, and prefers the latest state while a publish is in flight. Calls are never made from Python, so a database outage does not block physics. An uncertain/failed publish is logged and not replayed: the producer must continue with a new increasing sequence. Automatic reconnect is enabled for the SDK connection.

JSON contract (all numeric values must be finite; `sequence` and `source_time_us` must be safe integers):

```json
{"sequence":42,"source_time_us":4200000,"clock_domain":"PYTHON_MONOTONIC","robot_joints":[0.0,0.2],"cup_position":[0.45,0.0,0.18],"cup_orientation":[0,0,0,1],"applied_downward_load":3.2,"fingertip_normal_loads":[0.8,1.1,0.7],"cup_relative_displacement":0.002,"outcome":"IN_PROGRESS"}
```

Example producer (replace the values with the Python visualizer's existing output):

```python
import json, sys
print(json.dumps({"sequence": 1, "source_time_us": 100000,
  "clock_domain": "PYTHON_MONOTONIC", "robot_joints": [0.0, 0.2],
  "cup_position": [0.45, 0.0, 0.18], "cup_orientation": [0, 0, 0, 1],
  "applied_downward_load": 3.2, "fingertip_normal_loads": [0.8, 1.1, 0.7],
  "cup_relative_displacement": 0.002, "outcome": "IN_PROGRESS"}), flush=True)
```

## State and ownership

- `session` stores ownership, publisher identity, task/version, source, control mode, checkpoint, status, and stream generation.
- `latest_state` is keyed by session ID, so it is replaced rather than accumulated.
- `annotation` is timestamped in the episode/reference clock. Public sessions permit observer annotations; sessions may opt out at creation.
- `job` is an upserted current status record.

All writes are transactional reducers. Only the session publisher identity can publish state or update jobs. `publish_state` rejects non-finite data, invalid shapes, a wrong stream generation, duplicate sequences, and out-of-order sequences. A restarted publisher must create a new session ID; it cannot reset an old stream's sequence counter.

## Verify

```bash
cd spacetimedb && npm test
cd .. && npm run build && npm test
npm run test:integration
```

The integration command requires the local server and module already published. It verifies two subscriptions see the same state and annotation, a fresh client receives the current snapshot, unauthorized state writes fail, duplicate/out-of-order writes fail, stale detection is covered by the viewer test, and a sequence reset is rejected until a new session is created.
