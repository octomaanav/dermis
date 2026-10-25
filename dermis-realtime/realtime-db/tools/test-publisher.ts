import { randomUUID } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { connect, option, sleep, waitForTables } from './client.ts';

type RawSnapshot = {
  schema_version: 1; sequence: number; source_time_us: number; clock_domain: string; robot_joints: number[];
  cup_position: number[]; cup_orientation: number[]; applied_downward_load: number;
  fingertip_normal_loads: number[]; cup_relative_displacement: number; outcome: string;
};

const tokenPath = option('token-file', '.dermis-tokens/synthetic-publisher.token')!;
const durationSeconds = Number(option('duration', '12'));
const sessionId = option('session-id', `synthetic-${randomUUID()}`)!;
let stopping = false;
process.on('SIGINT', () => { stopping = true; });

const conn = await connect(tokenPath);
await waitForTables(conn);
await conn.reducers.createSession({ sessionId, task: 'cup_load_demo', version: 'synthetic-v1', source: 'SYNTHETIC',
  controlMode: 'MANUAL', checkpointId: undefined, allowAnnotations: true });
await conn.reducers.updateJob({ jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: 'QUEUED',
  datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: 0, resultSummary: undefined });
console.log(`SESSION_ID=${sessionId}`);
console.log('Synthetic publisher started: no model inference or training is running.');

let rawChunkSequence = 0n;
let rawBatch: RawSnapshot[] = [];
async function archiveRawBatch() {
  if (!rawBatch.length) return;
  rawChunkSequence += 1n;
  const ndjson = `${rawBatch.map(sample => JSON.stringify(sample)).join('\n')}\n`;
  const compressed = gzipSync(ndjson);
  const first = rawBatch[0]; const last = rawBatch.at(-1)!;
  await conn.reducers.publishRawDataSamples({
    chunkId: `${sessionId}.g1.c${rawChunkSequence}`, sessionId, streamGeneration: 1, chunkSequence: rawChunkSequence,
    sampleCount: rawBatch.length, firstSequence: BigInt(first.sequence), lastSequence: BigInt(last.sequence),
    firstSourceTimeMicros: BigInt(first.source_time_us), lastSourceTimeMicros: BigInt(last.source_time_us),
    encoding: 'GZIP_NDJSON_UTF8', uncompressedBytes: Buffer.byteLength(ndjson), payload: [...compressed],
  });
  rawBatch = [];
}

for (let index = 1; !stopping && index <= durationSeconds * 10; index += 1) {
  const t = index / 10;
  const load = 4 + 2 * Math.sin(t * 2);
  const raw: RawSnapshot = { schema_version: 1, sequence: index, source_time_us: Math.round(t * 1_000_000),
    clock_domain: 'SYNTHETIC_MONOTONIC', robot_joints: [0.2 * Math.sin(t), 0.35 * Math.sin(t * 1.2), 0.1 * Math.cos(t)],
    cup_position: [0.45 + 0.02 * Math.sin(t), 0, 0.18], cup_orientation: [0, 0, 0, 1],
    applied_downward_load: load, fingertip_normal_loads: [load * .3, load * .33, load * .37],
    cup_relative_displacement: 0.001 + 0.0004 * Math.abs(Math.sin(t)), outcome: 'IN_PROGRESS' };
  await conn.reducers.publishState({ sessionId, streamGeneration: 1, sequence: BigInt(index),
    sourceTimeMicros: BigInt(raw.source_time_us), clockDomain: raw.clock_domain, robotJoints: raw.robot_joints,
    cupPosition: raw.cup_position, cupOrientation: raw.cup_orientation, appliedDownwardLoad: raw.applied_downward_load,
    fingertipNormalLoads: raw.fingertip_normal_loads, cupRelativeDisplacement: raw.cup_relative_displacement, outcome: raw.outcome });
  rawBatch.push(raw);
  if (rawBatch.length === 10) await archiveRawBatch();
  if (index === 10) await conn.reducers.updateJob({ jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: 'RUNNING',
    datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: .5, resultSummary: 'Synthetic telemetry only; no model is running.' });
  await sleep(100);
}
await archiveRawBatch();
await conn.reducers.updateJob({ jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: 'SUCCEEDED',
  datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: 1, resultSummary: 'Synthetic telemetry complete; no model inference ran.' });
console.log('Synthetic publisher stopped cleanly.');
conn.disconnect();
