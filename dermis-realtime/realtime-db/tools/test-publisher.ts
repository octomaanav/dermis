import { randomUUID } from 'node:crypto';
import { connect, option, sleep, waitForTables } from './client.ts';

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

for (let index = 1; !stopping && index <= durationSeconds * 10; index += 1) {
  const t = index / 10;
  const load = 4 + 2 * Math.sin(t * 2);
  await conn.reducers.publishState({ sessionId, streamGeneration: 1, sequence: BigInt(index),
    sourceTimeMicros: BigInt(Math.round(t * 1_000_000)), clockDomain: 'SYNTHETIC_MONOTONIC',
    robotJoints: [0.2 * Math.sin(t), 0.35 * Math.sin(t * 1.2), 0.1 * Math.cos(t)],
    cupPosition: [0.45 + 0.02 * Math.sin(t), 0, 0.18], cupOrientation: [0, 0, 0, 1],
    appliedDownwardLoad: load, fingertipNormalLoads: [load * .3, load * .33, load * .37],
    cupRelativeDisplacement: 0.001 + 0.0004 * Math.abs(Math.sin(t)), outcome: 'IN_PROGRESS' });
  if (index === 10) await conn.reducers.updateJob({ jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: 'RUNNING',
    datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: .5, resultSummary: 'Synthetic telemetry only; no model is running.' });
  await sleep(100);
}
await conn.reducers.updateJob({ jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: 'SUCCEEDED',
  datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: 1, resultSummary: 'Synthetic telemetry complete; no model inference ran.' });
console.log('Synthetic publisher stopped cleanly.');
conn.disconnect();
