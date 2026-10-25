/** Publish an existing fragile_cup neural-policy physics trace to a shared session.
 *
 * The source is explicitly REPLAY: this re-sends recorded PyBullet/checkpoint
 * results and never represents SpacetimeDB as the place model inference runs.
 */
import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { connect, option, sleep, waitForTables } from './client.ts';

type TraceHeader = { header: true; controller: string; schedule_id: string; scenario: { seed: number } };
type TraceRow = {
  t: number; robot_joints: Record<string, number>; cup_pos: number[]; cup_quat: number[];
  contact: { normal_l: number; normal_r: number }; applied_load_n: number;
  status: { slip_cm: number; outcome: string | null; movement: string; load_status: string };
};

const tracePath = option('trace');
const tokenPath = option('token-file', '.dermis-tokens/cup-trace-publisher.token')!;
const hz = Number(option('hz', '10'));
const loop = process.argv.includes('--loop');
if (!tracePath || !Number.isFinite(hz) || hz <= 0 || hz > 30) {
  console.error('Usage: npm run publisher:cup-trace -- --trace PATH [--hz 10] [--loop] [--token-file PATH]');
  process.exit(2);
}

const parsed = readFileSync(tracePath, 'utf8').trim().split(/\r?\n/).map(line => JSON.parse(line));
const header = parsed.shift() as TraceHeader;
if (!header?.header || !header.controller || !header.schedule_id || !parsed.length)
  throw new Error('trace must start with the fragile_cup worker header followed by state rows');
const rows = parsed as TraceRow[];
const sessionId = option('session-id', `cup-neural-replay-${randomUUID()}`)!;
const conn = await connect(tokenPath);
await waitForTables(conn);
await conn.reducers.createSession({ sessionId, task: 'fragile_cup', version: `trace/${header.schedule_id}`,
  source: 'REPLAY', controlMode: 'AUTONOMOUS', checkpointId: header.controller, allowAnnotations: true });
await conn.reducers.updateJob({ jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: 'RUNNING',
  datasetRef: `scenario-seed-${header.scenario.seed}`, checkpointRef: header.controller, progress: 0,
  resultSummary: 'Replaying recorded PyBullet state from the trained neural policy; no inference is running in SpacetimeDB.' });
console.log(`SESSION_ID=${sessionId}`);
console.log(`Replaying ${rows.length} recorded neural-policy simulation ticks from ${header.controller} at ${hz} Hz${loop ? ' (looping)' : ''}.`);

let sequence = 0n;
let stopping = false;
process.on('SIGINT', () => { stopping = true; });
const step = Math.max(1, Math.round(30 / hz));
do {
  for (let index = 0; !stopping && index < rows.length; index += step) {
    const row = rows[index];
    sequence += 1n;
    const outcome = row.status.outcome === 'success' ? 'SUCCESS'
      : row.status.outcome === 'drop' ? 'DROPPED' : row.status.outcome ? 'FAILED' : 'IN_PROGRESS';
    const joints = Object.entries(row.robot_joints).sort(([a], [b]) => Number(a.slice(1)) - Number(b.slice(1))).map(([, value]) => value);
    await conn.reducers.publishState({ sessionId, streamGeneration: 1, sequence,
      sourceTimeMicros: BigInt(Math.round(row.t * 1_000_000)), clockDomain: 'PYBULLET_TRACE_REPLAY',
      robotJoints: joints, cupPosition: row.cup_pos, cupOrientation: row.cup_quat,
      appliedDownwardLoad: row.applied_load_n, fingertipNormalLoads: [row.contact.normal_l, row.contact.normal_r],
      cupRelativeDisplacement: Math.max(0, row.status.slip_cm) / 100, outcome });
    const progress = Math.min(1, (index + step) / rows.length);
    if (index === 0 || index + step >= rows.length) await conn.reducers.updateJob({
      jobId: `${sessionId}-evaluation`, sessionId, type: 'EVALUATION', status: index + step >= rows.length && !loop ? 'SUCCEEDED' : 'RUNNING',
      datasetRef: `scenario-seed-${header.scenario.seed}`, checkpointRef: header.controller, progress,
      resultSummary: index + step >= rows.length && !loop
        ? `Recorded neural-policy trace completed with ${outcome}.`
        : `Replaying PyBullet trace: ${row.status.movement}; load status ${row.status.load_status}.`,
    });
    await sleep(1_000 / hz);
  }
} while (loop && !stopping);
conn.disconnect();
