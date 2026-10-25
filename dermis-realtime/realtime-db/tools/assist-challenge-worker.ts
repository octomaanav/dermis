/**
 * Authorized worker for the actual learned three-finger assistive-grasp simulation.
 *
 * SpacetimeDB coordinates the request and shares the measured PyBullet frames; this
 * process owns local physics and neural-policy inference.  It never accepts a path
 * or motor target from a browser.
 */
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { join, resolve } from 'node:path';
import { connect, option, sleep, waitForTables } from './client.ts';

type Row = { t: number; robot_joints: number[]; mug_pos: number[]; mug_quat: number[]; contact: number[];
  applied_load_n: number; displacement_m: number; outcome: 'IN_PROGRESS' | 'SUCCESS' | 'DROPPED' | 'FAILED';
  hold_state: string; phase: string };
const sessionId = option('session-id');
const tokenPath = option('token-file', '.dermis-tokens/assist-room.token')!;
if (!sessionId) throw new Error('Usage: npm run worker:assist -- --session-id SESSION_ID [--token-file PATH]');
const root = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const python = process.env.DERMIS_PYTHON ?? (existsSync(join(root, '.venv/bin/python')) ? join(root, '.venv/bin/python') : 'python3');
const checkpoint = 'runs/assist/candidates/fallback_scripted/checkpoint';
let stopping = false;
const active = new Set<string>();
process.on('SIGINT', () => { stopping = true; });

const conn = await connect(tokenPath);
await waitForTables(conn);
console.log(`Assistive-grasp worker ready for ${sessionId}; fixed checkpoint: ${checkpoint}`);

async function runPython(requestId: string) {
  const child = spawn(python, ['-u', 'dermis-realtime/realtime-db/tools/assist-trace-exporter.py', '--checkpoint', checkpoint, '--seed', '2000000', '--realtime'],
    { cwd: root, stdio: ['ignore', 'pipe', 'inherit'] });
  const completion = new Promise<number | null>((resolveCode, reject) => {
    child.once('error', reject);
    child.once('close', resolveCode);
  });
  let sequence = conn.db.latestState.sessionId.find(sessionId!)?.sequence ?? 0n;
  let final: Row | undefined;
  const lines = createInterface({ input: child.stdout! });
  try {
    for await (const line of lines) {
      if (stopping) { child.kill('SIGTERM'); throw new Error('worker stopped before challenge completed'); }
      const parsed = JSON.parse(line) as Row | { header: true };
      if ('header' in parsed) continue;
      sequence += 1n;
      final = parsed;
      await conn.reducers.publishState({ sessionId: sessionId!, streamGeneration: 1, sequence,
        sourceTimeMicros: BigInt(Math.round(parsed.t * 1_000_000)), clockDomain: 'PYBULLET_LOCAL_SIMULATION',
        robotJoints: parsed.robot_joints, cupPosition: parsed.mug_pos, cupOrientation: parsed.mug_quat,
        appliedDownwardLoad: parsed.applied_load_n, fingertipNormalLoads: parsed.contact,
        cupRelativeDisplacement: parsed.displacement_m, outcome: parsed.outcome });
      if (sequence % 30n === 0n || parsed.outcome !== 'IN_PROGRESS') await conn.reducers.updateJob({
        jobId: `${requestId}-evaluation`, sessionId: sessionId!, type: 'EVALUATION', status: 'RUNNING',
        datasetRef: 'assistive-grasp-scenario-seed-2000000', checkpointRef: checkpoint, progress: undefined,
        resultSummary: `Local three-finger PyBullet state: ${parsed.hold_state}; phase ${parsed.phase}.`,
      });
    }
    const code = await completion;
    if (code !== 0) throw new Error(`local Python worker exited with ${code}`);
    if (!final) throw new Error('local Python worker emitted no state frames');
    return final;
  } finally {
    if (!child.killed) child.kill('SIGTERM');
  }
}

async function execute(requestId: string) {
  active.add(requestId);
  try {
    await conn.reducers.claimTask({ requestId });
    await conn.reducers.updateTask({ requestId, status: 'RUNNING', resultSummary: 'Local three-finger learned-hand simulation is running.' });
    await conn.reducers.updateJob({ jobId: `${requestId}-evaluation`, sessionId: sessionId!, type: 'EVALUATION', status: 'RUNNING',
      datasetRef: 'assistive-grasp-scenario-seed-2000000', checkpointRef: checkpoint, progress: 0,
      resultSummary: 'Running local PyBullet physics and the fixed learned assistive-grasp policy.' });
    const final = await runPython(requestId);
    const success = final.outcome === 'SUCCESS';
    const terminal = success ? 'SUCCEEDED' : 'FAILED';
    await conn.reducers.updateTask({ requestId, status: terminal, resultSummary: `Actual three-finger simulation completed: ${final.outcome}.` });
    await conn.reducers.updateJob({ jobId: `${requestId}-evaluation`, sessionId: sessionId!, type: 'EVALUATION', status: terminal,
      datasetRef: 'assistive-grasp-scenario-seed-2000000', checkpointRef: checkpoint, progress: 1,
      resultSummary: `Actual local learned assistive-grasp simulation completed: ${final.outcome}.` });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`challenge ${requestId}: ${message}`);
    try { await conn.reducers.updateTask({ requestId, status: 'FAILED', resultSummary: `Local worker failed: ${message}` }); } catch { /* claim state remains auditable */ }
  } finally { active.delete(requestId); }
}

while (!stopping) {
  for (const request of conn.db.taskRequest.sessionId.filter(sessionId)) {
    if (request.taskType === 'AUTONOMOUS_ASSISTIVE_GRASP' && request.status === 'QUEUED' && !active.has(request.requestId)) void execute(request.requestId);
  }
  await sleep(150);
}
conn.disconnect();
