/**
 * Authorized local worker for the fixed fragile_cup neural policy.
 *
 * It accepts only the database's allowlisted AUTONOMOUS_CUP_TRIAL request,
 * runs the fixed checkpoint locally in Python, and publishes the measured
 * PyBullet trace. The browser never controls a motor or selects a path.
 */
import { randomUUID } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { connect, option, sleep, waitForTables } from './client.ts';

type TraceRow = { t: number; robot_joints: Record<string, number>; cup_pos: number[]; cup_quat: number[];
  contact: { normal_l: number; normal_r: number }; applied_load_n: number;
  status: { slip_cm: number; outcome: string | null; movement: string; load_status: string } };
const sessionId = option('session-id');
const tokenPath = option('token-file', '.dermis-tokens/cup-room.token')!;
if (!sessionId) throw new Error('Usage: npm run worker:cup -- --session-id SESSION_ID [--token-file PATH]');
const root = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const checkpoint = 'runs/cup/autonomous_demo/cup_autonomous_demo_v1';
// Prefer this repository's environment; an unrelated system Python commonly
// lacks the editable dermis package and PyBullet dependencies.
const python = process.env.DERMIS_PYTHON ?? (existsSync(join(root, '.venv/bin/python')) ? join(root, '.venv/bin/python') : 'python3');
let stopping = false;
const active = new Set<string>();
process.on('SIGINT', () => { stopping = true; });

const conn = await connect(tokenPath);
await waitForTables(conn);
console.log(`Cup challenge worker ready for ${sessionId}; fixed checkpoint: ${checkpoint}`);

function runPython(requestId: string) {
  const directory = mkdtempSync(join(tmpdir(), 'dermis-cup-challenge-'));
  const trace = join(directory, `${requestId}.jsonl`);
  return new Promise<{ rows: TraceRow[]; cleanup: () => void }>((resolvePromise, reject) => {
    const child = spawn(python, ['-m', 'dermis', 'cup', 'worker', '--checkpoint', checkpoint,
      '--seed', '2000000', '--out', trace], { cwd: root, stdio: 'inherit' });
    child.once('error', error => reject(error));
    child.once('close', code => {
      if (code !== 0) return reject(new Error(`local Python worker exited with ${code}`));
      try {
        const rows = readFileSync(trace, 'utf8').trim().split(/\r?\n/).slice(1).map(line => JSON.parse(line) as TraceRow);
        resolvePromise({ rows, cleanup: () => rmSync(directory, { recursive: true, force: true }) });
      } catch (error) { reject(error); }
    });
  });
}

async function execute(requestId: string) {
  active.add(requestId);
  try {
    await conn.reducers.claimTask({ requestId });
    await conn.reducers.updateTask({ requestId, status: 'RUNNING', resultSummary: 'Local simulator is running the fixed trained checkpoint.' });
    await conn.reducers.updateJob({ jobId: `${requestId}-evaluation`, sessionId: sessionId!, type: 'EVALUATION', status: 'RUNNING',
      datasetRef: 'cup-scenario-seed-2000000', checkpointRef: checkpoint, progress: 0,
      resultSummary: 'Running local PyBullet physics and neural-policy inference.' });
    const trace = await runPython(requestId);
    let sequence = conn.db.latestState.sessionId.find(sessionId!)?.sequence ?? 0n;
    for (const [index, row] of trace.rows.entries()) {
      if (stopping) throw new Error('worker stopped before challenge completed');
      sequence += 1n;
      const outcome = row.status.outcome === 'success' ? 'SUCCESS' : row.status.outcome === 'drop' ? 'DROPPED'
        : row.status.outcome ? 'FAILED' : 'IN_PROGRESS';
      const joints = Object.entries(row.robot_joints).sort(([a], [b]) => Number(a.slice(1)) - Number(b.slice(1))).map(([, value]) => value);
      await conn.reducers.publishState({ sessionId: sessionId!, streamGeneration: 1, sequence,
        sourceTimeMicros: BigInt(Math.round(row.t * 1_000_000)), clockDomain: 'PYBULLET_LOCAL_SIMULATION',
        robotJoints: joints, cupPosition: row.cup_pos, cupOrientation: row.cup_quat,
        appliedDownwardLoad: row.applied_load_n, fingertipNormalLoads: [row.contact.normal_l, row.contact.normal_r],
        cupRelativeDisplacement: Math.max(0, row.status.slip_cm) / 100, outcome });
      if (index % 30 === 0 || index === trace.rows.length - 1) await conn.reducers.updateJob({
        jobId: `${requestId}-evaluation`, sessionId: sessionId!, type: 'EVALUATION', status: 'RUNNING',
        datasetRef: 'cup-scenario-seed-2000000', checkpointRef: checkpoint, progress: (index + 1) / trace.rows.length,
        resultSummary: `Local PyBullet state ${index + 1}/${trace.rows.length}: ${row.status.movement}; ${row.status.load_status}.`,
      });
      await sleep(1000 / 15);
    }
    const final = trace.rows.at(-1)!;
    const result = final.status.outcome === 'success' ? 'success' : final.status.outcome ?? 'failed';
    const terminalStatus = result === 'success' ? 'SUCCEEDED' : 'FAILED';
    await conn.reducers.updateTask({ requestId, status: terminalStatus, resultSummary: `Actual local simulation completed: ${result}.` });
    await conn.reducers.updateJob({ jobId: `${requestId}-evaluation`, sessionId: sessionId!, type: 'EVALUATION', status: terminalStatus,
      datasetRef: 'cup-scenario-seed-2000000', checkpointRef: checkpoint, progress: 1,
      resultSummary: `Actual local neural-policy simulation completed: ${result}.` });
    trace.cleanup();
  } catch (error) {
    console.error(`challenge ${requestId}: ${error instanceof Error ? error.message : error}`);
    // Only transition a claim we own. An uncertain process stop deliberately stays visible as RUNNING.
    try { await conn.reducers.updateTask({ requestId, status: 'FAILED', resultSummary: `Local worker failed before completion: ${error instanceof Error ? error.message : error}` }); } catch { /* uncertain claim is left intact */ }
  } finally { active.delete(requestId); }
}

while (!stopping) {
  for (const request of conn.db.taskRequest.sessionId.filter(sessionId)) {
    if (request.taskType === 'AUTONOMOUS_CUP_TRIAL' && request.status === 'QUEUED' && !active.has(request.requestId)) void execute(request.requestId);
  }
  await sleep(150);
}
conn.disconnect();
