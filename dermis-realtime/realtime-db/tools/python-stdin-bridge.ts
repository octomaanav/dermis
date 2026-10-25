/**
 * Authenticated stdin bridge for a local Python visualizer. It intentionally
 * has no HTTP listener: Python remains isolated from database credentials.
 */
import readline from 'node:readline';
import { connect, option } from './client.ts';

type Snapshot = {
  sequence: number; source_time_us: number; clock_domain: string; robot_joints: number[];
  cup_position: number[]; cup_orientation: number[]; applied_downward_load: number;
  fingertip_normal_loads: number[]; cup_relative_displacement: number; outcome: string;
};

const sessionId = option('session-id');
if (!sessionId) throw new Error('--session-id is required; create the session with the same secure publisher token first');
const generation = Number(option('stream-generation', '1'));
const tokenPath = option('token-file', '.dermis-tokens/python-bridge.token')!;
let online = false;
let latest: Snapshot | undefined;
let publishing = false;

function finite(value: unknown, field: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${field} must be finite`);
  return value;
}
function array(value: unknown, field: string, min: number, max: number) {
  if (!Array.isArray(value) || value.length < min || value.length > max) throw new Error(`${field} length must be ${min}..${max}`);
  return value.map((entry, index) => finite(entry, `${field}[${index}]`));
}
function parse(line: string): Snapshot {
  if (Buffer.byteLength(line) > 65_536) throw new Error('snapshot exceeds 64 KiB');
  const raw = JSON.parse(line) as Record<string, unknown>;
  const sequence = finite(raw.sequence, 'sequence');
  const sourceTime = finite(raw.source_time_us, 'source_time_us');
  if (!Number.isSafeInteger(sequence) || sequence < 0 || !Number.isSafeInteger(sourceTime)) throw new Error('sequence and source_time_us must be safe integers');
  if (typeof raw.clock_domain !== 'string' || raw.clock_domain.length === 0 || raw.clock_domain.length > 64) throw new Error('clock_domain is invalid');
  if (typeof raw.outcome !== 'string') throw new Error('outcome is required');
  return { sequence, source_time_us: sourceTime, clock_domain: raw.clock_domain,
    robot_joints: array(raw.robot_joints, 'robot_joints', 0, 64), cup_position: array(raw.cup_position, 'cup_position', 3, 3),
    cup_orientation: array(raw.cup_orientation, 'cup_orientation', 4, 4), applied_downward_load: finite(raw.applied_downward_load, 'applied_downward_load'),
    fingertip_normal_loads: array(raw.fingertip_normal_loads, 'fingertip_normal_loads', 0, 16),
    cup_relative_displacement: finite(raw.cup_relative_displacement, 'cup_relative_displacement'), outcome: raw.outcome };
}

const conn = await connect(tokenPath, state => { online = state; if (state) void flush(); });
async function flush() {
  if (publishing || !online || !latest) return;
  publishing = true;
  const snapshot = latest; latest = undefined;
  try {
    await conn.reducers.publishState({ sessionId, streamGeneration: generation, sequence: BigInt(snapshot.sequence),
      sourceTimeMicros: BigInt(snapshot.source_time_us), clockDomain: snapshot.clock_domain, robotJoints: snapshot.robot_joints,
      cupPosition: snapshot.cup_position, cupOrientation: snapshot.cup_orientation, appliedDownwardLoad: snapshot.applied_downward_load,
      fingertipNormalLoads: snapshot.fingertip_normal_loads, cupRelativeDisplacement: snapshot.cup_relative_displacement, outcome: snapshot.outcome });
  } catch (error) {
    // Do not replay an uncertain reducer call after reconnect: Python must send a new increasing sequence.
    console.error(`dropped bridge snapshot ${snapshot.sequence}: ${error instanceof Error ? error.message : String(error)}`);
  } finally { publishing = false; if (latest) void flush(); }
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on('line', line => {
  try { latest = parse(line); void flush(); }
  catch (error) { console.error(`ignored stdin snapshot: ${error instanceof Error ? error.message : String(error)}`); }
});
input.on('close', () => { console.error('stdin closed; bridge stopping'); conn.disconnect(); });
