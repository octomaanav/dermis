/** Create a shared room for the actual learned three-finger assistive-hand simulation. */
import { randomUUID } from 'node:crypto';
import { connect, option, waitForTables } from './client.ts';

const tokenPath = option('token-file', '.dermis-tokens/assist-room.token')!;
const sessionId = option('session-id', `assist-room-${randomUUID()}`)!;
const checkpoint = 'runs/assist/candidates/fallback_scripted/checkpoint';
const conn = await connect(tokenPath);
await waitForTables(conn);
await conn.reducers.createSession({ sessionId, task: 'assist_hand', version: 'assist-hand-fallback-scripted-v1',
  source: 'SCRIPTED_SIMULATION', controlMode: 'AUTONOMOUS', checkpointId: checkpoint, allowAnnotations: true });
console.log(`SESSION_ID=${sessionId}`);
console.log(`Room is ready for bounded three-finger assistive-grasp requests using ${checkpoint}.`);
conn.disconnect();
