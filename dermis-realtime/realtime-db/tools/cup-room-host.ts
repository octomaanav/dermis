/** Create a room for actual local fragile_cup neural-policy challenges. */
import { randomUUID } from 'node:crypto';
import { connect, option, waitForTables } from './client.ts';

const tokenPath = option('token-file', '.dermis-tokens/cup-room.token')!;
const sessionId = option('session-id', `cup-room-${randomUUID()}`)!;
const checkpoint = 'runs/cup/autonomous_demo/cup_autonomous_demo_v1';
const conn = await connect(tokenPath);
await waitForTables(conn);
await conn.reducers.createSession({ sessionId, task: 'fragile_cup', version: 'cup-autonomous-demo-v1',
  source: 'SCRIPTED_SIMULATION', controlMode: 'AUTONOMOUS', checkpointId: checkpoint, allowAnnotations: true });
console.log(`SESSION_ID=${sessionId}`);
console.log(`Room is ready for bounded cup-trial requests using ${checkpoint}. Keep the cup challenge worker running with the same token.`);
conn.disconnect();
