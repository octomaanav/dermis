/** Queue one visible, authorized cup-policy trial for a shared room. */
import { randomUUID } from 'node:crypto';
import { connect, option, waitForTables } from './client.ts';

const sessionId = option('session-id');
const tokenPath = option('token-file', '.dermis-tokens/cup-room.token')!;
if (!sessionId) throw new Error('Usage: npm run cup:request -- --session-id SESSION_ID [--token-file PATH]');

const conn = await connect(tokenPath);
await waitForTables(conn);
const requestId = option('request-id', `judge-cup-${randomUUID()}`)!;
await conn.reducers.requestAutonomousTrial({ requestId, sessionId });
console.log(`REQUEST_ID=${requestId}`);
conn.disconnect();
