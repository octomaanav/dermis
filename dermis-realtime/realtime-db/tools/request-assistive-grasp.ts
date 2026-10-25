/** Queue one shared autonomous three-finger grasp request. */
import { randomUUID } from 'node:crypto';
import { connect, option, waitForTables } from './client.ts';

const sessionId = option('session-id');
const tokenPath = option('token-file', '.dermis-tokens/assist-observer.token')!;
if (!sessionId) throw new Error('Usage: npm run assist:request -- --session-id SESSION_ID [--token-file PATH]');
const conn = await connect(tokenPath);
await waitForTables(conn);
const requestId = option('request-id', `judge-assist-${randomUUID()}`)!;
await conn.reducers.requestAutonomousAssistiveGrasp({ requestId, sessionId });
console.log(`REQUEST_ID=${requestId}`);
conn.disconnect();
