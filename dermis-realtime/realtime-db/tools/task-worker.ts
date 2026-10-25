/**
 * Local task coordinator for the two-device demo.
 *
 * This process is deliberately separate from the browser and SpacetimeDB
 * module. It claims a request transactionally, then invokes an explicit local
 * runner. The runner is responsible for real simulator control and telemetry;
 * this program reports only the runner's actual exit result.
 */
import { spawn } from 'node:child_process';
import { connect, option, sleep, waitForTables } from './client.ts';

const tokenPath = option('token-file', '.dermis-tokens/synthetic-publisher.token')!;
const sessionId = option('session-id');
const command = option('command');
if (!sessionId || !command) {
  console.error('Usage: npm run worker -- --session-id SESSION_ID --command /absolute/path/to/local-simulator-runner [--token-file PATH]');
  console.error('The runner receives SESSION_ID and REQUEST_ID as its two arguments.');
  process.exit(2);
}

let stopping = false;
const active = new Set<string>();
process.on('SIGINT', () => { stopping = true; });

const conn = await connect(tokenPath);
await waitForTables(conn);
console.log(`Task worker ready for ${sessionId}. It will only run ${command} after it claims a queued request.`);

async function execute(requestId: string) {
  active.add(requestId);
  try {
    await conn.reducers.claimTask({ requestId });
    await conn.reducers.updateTask({ requestId, status: 'RUNNING', resultSummary: undefined });
    const result = await new Promise<{ code: number | null; error?: string }>(resolve => {
      const child = spawn(command!, [sessionId!, requestId], { stdio: 'inherit' });
      child.once('error', error => resolve({ code: null, error: error.message }));
      child.once('close', code => resolve({ code }));
    });
    if (result.code === 0) {
      await conn.reducers.updateTask({ requestId, status: 'SUCCEEDED', resultSummary: 'Local runner exited successfully; inspect subscribed simulation state for the actual outcome.' });
    } else {
      await conn.reducers.updateTask({ requestId, status: 'FAILED', resultSummary: result.error ?? `Local runner exited with code ${result.code}.` });
    }
  } catch (error) {
    // Another worker may have won the atomic claim. Do not overwrite its status.
    console.error(`task ${requestId}: ${error instanceof Error ? error.message : error}`);
  } finally {
    active.delete(requestId);
  }
}

while (!stopping) {
  for (const request of conn.db.taskRequest.sessionId.filter(sessionId!)) {
    if (request.status === 'QUEUED' && !active.has(request.requestId)) void execute(request.requestId);
  }
  await sleep(150);
}
conn.disconnect();
