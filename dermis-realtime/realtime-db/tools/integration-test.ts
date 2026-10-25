import { randomUUID } from 'node:crypto';
import { connect, sleep, waitForTables } from './client.ts';

async function eventually(check: () => boolean, message: string) {
  for (let i = 0; i < 50; i += 1) { if (check()) return; await sleep(100); }
  throw new Error(message);
}
function state(sessionId: string, sequence: bigint) {
  return { sessionId, streamGeneration: 1, sequence, sourceTimeMicros: sequence * 100_000n,
    clockDomain: 'INTEGRATION_TEST', robotJoints: [0, .2], cupPosition: [.45, 0, .18],
    cupOrientation: [0, 0, 0, 1], appliedDownwardLoad: 3, fingertipNormalLoads: [1, 1, 1],
    cupRelativeDisplacement: .001, outcome: 'IN_PROGRESS' };
}

const nonce = randomUUID();
const owner = await connect(`.dermis-tokens/integration-owner-${nonce}.token`);
const observerA = await connect(`.dermis-tokens/integration-observer-a-${nonce}.token`);
const observerB = await connect(`.dermis-tokens/integration-observer-b-${nonce}.token`);
await Promise.all([waitForTables(owner), waitForTables(observerA), waitForTables(observerB)]);
const sessionId = `integration-${nonce}`;
try {
  await owner.reducers.createSession({ sessionId, task: 'fragile_cup', version: 'integration-v1', source: 'SYNTHETIC',
    controlMode: 'MANUAL', checkpointId: undefined, allowAnnotations: true });
  await owner.reducers.publishState(state(sessionId, 1n));
  await eventually(() => observerA.db.latestState.sessionId.find(sessionId)?.sequence === 1n
    && observerB.db.latestState.sessionId.find(sessionId)?.sequence === 1n, 'two observers did not receive state');
  await observerA.reducers.addAnnotation({ sessionId, episodeTimeMicros: 100_000n, label: 'shared', note: 'visible to both' });
  await eventually(() => [...observerA.db.annotation.sessionId.filter(sessionId)].length === 1
    && [...observerB.db.annotation.sessionId.filter(sessionId)].length === 1, 'annotation was not shared');
  await owner.reducers.publishState(state(sessionId, 2n));
  const requestId = `trial-${nonce}`;
  await observerB.reducers.requestAutonomousTrial({ requestId, sessionId });
  await eventually(() => observerA.db.taskRequest.requestId.find(requestId)?.status === 'QUEUED', 'task request did not fan out');
  let unauthorizedClaim = false; try { await observerA.reducers.claimTask({ requestId }); } catch { unauthorizedClaim = true; }
  if (!unauthorizedClaim) throw new Error('unauthorized worker claim was accepted');
  await owner.reducers.claimTask({ requestId });
  await owner.reducers.updateTask({ requestId, status: 'RUNNING', resultSummary: undefined });
  await owner.reducers.updateTask({ requestId, status: 'SUCCEEDED', resultSummary: 'integration worker completed' });
  await eventually(() => observerA.db.taskRequest.requestId.find(requestId)?.status === 'SUCCEEDED'
    && [...observerB.db.taskEvent.requestId.filter(requestId)].length === 4, 'task lifecycle did not fan out');
  let unauthorized = false; try { await observerB.reducers.publishState(state(sessionId, 3n)); } catch { unauthorized = true; }
  if (!unauthorized) throw new Error('unauthorized publisher was accepted');
  let duplicate = false; try { await owner.reducers.publishState(state(sessionId, 2n)); } catch { duplicate = true; }
  if (!duplicate) throw new Error('duplicate sequence was accepted');
  let outOfOrder = false; try { await owner.reducers.publishState(state(sessionId, 1n)); } catch { outOfOrder = true; }
  if (!outOfOrder) throw new Error('out-of-order sequence was accepted');

  const refreshed = await connect(`.dermis-tokens/integration-refresh-${nonce}.token`);
  await waitForTables(refreshed);
  await eventually(() => refreshed.db.latestState.sessionId.find(sessionId)?.sequence === 2n, 'fresh subscription did not restore current state');
  refreshed.disconnect();

  let resetRejected = false; try { await owner.reducers.publishState({ ...state(sessionId, 1n), streamGeneration: 2 }); } catch { resetRejected = true; }
  if (!resetRejected) throw new Error('publisher restart reset was accepted');
  console.log('PASS: two-client subscription, annotation/task fan-out, transactional worker claim, refresh restore, authorization, ordering, and explicit restart checks.');
} finally { owner.disconnect(); observerA.disconnect(); observerB.disconnect(); }
