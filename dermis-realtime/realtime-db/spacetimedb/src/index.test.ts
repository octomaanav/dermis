import { describe, expect, it } from 'vitest';
import { ConnectionId } from 'spacetimedb';
import { createModuleTestHarness, TestAuth } from 'spacetimedb/server/test-utils';
import spacetime, * as moduleExports from './index';

function auth(subject: string, id: bigint) {
  return TestAuth.fromJwtPayload(JSON.stringify({ iss: 'dermis-test', sub: subject }), new ConnectionId(id));
}

function createSession(ctx: Parameters<typeof moduleExports.createSession>[0], sessionId: string) {
  moduleExports.createSession(ctx, {
    sessionId, task: 'cup_transfer', version: 'test-v1', source: 'SYNTHETIC', controlMode: 'MANUAL',
    checkpointId: undefined, allowAnnotations: true,
  });
}

function publish(ctx: Parameters<typeof moduleExports.publishState>[0], sessionId: string, sequence: bigint) {
  moduleExports.publishState(ctx, {
    sessionId, streamGeneration: 1, sequence, sourceTimeMicros: sequence * 100_000n,
    clockDomain: 'TEST_MONOTONIC', robotJoints: [0, 0.1, 0.2], cupPosition: [0, 0, 0.2],
    cupOrientation: [0, 0, 0, 1], appliedDownwardLoad: 3, fingertipNormalLoads: [0.5, 0.6, 0.7],
    cupRelativeDisplacement: 0.002, outcome: 'IN_PROGRESS',
  });
}

function publishRaw(ctx: Parameters<typeof moduleExports.publishRawDataSamples>[0], sessionId: string, chunkSequence: bigint, firstSequence: bigint) {
  moduleExports.publishRawDataSamples(ctx, {
    chunkId: `${sessionId}.g1.c${chunkSequence}`, sessionId, streamGeneration: 1, chunkSequence, sampleCount: 2,
    firstSequence, lastSequence: firstSequence + 1n, firstSourceTimeMicros: firstSequence * 100_000n,
    lastSourceTimeMicros: (firstSequence + 1n) * 100_000n, encoding: 'GZIP_NDJSON_UTF8',
    uncompressedBytes: 48, payload: [31, 139, 8, 0],
  });
}

describe('Dermis shared-state module', () => {
  it('keeps one latest state and rejects duplicate or out-of-order sequences', () => {
    const test = createModuleTestHarness(spacetime, moduleExports);
    const alice = auth('alice', 1n);
    test.withReducerTx(alice, ctx => createSession(ctx, 'session-order'));
    test.withReducerTx(alice, ctx => publish(ctx, 'session-order', 1n));
    test.withReducerTx(alice, ctx => publish(ctx, 'session-order', 2n));

    expect([...test.db.latestState.iter()]).toHaveLength(1);
    expect(test.db.latestState.sessionId.find('session-order')?.sequence).toBe(2n);
    expect(() => test.withReducerTx(alice, ctx => publish(ctx, 'session-order', 2n)))
      .toThrow('sequence must strictly increase');
    expect(() => test.withReducerTx(alice, ctx => publish(ctx, 'session-order', 1n)))
      .toThrow('sequence must strictly increase');
    expect(test.db.latestState.sessionId.find('session-order')?.sequence).toBe(2n);
  });

  it('allows public-session observers to annotate but rejects their state and job writes', () => {
    const test = createModuleTestHarness(spacetime, moduleExports);
    const owner = auth('owner', 1n);
    const observer = auth('observer', 2n);
    test.withReducerTx(owner, ctx => createSession(ctx, 'session-owner'));
    test.withReducerTx(observer, ctx => moduleExports.addAnnotation(ctx, {
      sessionId: 'session-owner', episodeTimeMicros: 12n, label: 'slip', note: 'cup began to slide',
    }));
    expect([...test.db.annotation.iter()]).toHaveLength(1);
    expect(() => test.withReducerTx(observer, ctx => publish(ctx, 'session-owner', 1n)))
      .toThrow('only the session publisher');
    expect(() => test.withReducerTx(observer, ctx => moduleExports.updateJob(ctx, {
      jobId: 'observer-job', sessionId: 'session-owner', type: 'EVALUATION', status: 'RUNNING',
      datasetRef: undefined, checkpointRef: undefined, progress: 0.1, resultSummary: undefined,
    }))).toThrow('only the session publisher');
  });

  it('requires an explicit new session after a publisher restart and atomically upserts jobs', () => {
    const test = createModuleTestHarness(spacetime, moduleExports);
    const publisher = auth('publisher', 1n);
    test.withReducerTx(publisher, ctx => createSession(ctx, 'before-restart'));
    test.withReducerTx(publisher, ctx => publish(ctx, 'before-restart', 1n));
    expect(() => test.withReducerTx(publisher, ctx => moduleExports.publishState(ctx, {
      sessionId: 'before-restart', streamGeneration: 2, sequence: 1n, sourceTimeMicros: 1n,
      clockDomain: 'TEST_MONOTONIC', robotJoints: [], cupPosition: [0, 0, 0], cupOrientation: [0, 0, 0, 1],
      appliedDownwardLoad: 0, fingertipNormalLoads: [], cupRelativeDisplacement: 0, outcome: 'UNKNOWN',
    }))).toThrow('stream generation mismatch');

    test.withReducerTx(publisher, ctx => createSession(ctx, 'after-restart'));
    test.withReducerTx(publisher, ctx => publish(ctx, 'after-restart', 1n));
    test.withReducerTx(publisher, ctx => moduleExports.updateJob(ctx, {
      jobId: 'synthetic-evaluation', sessionId: 'after-restart', type: 'EVALUATION', status: 'QUEUED',
      datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: 0, resultSummary: undefined,
    }));
    test.withReducerTx(publisher, ctx => moduleExports.updateJob(ctx, {
      jobId: 'synthetic-evaluation', sessionId: 'after-restart', type: 'EVALUATION', status: 'SUCCEEDED',
      datasetRef: 'synthetic-fixture', checkpointRef: undefined, progress: 1,
      resultSummary: 'Synthetic telemetry only; no model ran.',
    }));
    expect([...test.db.job.iter()]).toHaveLength(1);
    expect(test.db.job.jobId.find('synthetic-evaluation')).toMatchObject({ status: 'SUCCEEDED', progress: 1 });
  });

  it('archives ordered compressed raw batches without allowing observer writes', () => {
    const test = createModuleTestHarness(spacetime, moduleExports);
    const publisher = auth('publisher', 1n);
    const observer = auth('observer', 2n);
    test.withReducerTx(publisher, ctx => createSession(ctx, 'raw-history'));
    test.withReducerTx(publisher, ctx => publishRaw(ctx, 'raw-history', 1n, 1n));
    test.withReducerTx(publisher, ctx => publishRaw(ctx, 'raw-history', 2n, 3n));
    expect([...test.db.rawDataSamples.sessionId.filter('raw-history')]).toHaveLength(2);
    expect(test.db.rawDataCursor.sessionId.find('raw-history')).toMatchObject({ lastChunkSequence: 2n, lastSequence: 4n });
    expect(() => test.withReducerTx(publisher, ctx => publishRaw(ctx, 'raw-history', 2n, 5n))).toThrow('raw data chunk already exists');
    expect(() => test.withReducerTx(observer, ctx => publishRaw(ctx, 'raw-history', 3n, 5n))).toThrow('only the session publisher');
  });
});
