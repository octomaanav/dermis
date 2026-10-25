/** Dermis shared simulation-state module. No physics, training, or control lives here. */
import { schema, t, table, SenderError, type InferSchema, type ReducerCtx } from 'spacetimedb/server';

// These two tables belong to the original cloud starter that was already
// published under this database name.  Retain them during the migration so
// publishing Dermis never silently destroys an existing online row.
const legacyUser = table({ name: 'user', public: true }, {
  identity: t.identity().primaryKey(), name: t.option(t.string()), online: t.bool(),
});
const legacyMessage = table({ name: 'message', public: true }, {
  sender: t.identity(), sent: t.timestamp(), text: t.string(),
});

const session = table({ name: 'session', public: true }, {
  sessionId: t.string().primaryKey(),
  ownerIdentity: t.identity(), publisherIdentity: t.identity(),
  task: t.string(), version: t.string(), source: t.string(), controlMode: t.string(),
  checkpointId: t.option(t.string()), status: t.string(), streamGeneration: t.u32(),
  allowAnnotations: t.bool(), createdAt: t.timestamp(),
});

// The primary key keeps exactly one current telemetry row per session.
const latestState = table({ name: 'latest_state', public: true }, {
  sessionId: t.string().primaryKey(), streamGeneration: t.u32(), sequence: t.u64(),
  sourceTimeMicros: t.i64(), clockDomain: t.string(), serverReceiptTime: t.timestamp(),
  robotJoints: t.array(t.f64()), cupPosition: t.array(t.f64()), cupOrientation: t.array(t.f64()),
  appliedDownwardLoad: t.f64(), fingertipNormalLoads: t.array(t.f64()),
  cupRelativeDisplacement: t.f64(), outcome: t.string(),
});

// Historical/raw telemetry is intentionally stored as compressed batches, not
// one JSON object per database row. A public visualizer can subscribe to this
// table, filter by session, and inflate `payload` according to `encoding`.
// The cursor lets the reducer validate append order without scanning history.
const rawDataSamples = table({ name: 'raw_data_samples', public: true }, {
  chunkId: t.string().primaryKey(), sessionId: t.string().index('btree'), streamGeneration: t.u32(),
  chunkSequence: t.u64(), sampleCount: t.u16(), firstSequence: t.u64(), lastSequence: t.u64(),
  firstSourceTimeMicros: t.i64(), lastSourceTimeMicros: t.i64(), encoding: t.string(),
  uncompressedBytes: t.u32(), payload: t.array(t.u8()), createdAt: t.timestamp(),
});
const rawDataCursor = table({ name: 'raw_data_cursor' }, {
  sessionId: t.string().primaryKey(), streamGeneration: t.u32(), lastChunkSequence: t.u64(),
  lastSequence: t.u64(), lastSourceTimeMicros: t.i64(), updatedAt: t.timestamp(),
});

const annotation = table({ name: 'annotation', public: true }, {
  id: t.u64().primaryKey().autoInc(), sessionId: t.string().index('btree'),
  episodeTimeMicros: t.i64(), authorIdentity: t.identity(), label: t.string(), note: t.string(),
  createdAt: t.timestamp(),
});

const job = table({ name: 'job', public: true }, {
  jobId: t.string().primaryKey(), sessionId: t.string().index('btree'), type: t.string(), status: t.string(),
  datasetRef: t.option(t.string()), checkpointRef: t.option(t.string()), progress: t.option(t.f64()),
  resultSummary: t.option(t.string()), updatedAt: t.timestamp(),
});

// Task requests are coordination records. A local worker must explicitly claim
// one before it may advance it; the database never runs robot code itself.
const taskRequest = table({ name: 'task_request', public: true }, {
  requestId: t.string().primaryKey(), sessionId: t.string().index('btree'), taskType: t.string(),
  status: t.string(), requestedBy: t.identity(), requestedAt: t.timestamp(),
  claimedBy: t.option(t.identity()), updatedAt: t.timestamp(), resultSummary: t.option(t.string()),
});

// This is intentionally append-only and bounded by demo task volume, unlike
// telemetry. It gives observers an auditable lifecycle rather than a decorative badge.
const taskEvent = table({ name: 'task_event', public: true }, {
  id: t.u64().primaryKey().autoInc(), requestId: t.string().index('btree'), sessionId: t.string().index('btree'),
  status: t.string(), actorIdentity: t.identity(), detail: t.string(), createdAt: t.timestamp(),
});

// Connection lifecycle records power the engineering panel's actual connected-viewer count.
const viewer = table({ name: 'viewer', public: true }, {
  connectionId: t.connectionId().primaryKey(), identity: t.identity(), connectedAt: t.timestamp(),
});

const spacetimedb = schema({ legacyUser, legacyMessage, session, latestState, rawDataSamples, rawDataCursor, annotation, job, taskRequest, taskEvent, viewer });
export default spacetimedb;
type Ctx = ReducerCtx<InferSchema<typeof spacetimedb>>;

const SOURCES = ['REAL_GLOVE', 'SCRIPTED_SIMULATION', 'SYNTHETIC', 'REPLAY'];
const CONTROL_MODES = ['MANUAL', 'ASSISTED', 'AUTONOMOUS', 'REPLAY'];
const OUTCOMES = ['UNKNOWN', 'IN_PROGRESS', 'SUCCESS', 'DROPPED', 'FAILED'];
const JOB_TYPES = ['TRAINING', 'EVALUATION', 'EXPORT', 'INGESTION'];
const JOB_STATUSES = ['QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED'];
const TASK_TYPES = ['AUTONOMOUS_CUP_TRIAL', 'AUTONOMOUS_ASSISTIVE_GRASP'];
const TASK_STATUSES = ['QUEUED', 'CLAIMED', 'RUNNING', 'SUCCEEDED', 'FAILED'];

function bounded(value: string, field: string, max: number, allowEmpty = false) {
  if (typeof value !== 'string' || (!allowEmpty && value.trim().length === 0) || value.length > max)
    throw new SenderError(`${field} must be ${allowEmpty ? 'at most' : 'between 1 and'} ${max} characters`);
}
function enumValue(value: string, field: string, values: readonly string[]) {
  if (!values.includes(value)) throw new SenderError(`${field} has an unsupported value`);
}
function validateSessionId(value: string) {
  bounded(value, 'sessionId', 96);
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value))
    throw new SenderError('sessionId may contain only letters, numbers, dot, underscore, and hyphen');
}
function validateRequestId(value: string) {
  bounded(value, 'requestId', 96);
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value))
    throw new SenderError('requestId may contain only letters, numbers, dot, underscore, and hyphen');
}
function finite(value: number, field: string) {
  if (!Number.isFinite(value)) throw new SenderError(`${field} must be finite`);
}
function finiteArray(value: number[], field: string, min: number, max: number, nonNegative = false) {
  if (value.length < min || value.length > max)
    throw new SenderError(`${field} must contain ${min === max ? min : `${min}..${max}`} values`);
  value.forEach((entry, index) => {
    finite(entry, `${field}[${index}]`);
    if (nonNegative && entry < 0) throw new SenderError(`${field}[${index}] must be non-negative`);
  });
}
function optionalBounded(value: string | undefined, field: string, max: number) {
  if (value !== undefined) bounded(value, field, max);
}
function requireSession(ctx: Ctx, requestedSessionId: string) {
  validateSessionId(requestedSessionId);
  const row = ctx.db.session.sessionId.find(requestedSessionId);
  if (!row) throw new SenderError('session does not exist');
  return row;
}
function requirePublisher(ctx: Ctx, requestedSessionId: string) {
  const row = requireSession(ctx, requestedSessionId);
  if (!row.publisherIdentity.isEqual(ctx.sender))
    throw new SenderError('only the session publisher may perform this update');
  return row;
}
function appendTaskEvent(ctx: Ctx, requestId: string, sessionId: string, status: string, detail: string) {
  ctx.db.taskEvent.insert({ id: 0n, requestId, sessionId, status, actorIdentity: ctx.sender, detail, createdAt: ctx.timestamp });
}

export const createSession = spacetimedb.reducer({
  sessionId: t.string(), task: t.string(), version: t.string(), source: t.string(), controlMode: t.string(),
  checkpointId: t.option(t.string()), allowAnnotations: t.bool(),
}, (ctx, args) => {
  validateSessionId(args.sessionId); bounded(args.task, 'task', 160); bounded(args.version, 'version', 80);
  enumValue(args.source, 'source', SOURCES); enumValue(args.controlMode, 'controlMode', CONTROL_MODES);
  optionalBounded(args.checkpointId, 'checkpointId', 256);
  if (ctx.db.session.sessionId.find(args.sessionId)) throw new SenderError('sessionId already exists');
  ctx.db.session.insert({
    sessionId: args.sessionId, task: args.task, version: args.version, source: args.source,
    controlMode: args.controlMode, checkpointId: args.checkpointId, allowAnnotations: args.allowAnnotations,
    ownerIdentity: ctx.sender, publisherIdentity: ctx.sender, status: 'ACTIVE', streamGeneration: 1,
    createdAt: ctx.timestamp,
  });
});

export const publishState = spacetimedb.reducer({
  sessionId: t.string(), streamGeneration: t.u32(), sequence: t.u64(), sourceTimeMicros: t.i64(),
  clockDomain: t.string(), robotJoints: t.array(t.f64()), cupPosition: t.array(t.f64()),
  cupOrientation: t.array(t.f64()), appliedDownwardLoad: t.f64(), fingertipNormalLoads: t.array(t.f64()),
  cupRelativeDisplacement: t.f64(), outcome: t.string(),
}, (ctx, args) => {
  const currentSession = requirePublisher(ctx, args.sessionId);
  if (currentSession.status !== 'ACTIVE') throw new SenderError('session is not active');
  if (args.streamGeneration !== currentSession.streamGeneration)
    throw new SenderError('stream generation mismatch; create a new session after a publisher restart');
  bounded(args.clockDomain, 'clockDomain', 64);
  finiteArray(args.robotJoints, 'robotJoints', 0, 64);
  finiteArray(args.cupPosition, 'cupPosition', 3, 3); finiteArray(args.cupOrientation, 'cupOrientation', 4, 4);
  finite(args.appliedDownwardLoad, 'appliedDownwardLoad');
  if (args.appliedDownwardLoad < 0) throw new SenderError('appliedDownwardLoad must be non-negative');
  finiteArray(args.fingertipNormalLoads, 'fingertipNormalLoads', 0, 16, true);
  finite(args.cupRelativeDisplacement, 'cupRelativeDisplacement');
  if (args.cupRelativeDisplacement < 0) throw new SenderError('cupRelativeDisplacement must be non-negative');
  enumValue(args.outcome, 'outcome', OUTCOMES);
  const current = ctx.db.latestState.sessionId.find(args.sessionId);
  if (current) {
    if (current.streamGeneration !== args.streamGeneration || args.sequence <= current.sequence)
      throw new SenderError('sequence must strictly increase within the current stream generation');
    ctx.db.latestState.sessionId.update({ ...args, serverReceiptTime: ctx.timestamp });
  } else ctx.db.latestState.insert({ ...args, serverReceiptTime: ctx.timestamp });
});

export const publishRawDataSamples = spacetimedb.reducer({
  chunkId: t.string(), sessionId: t.string(), streamGeneration: t.u32(), chunkSequence: t.u64(), sampleCount: t.u16(),
  firstSequence: t.u64(), lastSequence: t.u64(), firstSourceTimeMicros: t.i64(), lastSourceTimeMicros: t.i64(),
  encoding: t.string(), uncompressedBytes: t.u32(), payload: t.array(t.u8()),
}, (ctx, args) => {
  const currentSession = requirePublisher(ctx, args.sessionId);
  if (currentSession.status !== 'ACTIVE') throw new SenderError('session is not active');
  if (args.streamGeneration !== currentSession.streamGeneration)
    throw new SenderError('stream generation mismatch; create a new session after a publisher restart');
  bounded(args.chunkId, 'chunkId', 160);
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(args.chunkId)) throw new SenderError('chunkId contains unsupported characters');
  if (args.chunkId !== `${args.sessionId}.g${args.streamGeneration}.c${args.chunkSequence}`)
    throw new SenderError('chunkId must be derived from session, stream generation, and chunk sequence');
  if (args.sampleCount < 1 || args.sampleCount > 256) throw new SenderError('sampleCount must be between 1 and 256');
  if (args.firstSequence > args.lastSequence) throw new SenderError('raw sample sequence range is invalid');
  if (args.firstSourceTimeMicros > args.lastSourceTimeMicros) throw new SenderError('raw sample time range is invalid');
  if (args.encoding !== 'GZIP_NDJSON_UTF8') throw new SenderError('only GZIP_NDJSON_UTF8 raw batches are supported');
  if (args.payload.length < 1 || args.payload.length > 262_144) throw new SenderError('compressed raw payload must be 1..262144 bytes');
  if (args.uncompressedBytes < args.payload.length || args.uncompressedBytes > 4_194_304)
    throw new SenderError('uncompressed raw payload must be within 1..4194304 bytes and no smaller than its compressed payload');
  if (ctx.db.rawDataSamples.chunkId.find(args.chunkId)) throw new SenderError('raw data chunk already exists');
  const cursor = ctx.db.rawDataCursor.sessionId.find(args.sessionId);
  if (cursor) {
    if (cursor.streamGeneration !== args.streamGeneration) throw new SenderError('raw data stream generation mismatch');
    if (args.chunkSequence <= cursor.lastChunkSequence || args.firstSequence <= cursor.lastSequence || args.firstSourceTimeMicros < cursor.lastSourceTimeMicros)
      throw new SenderError('raw data chunks must append in increasing chunk, sequence, and source-time order');
  }
  const row = { ...args, createdAt: ctx.timestamp };
  ctx.db.rawDataSamples.insert(row);
  const nextCursor = { sessionId: args.sessionId, streamGeneration: args.streamGeneration,
    lastChunkSequence: args.chunkSequence, lastSequence: args.lastSequence,
    lastSourceTimeMicros: args.lastSourceTimeMicros, updatedAt: ctx.timestamp };
  if (cursor) ctx.db.rawDataCursor.sessionId.update(nextCursor); else ctx.db.rawDataCursor.insert(nextCursor);
});

export const addAnnotation = spacetimedb.reducer(
  { sessionId: t.string(), episodeTimeMicros: t.i64(), label: t.string(), note: t.string() },
  (ctx, args) => {
    const currentSession = requireSession(ctx, args.sessionId);
    const ownerOrPublisher = currentSession.ownerIdentity.isEqual(ctx.sender) || currentSession.publisherIdentity.isEqual(ctx.sender);
    if (!currentSession.allowAnnotations && !ownerOrPublisher)
      throw new SenderError('observers are not permitted to annotate this session');
    bounded(args.label, 'label', 80); bounded(args.note, 'note', 2_000, true);
    ctx.db.annotation.insert({ id: 0n, ...args, authorIdentity: ctx.sender, createdAt: ctx.timestamp });
  }
);

export const updateJob = spacetimedb.reducer({
  jobId: t.string(), sessionId: t.string(), type: t.string(), status: t.string(),
  datasetRef: t.option(t.string()), checkpointRef: t.option(t.string()), progress: t.option(t.f64()),
  resultSummary: t.option(t.string()),
}, (ctx, args) => {
  requirePublisher(ctx, args.sessionId); bounded(args.jobId, 'jobId', 120);
  enumValue(args.type, 'type', JOB_TYPES); enumValue(args.status, 'status', JOB_STATUSES);
  optionalBounded(args.datasetRef, 'datasetRef', 512); optionalBounded(args.checkpointRef, 'checkpointRef', 512);
  optionalBounded(args.resultSummary, 'resultSummary', 4_000);
  if (args.progress !== undefined && (!Number.isFinite(args.progress) || args.progress < 0 || args.progress > 1))
    throw new SenderError('progress must be finite and between 0 and 1');
  const current = ctx.db.job.jobId.find(args.jobId);
  if (current && current.sessionId !== args.sessionId) throw new SenderError('jobId already belongs to another session');
  const row = {
    jobId: args.jobId, sessionId: args.sessionId, type: args.type, status: args.status,
    datasetRef: args.datasetRef, checkpointRef: args.checkpointRef, progress: args.progress,
    resultSummary: args.resultSummary, updatedAt: ctx.timestamp,
  };
  if (current) ctx.db.job.jobId.update(row); else ctx.db.job.insert(row);
});

function requestAutonomousTask(ctx: Ctx, args: { requestId: string; sessionId: string }, taskType: string, detail: string) {
    if (!TASK_TYPES.includes(taskType)) throw new SenderError('unsupported task type');
    const currentSession = requireSession(ctx, args.sessionId); validateRequestId(args.requestId);
    if (taskType === 'AUTONOMOUS_CUP_TRIAL' && currentSession.task !== 'fragile_cup')
      throw new SenderError('cup trials require a fragile_cup session');
    if (taskType === 'AUTONOMOUS_ASSISTIVE_GRASP' && currentSession.task !== 'assist_hand')
      throw new SenderError('assistive grasps require an assist_hand session');
    if (ctx.db.taskRequest.requestId.find(args.requestId)) throw new SenderError('requestId already exists');
    const pending = [...ctx.db.taskRequest.sessionId.filter(args.sessionId)]
      .filter(request => ['QUEUED', 'CLAIMED', 'RUNNING'].includes(request.status)).length;
    if (pending >= 3) throw new SenderError('this room already has the maximum of three pending challenges');
    const row = {
      requestId: args.requestId, sessionId: args.sessionId, taskType, status: 'QUEUED',
      requestedBy: ctx.sender, requestedAt: ctx.timestamp, claimedBy: undefined, updatedAt: ctx.timestamp,
      resultSummary: undefined,
    };
    ctx.db.taskRequest.insert(row);
    appendTaskEvent(ctx, args.requestId, args.sessionId, 'QUEUED', detail);
}

export const requestAutonomousTrial = spacetimedb.reducer(
  { requestId: t.string(), sessionId: t.string() },
  (ctx, args) => {
    requestAutonomousTask(ctx, args, 'AUTONOMOUS_CUP_TRIAL', 'Observer requested an autonomous cup trial.');
  },
);

export const requestAutonomousAssistiveGrasp = spacetimedb.reducer(
  { requestId: t.string(), sessionId: t.string() },
  (ctx, args) => {
    requestAutonomousTask(ctx, args, 'AUTONOMOUS_ASSISTIVE_GRASP', 'Observer requested an autonomous three-finger grasp.');
  },
);

export const claimTask = spacetimedb.reducer({ requestId: t.string() }, (ctx, args) => {
  validateRequestId(args.requestId);
  const current = ctx.db.taskRequest.requestId.find(args.requestId);
  if (!current) throw new SenderError('task request does not exist');
  requirePublisher(ctx, current.sessionId);
  if (current.status !== 'QUEUED') throw new SenderError('task may only be claimed from QUEUED');
  const row = { ...current, status: 'CLAIMED', claimedBy: ctx.sender, updatedAt: ctx.timestamp };
  ctx.db.taskRequest.requestId.update(row);
  appendTaskEvent(ctx, current.requestId, current.sessionId, 'CLAIMED', 'Authorized local worker claimed the request.');
});

export const updateTask = spacetimedb.reducer(
  { requestId: t.string(), status: t.string(), resultSummary: t.option(t.string()) },
  (ctx, args) => {
    validateRequestId(args.requestId); enumValue(args.status, 'status', TASK_STATUSES);
    optionalBounded(args.resultSummary, 'resultSummary', 4_000);
    const current = ctx.db.taskRequest.requestId.find(args.requestId);
    if (!current) throw new SenderError('task request does not exist');
    requirePublisher(ctx, current.sessionId);
    if (!current.claimedBy?.isEqual(ctx.sender)) throw new SenderError('only the worker that claimed this task may update it');
    const validNext = (current.status === 'CLAIMED' && ['RUNNING', 'FAILED'].includes(args.status))
      || (current.status === 'RUNNING' && ['SUCCEEDED', 'FAILED'].includes(args.status));
    if (!validNext) throw new SenderError(`invalid task transition from ${current.status} to ${args.status}`);
    const row = { ...current, status: args.status, resultSummary: args.resultSummary, updatedAt: ctx.timestamp };
    ctx.db.taskRequest.requestId.update(row);
    appendTaskEvent(ctx, current.requestId, current.sessionId, args.status,
      args.resultSummary ?? `Worker set task state to ${args.status}.`);
  },
);

// A restarted publisher must create a fresh session ID. Reducers are transactional,
// so a failed validation rolls back the entire table update.
export const init = spacetimedb.init(_ctx => {});

export const clientConnected = spacetimedb.clientConnected(ctx => {
  if (ctx.connectionId) ctx.db.viewer.insert({ connectionId: ctx.connectionId, identity: ctx.sender, connectedAt: ctx.timestamp });
});
export const clientDisconnected = spacetimedb.clientDisconnected(ctx => {
  if (ctx.connectionId) ctx.db.viewer.connectionId.delete(ctx.connectionId);
});
