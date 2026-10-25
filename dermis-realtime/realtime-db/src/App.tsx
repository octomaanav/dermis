import { FormEvent, useEffect, useMemo, useState } from 'react';
import './App.css';
import { reducers, tables } from './module_bindings';
import { useReducer, useSpacetimeDB, useTable } from 'spacetimedb/react';
import type { LatestState } from './module_bindings/types';
import RobotScene from './RobotScene';
import JoinQr from './JoinQr';

const STALE_AFTER_MS = 1_500;

function shareOriginFrom(value: string) {
  const trimmed = value.trim().replace(/\/+$/, '');
  if (!trimmed) return '';
  return /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
}

export function isStateStale(state: LatestState | undefined, now = Date.now()) {
  if (!state) return true;
  return now - state.serverReceiptTime.toDate().getTime() > STALE_AFTER_MS;
}

function when(value: { toDate(): Date }) {
  return value.toDate().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function sessionName(session: { task: string; source: string; createdAt: { toDate(): Date } }) {
  const task = session.task === 'cup_load_demo' ? 'Cup pickup teaching demo'
    : session.task === 'fragile_cup' ? 'Fragile cup pickup' : session.task.replace(/[_-]/g, ' ');
  const source = session.source === 'REAL_GLOVE' ? 'glove'
    : session.source === 'SCRIPTED_SIMULATION' ? 'live simulator'
      : session.source === 'REPLAY' ? 'recorded replay' : 'synthetic fixture';
  return `${task} · ${source} · ${when(session.createdAt)}`;
}

function sourceLabel(source: string) {
  return ({ REAL_GLOVE: 'REAL GLOVE', SCRIPTED_SIMULATION: 'SIMULATION — LEARNED POLICY',
    SYNTHETIC: 'SYNTHETIC PREVIEW', REPLAY: 'RECORDED PLAYBACK' } as Record<string, string>)[source] ?? source;
}

function LiveChart({ history }: { history: Array<{ sequence: bigint; value: number; receivedAt: number }> }) {
  if (history.length < 2) return <div className="chart-empty"><b>{history[0] ? `${history[0].value.toFixed(2)} N` : '—'}</b><span>Waiting for the next shared frame…</span></div>;
  const max = Math.max(1, ...history.map(point => point.value));
  const points = history.map((point, index) => {
    const x = (index / Math.max(history.length - 1, 1)) * 100;
    const y = 92 - (point.value / max) * 84;
    return `${x},${y}`;
  }).join(' ');
  return <svg className="chart" viewBox="0 0 100 100" role="img" aria-label="live applied-load chart">
    <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />
  </svg>;
}

export default function App() {
  const { identity, isActive } = useSpacetimeDB();
  const addAnnotation = useReducer(reducers.addAnnotation);
  const requestAutonomousTrial = useReducer(reducers.requestAutonomousTrial);
  const requestAutonomousAssistiveGrasp = useReducer(reducers.requestAutonomousAssistiveGrasp);
  const [sessions, sessionsReady] = useTable(tables.session);
  const [states, statesReady] = useTable(tables.latestState);
  const [annotations, annotationsReady] = useTable(tables.annotation);
  const [jobs, jobsReady] = useTable(tables.job);
  const [taskRequests, taskRequestsReady] = useTable(tables.taskRequest);
  const [taskEvents, taskEventsReady] = useTable(tables.taskEvent);
  const [viewers, viewersReady] = useTable(tables.viewer);
  // Fresh viewers should open the newest demo, not a stale session remembered from an old run.
  const roomIdFromPath = typeof window === 'undefined' ? '' : decodeURIComponent(window.location.pathname.match(/^\/room\/([^/]+)/)?.[1] ?? '');
  const phoneRoom = typeof window !== 'undefined' && window.location.pathname.startsWith('/room/');
  const [selectedSessionId, setSelectedSessionId] = useState(roomIdFromPath);
  const [label, setLabel] = useState('observation');
  const [note, setNote] = useState('');
  const [annotationError, setAnnotationError] = useState('');
  const [taskError, setTaskError] = useState('');
  const [now, setNow] = useState(Date.now());
  const [history, setHistory] = useState<Array<{ sequence: bigint; value: number; receivedAt: number }>>([]);
  const initialShareOrigin = typeof window === 'undefined' ? '' : (
    localStorage.getItem('dermis.room-share-origin') ??
    import.meta.env.VITE_PUBLIC_ROOM_ORIGIN ??
    (['localhost', '127.0.0.1'].includes(window.location.hostname) ? '' : window.location.origin)
  );
  const [shareOrigin, setShareOrigin] = useState(initialShareOrigin);
  const [copied, setCopied] = useState(false);

  const orderedSessions = useMemo(() => {
    const nonIntegration = sessions.filter(session => session.task !== 'integration');
    // Synthetic fixtures are useful for development but should never be the default judge demo.
    const liveExperiments = nonIntegration.filter(session => session.source !== 'SYNTHETIC');
    return [...(liveExperiments.length ? liveExperiments : nonIntegration)].sort((a, b) =>
      b.createdAt.toDate().getTime() - a.createdAt.toDate().getTime());
  }, [sessions]);
  useEffect(() => {
    if (selectedSessionId && sessions.some(session => session.sessionId === selectedSessionId)) return;
    if (orderedSessions[0]) setSelectedSessionId(orderedSessions[0].sessionId);
  }, [orderedSessions, selectedSessionId, sessions]);
  useEffect(() => { localStorage.setItem('dermis.room-share-origin', shareOrigin); }, [shareOrigin]);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, []);

  const selectedSession = sessions.find(session => session.sessionId === selectedSessionId);
  const latest = states.find(state => state.sessionId === selectedSessionId);
  const selectedAnnotations = useMemo(() => annotations.filter(row => row.sessionId === selectedSessionId)
    .sort((a, b) => Number(a.episodeTimeMicros - b.episodeTimeMicros)), [annotations, selectedSessionId]);
  const selectedJobs = useMemo(() => jobs.filter(row => row.sessionId === selectedSessionId)
    .sort((a, b) => b.updatedAt.toDate().getTime() - a.updatedAt.toDate().getTime()), [jobs, selectedSessionId]);
  const selectedTasks = useMemo(() => taskRequests.filter(row => row.sessionId === selectedSessionId)
    .sort((a, b) => b.updatedAt.toDate().getTime() - a.updatedAt.toDate().getTime()), [taskRequests, selectedSessionId]);
  const selectedTaskEvents = useMemo(() => taskEvents.filter(row => row.sessionId === selectedSessionId)
    .sort((a, b) => a.createdAt.toDate().getTime() - b.createdAt.toDate().getTime()), [taskEvents, selectedSessionId]);

  useEffect(() => {
    if (!latest) return;
    setHistory(previous => {
      if (previous[previous.length - 1]?.sequence === latest.sequence) return previous;
      return [...previous, { sequence: latest.sequence, value: latest.appliedDownwardLoad, receivedAt: Date.now() }].slice(-90);
    });
  }, [latest]);
  useEffect(() => setHistory([]), [selectedSessionId]);

  const submitAnnotation = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!latest || !selectedSessionId) return;
    setAnnotationError('');
    try {
      await addAnnotation({ sessionId: selectedSessionId, episodeTimeMicros: latest.sourceTimeMicros, label, note });
      setNote('');
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : 'Unable to publish annotation');
    }
  };

  const requestTrial = async () => {
    if (!selectedSessionId) return;
    setTaskError('');
    try {
      const requestId = `trial-${crypto.randomUUID()}`;
      if (selectedSession?.task === 'assist_hand') await requestAutonomousAssistiveGrasp({ requestId, sessionId: selectedSessionId });
      else await requestAutonomousTrial({ requestId, sessionId: selectedSessionId });
    } catch (error) {
      setTaskError(error instanceof Error ? error.message : 'Unable to queue trial');
    }
  };
  const markMoment = async () => {
    if (!latest || !selectedSessionId) return;
    setAnnotationError('');
    try {
      await addAnnotation({ sessionId: selectedSessionId, episodeTimeMicros: latest.sourceTimeMicros,
        label: 'judge_mark', note: note.trim() || 'Judge marked this moment.' });
      setNote('');
    } catch (error) { setAnnotationError(error instanceof Error ? error.message : 'Unable to mark this moment'); }
  };

  const ready = sessionsReady && statesReady && annotationsReady && jobsReady && taskRequestsReady && taskEventsReady && viewersReady;
  const stale = isStateStale(latest, now);
  const snapshotRate = history.length < 2 ? 'sampling…' : (() => {
    const elapsedSeconds = Math.max((history[history.length - 1].receivedAt - history[0].receivedAt) / 1_000, .001);
    return `${((history.length - 1) / elapsedSeconds).toFixed(1)} Hz`;
  })();
  const sampleAge = latest ? `${Math.max(0, now - latest.serverReceiptTime.toDate().getTime())} ms` : '—';
  const publicOrigin = shareOriginFrom(shareOrigin);
  const roomLink = publicOrigin && selectedSessionId ? `${publicOrigin}/room/${encodeURIComponent(selectedSessionId)}` : '';
  const participantCount = new Set(viewers.map(viewer => viewer.identity.toHexString())).size;
  const copyRoomLink = async () => {
    if (!roomLink) return;
    await navigator.clipboard.writeText(roomLink);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1_500);
  };
  return <main className={`viewer ${phoneRoom ? 'phone-room' : ''}`}>
    <header className="topbar">
      <div><p className="eyebrow">DERMIS · SPACETIMEDB SHARED SESSION</p><h1>Teach the robot together</h1>
        <p className="intro">One person demonstrates. A second person coaches. The local simulator and every observer see the same live experiment.</p></div>
      <div className="indicators">
        <span className={`pill ${isActive ? 'ok' : 'bad'}`}>{isActive ? 'CONNECTED' : 'CONNECTING'}</span>
        <span className={`pill ${ready ? 'ok' : 'warn'}`}>{ready ? 'SUBSCRIBED' : 'SYNCING'}</span>
        <span className={`pill ${stale ? 'bad' : 'ok'}`}>{stale ? 'STALE' : 'LIVE'}</span>
        <span className="pill neutral">{participantCount} CONNECTED</span>
      </div>
      {selectedSession && !phoneRoom && <div className="room-join">
        {roomLink ? <JoinQr href={roomLink} /> : <div className="qr-placeholder" aria-hidden="true">QR</div>}
        <div className="room-details"><b>Room {selectedSession.sessionId.slice(-8)}</b>
          <label>Phone join address<input value={shareOrigin} onChange={event => setShareOrigin(event.target.value)} placeholder="http://MAC_WIFI_IP:5173" inputMode="url" /></label>
          <div className="room-actions"><button onClick={() => void copyRoomLink()} disabled={!roomLink}>{copied ? 'Copied' : 'Copy join link'}</button><button onClick={() => void document.documentElement.requestFullscreen()}>Present</button></div>
          {!roomLink && <small>Enter this Mac’s Wi-Fi address to create a phone-safe QR code.</small>}
        </div>
      </div>}
    </header>

    <section className="session-bar" aria-label="session selection">
      <label>Shared experiment
        <select value={selectedSessionId} onChange={event => setSelectedSessionId(event.target.value)}>
          {!orderedSessions.length && <option value="">No shared sessions</option>}
          {orderedSessions.map(session => <option key={session.sessionId} value={session.sessionId}>
            {sessionName(session)}
          </option>)}
        </select>
      </label>
      {selectedSession && <>
        <span className="pill neutral">{sourceLabel(selectedSession.source)}</span>
        <span className="pill neutral">{selectedSession.controlMode.toLowerCase()} control</span>
      </>}
      <span className="identity">observer {identity?.toHexString().slice(0, 8) ?? '…'}</span>
    </section>

    {!isActive && <section className="connection-help"><b>Second-device connection help</b><span>Open this dashboard from the Mac’s current Wi-Fi IP on both devices. Both devices must be on the same Wi-Fi; allow incoming connections for Node/Vite and SpacetimeDB if macOS asks.</span></section>}
    {!selectedSession ? <section className="empty"><h2>Waiting for a teaching session</h2><p>Start a publisher, then every viewer will receive the current shared experiment.</p></section> : <>
      <section className="experiment-card"><div><span>Controller</span><b>{selectedSession.controlMode === 'AUTONOMOUS' ? selectedSession.task === 'assist_hand' ? 'Three-finger learned policy' : 'Cup learned policy' : 'Manual demonstrator'}</b></div><div><span>Model checkpoint</span><b>{selectedSession.checkpointId ?? 'Not applicable'}</b></div><div><span>Live collaboration</span><b>State · notes · requests</b></div></section>
      <section className="visual-stage"><RobotScene key={selectedSessionId} state={latest} stale={stale} task={selectedSession.task} /><div className="visual-legend"><b>{selectedSession.source === 'REPLAY' ? 'Recorded neural-policy playback' : selectedSession.source === 'SCRIPTED_SIMULATION' ? selectedSession.task === 'assist_hand' ? 'Live PyBullet three-finger assistive grasp + learned policy' : 'Live PyBullet cup simulation + learned policy' : 'Subscribed experiment state'}</b><span>{selectedSession.task === 'assist_hand' ? 'The palm, all nine phalanx angles, mug pose, and contact loads are rendered from the shared snapshot.' : 'Parallel-jaw gripper and cup are rendered only from the shared snapshot.'}</span></div></section>
      <section className="metrics" aria-label="live state">
        <article><p>Downward {selectedSession.task === 'assist_hand' ? 'mug' : 'cup'} force</p><strong>{latest ? `${latest.appliedDownwardLoad.toFixed(2)} N` : '—'}</strong></article>
        <article><p>{selectedSession.task === 'assist_hand' ? 'Mug' : 'Cup'} movement</p><strong>{latest ? `${(latest.cupRelativeDisplacement * 1000).toFixed(1)} mm` : '—'}</strong></article>
        <article><p>{selectedSession.task === 'assist_hand' ? 'Grasp' : 'Pickup'} result</p><strong className={latest?.outcome === 'DROPPED' ? 'danger-text' : ''}>{latest?.outcome?.replace('_', ' ') ?? 'WAITING'}</strong></article>
        <article><p>Latest shared frame</p><strong>{latest ? String(latest.sequence) : '—'}</strong></article>
      </section>
      <section className="grid">
        <article className="panel chart-panel"><h2>Force during the {selectedSession.task === 'assist_hand' ? 'grasp' : 'pickup'}</h2><LiveChart history={history} />
          <p className="muted">{latest ? `Last shared update: ${when(latest.serverReceiptTime)}. This chart fills as new frames arrive.` : 'No snapshot yet'}</p>
        </article>
        <article className="panel"><h2>Fingertip contact</h2>
          {latest ? <div className="loads">{latest.fingertipNormalLoads.map((load, index) => <div key={index}><span>{selectedSession.task === 'fragile_cup' ? ['Left gripper finger', 'Right gripper finger'][index] ?? `Contact ${index + 1}` : ['Thumb', 'Index', 'Middle'][index] ?? `Finger ${index + 1}`}</span><b>{load.toFixed(2)} N</b></div>)}</div> : <p className="muted">Waiting for state</p>}
          {latest && <p className="muted">These are the contact forces currently shared by the experiment.</p>}
        </article>
        <article className="panel"><h2>Training / evaluation</h2>
          {!selectedJobs.length && <p className="muted">No learning or evaluation job is running.</p>}
          {selectedJobs.map(job => <div className="job" key={job.jobId}><b>{job.type} · {job.status}</b><span>{job.progress === undefined ? 'progress unavailable' : `${Math.round(job.progress * 100)}%`}</span>{job.resultSummary && <small>{job.resultSummary}</small>}</div>)}
        </article>
        <article className="panel task-panel"><h2>Ask the robot to try a {selectedSession.task === 'assist_hand' ? 'three-finger grasp' : 'cup pickup'}</h2>
          <p className="muted">This creates one shared request for the local simulator. Every observer sees when the worker claims and completes it.</p>
          <button onClick={requestTrial} disabled={!isActive || !selectedSession}>Request {selectedSession.task === 'assist_hand' ? 'three-finger grasp' : 'cup pickup trial'}</button>
          {taskError && <p className="error">{taskError}</p>}
          {stale && selectedTasks.some(task => task.status === 'RUNNING') && <p className="error">Worker state is stale. The run may be unavailable; do not retry until the presenter confirms its outcome.</p>}
          {!selectedTasks.length && <p className="muted">No requests for this session.</p>}
          {selectedTasks.map(task => <div className="task" key={task.requestId}>
            <b>{task.status} · {task.taskType}</b><span>{task.requestId}</span>
            <small>requested {when(task.requestedAt)} · {task.requestedBy.toHexString().slice(0, 8)}</small>
            {task.resultSummary && <p>{task.resultSummary}</p>}
          </div>)}
          {!!selectedTaskEvents.length && <div className="task-events">{selectedTaskEvents.map(event => <div key={String(event.id)}>
            <b>{event.status}</b> · {event.detail} <small>{when(event.createdAt)} · {event.actorIdentity.toHexString().slice(0, 8)}</small>
          </div>)}</div>}
        </article>
        <article className="panel annotations"><h2>Coach notes</h2>
          <form onSubmit={submitAnnotation}><input value={label} maxLength={80} onChange={event => setLabel(event.target.value)} aria-label="annotation label" />
            <textarea value={note} maxLength={2000} onChange={event => setNote(event.target.value)} placeholder="What should the robot learn from this moment?" aria-label="annotation note" />
            <div className="annotation-actions"><button disabled={!isActive || !latest || !label.trim()}>Share coach note</button><button type="button" onClick={markMoment} disabled={!isActive || !latest}>Mark this moment</button></div>{annotationError && <p className="error">{annotationError}</p>}
          </form>
          <div className="annotation-list">{!selectedAnnotations.length && <p className="muted">No annotations yet.</p>}
            {selectedAnnotations.map(row => <div className="annotation" key={String(row.id)}><b>{row.label}</b><span>{String(row.episodeTimeMicros)} µs · {row.authorIdentity.toHexString().slice(0, 8)}</span>{row.note && <p>{row.note}</p>}</div>)}
          </div>
        </article>
      </section>
      <details className="engineering panel"><summary>SpacetimeDB shared-session evidence</summary><div className="engineering-grid">
          <div><span>Connected clients</span><b>{viewers.length}</b></div><div><span>Observed snapshot rate</span><b>{snapshotRate}</b></div>
          <div><span>Latest sequence</span><b>{latest ? String(latest.sequence) : '—'}</b></div><div><span>State age</span><b>{sampleAge}</b></div>
        </div><p className="muted">Subscriptions synchronize state, notes, task lifecycle, and job status. Rate is measured by this viewer; state age is server-receipt age, not end-to-end latency.</p>
      </details>
    </>}
  </main>;
}
