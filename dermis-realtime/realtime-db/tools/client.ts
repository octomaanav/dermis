import { mkdirSync, readFileSync, writeFileSync, chmodSync } from 'node:fs';
import { dirname } from 'node:path';
import { DbConnection, tables } from '../src/module_bindings/index.ts';

export const host = process.env.SPACETIMEDB_HOST ?? 'ws://127.0.0.1:3000';
export const database = process.env.SPACETIMEDB_DB_NAME ?? 'dermis-realtime-db';

export function option(name: string, fallback?: string) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

export function loadToken(path: string) {
  try { return readFileSync(path, 'utf8').trim() || undefined; } catch { return undefined; }
}

export function saveToken(path: string, token: string) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, token, { encoding: 'utf8', mode: 0o600 });
  chmodSync(path, 0o600);
}

export async function connect(tokenPath: string, onOnline?: (online: boolean) => void) {
  return await new Promise<InstanceType<typeof DbConnection>>((resolve, reject) => {
    let settled = false;
    const conn = DbConnection.builder().withUri(option('host', host)!).withDatabaseName(option('database', database)!)
      .withToken(loadToken(tokenPath)).withAutomaticReconnect()
      .onConnect((_conn, identity, token) => {
        saveToken(tokenPath, token); onOnline?.(true);
        if (!settled) { settled = true; resolve(conn); }
        console.error(`connected as ${identity.toHexString().slice(0, 12)}`);
      })
      .onDisconnect((_ctx, error) => { onOnline?.(false); console.error(`disconnected: ${error?.message ?? 'closed'}`); })
      .onConnectError((_ctx, error, attempt) => {
        console.error(`connect error${attempt === undefined ? '' : ` (retry ${attempt})`}: ${error.message}`);
        if (!settled && attempt === undefined) { settled = true; reject(error); }
      }).build();
  });
}

export async function waitForTables(conn: InstanceType<typeof DbConnection>) {
  await new Promise<void>(resolve => conn.subscriptionBuilder().onApplied(resolve)
    .subscribe([tables.session, tables.latestState, tables.annotation, tables.job,
      tables.taskRequest, tables.taskEvent, tables.viewer]));
}

export const sleep = (milliseconds: number) => new Promise(resolve => setTimeout(resolve, milliseconds));
