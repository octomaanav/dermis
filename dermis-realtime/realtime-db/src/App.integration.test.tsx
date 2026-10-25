import { describe, expect, it } from 'vitest';
import { isStateStale } from './App';
import type { LatestState } from './module_bindings/types';

describe('viewer stale indicator', () => {
  it('marks an old subscribed snapshot stale without polling', () => {
    const now = 10_000;
    const fresh = { serverReceiptTime: { toDate: () => new Date(now - 100) } } as LatestState;
    const old = { serverReceiptTime: { toDate: () => new Date(now - 2_000) } } as LatestState;
    expect(isStateStale(fresh, now)).toBe(false);
    expect(isStateStale(old, now)).toBe(true);
    expect(isStateStale(undefined, now)).toBe(true);
  });
});
