import { beforeEach, describe, expect, it, vi } from 'vitest';

import * as db from '../db';
import {
  clearHistory,
  loadHistory,
  migrateLegacySession,
  newSessionId,
  recordSession,
  removeSession,
} from '../history';
import type { SessionRecord } from '../types';

const LEGACY_KEY = 'mavis_last_session';
const MIGRATED_KEY = 'mavis_history_migrated';

function session(overrides: Partial<SessionRecord> = {}): SessionRecord {
  return {
    id: newSessionId(),
    date: new Date().toISOString(),
    exercise: 'bicep',
    mode: 'free',
    totalReps: 10,
    goodReps: 9,
    badReps: 1,
    avgQuality: 88,
    topFault: null,
    durationSec: 300,
    repQualities: Array(10).fill(88),
    ...overrides,
  };
}

/** A minimal in-memory localStorage; the node test env has none. */
function installLocalStorage() {
  const data = new Map<string, string>();
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
    clear: () => data.clear(),
    key: (i: number) => [...data.keys()][i] ?? null,
    get length() {
      return data.size;
    },
  });
  return data;
}

beforeEach(async () => {
  installLocalStorage();
  await clearHistory();
});

describe('newSessionId', () => {
  it('produces unique ids', () => {
    const ids = new Set(Array.from({ length: 200 }, () => newSessionId()));
    expect(ids.size).toBe(200);
  });
});

describe('recordSession / loadHistory', () => {
  it('starts empty', async () => {
    expect(await loadHistory()).toEqual([]);
  });

  it('persists a session', async () => {
    await recordSession(session({ totalReps: 12 }));
    const history = await loadHistory();
    expect(history).toHaveLength(1);
    expect(history[0].totalReps).toBe(12);
  });

  it('keeps every session instead of overwriting the previous one', async () => {
    // The whole point of the change: the old localStorage key held exactly one
    // session and was clobbered on every workout.
    await recordSession(session({ totalReps: 5 }));
    await recordSession(session({ totalReps: 8 }));
    await recordSession(session({ totalReps: 11 }));
    expect(await loadHistory()).toHaveLength(3);
  });

  it('returns sessions newest first', async () => {
    await recordSession(session({ date: '2026-07-01T10:00:00.000Z', totalReps: 1 }));
    await recordSession(session({ date: '2026-07-03T10:00:00.000Z', totalReps: 3 }));
    await recordSession(session({ date: '2026-07-02T10:00:00.000Z', totalReps: 2 }));
    const reps = (await loadHistory()).map((s) => s.totalReps);
    expect(reps).toEqual([3, 2, 1]);
  });

  it('round-trips per-rep quality data', async () => {
    await recordSession(session({ repQualities: [70, 85, 92] }));
    const [saved] = await loadHistory();
    expect(saved.repQualities).toEqual([70, 85, 92]);
  });

  it('removes a session by id', async () => {
    const keep = session();
    const drop = session();
    await recordSession(keep);
    await recordSession(drop);
    await removeSession(drop.id);
    const remaining = await loadHistory();
    expect(remaining.map((s) => s.id)).toEqual([keep.id]);
  });
});

describe('migrateLegacySession', () => {
  it('imports the old single-session payload', async () => {
    localStorage.setItem(
      LEGACY_KEY,
      JSON.stringify({
        total_reps: 14,
        avg_quality: 82,
        good_reps: 13,
        most_common_fault: 'Elbow drifting forward',
        exercise: 'bicep',
        mode: 'free',
        date: '2026-07-10T09:00:00.000Z',
      }),
    );

    expect(await migrateLegacySession()).toBe(true);

    const history = await db.listSessions();
    expect(history).toHaveLength(1);
    expect(history[0].totalReps).toBe(14);
    expect(history[0].topFault).toBe('Elbow drifting forward');
    expect(history[0].badReps).toBe(1);
  });

  it('runs only once even if called repeatedly', async () => {
    localStorage.setItem(LEGACY_KEY, JSON.stringify({ total_reps: 5, date: '2026-07-10T09:00:00.000Z' }));

    expect(await migrateLegacySession()).toBe(true);
    expect(await migrateLegacySession()).toBe(false);
    expect(await migrateLegacySession()).toBe(false);

    expect(await db.listSessions()).toHaveLength(1);
  });

  it('does not duplicate the legacy record across loadHistory calls', async () => {
    localStorage.setItem(LEGACY_KEY, JSON.stringify({ total_reps: 5, date: '2026-07-10T09:00:00.000Z' }));
    await loadHistory();
    await loadHistory();
    expect(await loadHistory()).toHaveLength(1);
  });

  it('treats a "None" fault as no fault', async () => {
    localStorage.setItem(
      LEGACY_KEY,
      JSON.stringify({ total_reps: 3, most_common_fault: 'None', date: '2026-07-10T09:00:00.000Z' }),
    );
    await migrateLegacySession();
    const [imported] = await db.listSessions();
    expect(imported.topFault).toBeNull();
  });

  it('skips a zero-rep legacy session', async () => {
    // Usually means the camera was denied or the page closed early — importing
    // it would leave an empty row in history and drag the averages down.
    localStorage.setItem(
      LEGACY_KEY,
      JSON.stringify({ total_reps: 0, avg_quality: 0, date: '2026-07-10T09:00:00.000Z' }),
    );
    expect(await migrateLegacySession()).toBe(false);
    expect(await db.listSessions()).toEqual([]);
  });

  it('marks itself done when there is nothing to import', async () => {
    expect(await migrateLegacySession()).toBe(false);
    expect(localStorage.getItem(MIGRATED_KEY)).toBe('1');
  });

  it('survives a corrupt payload without blocking history', async () => {
    localStorage.setItem(LEGACY_KEY, '{not valid json');
    expect(await migrateLegacySession()).toBe(false);
    expect(await loadHistory()).toEqual([]);
  });

  it('leaves the legacy key in place', async () => {
    // Kept deliberately: it costs nothing, and a rolled-back deploy would
    // otherwise lose the user's last session.
    localStorage.setItem(LEGACY_KEY, JSON.stringify({ total_reps: 5, date: '2026-07-10T09:00:00.000Z' }));
    await migrateLegacySession();
    expect(localStorage.getItem(LEGACY_KEY)).not.toBeNull();
  });
});

describe('degrading without IndexedDB', () => {
  it('reads empty and writes without throwing', async () => {
    // Private browsing and blocked-storage origins must not break a workout.
    const spy = vi.spyOn(db, 'isAvailable').mockReturnValue(false);
    await expect(recordSession(session())).resolves.toBeUndefined();
    await expect(loadHistory()).resolves.toEqual([]);
    spy.mockRestore();
  });
});
