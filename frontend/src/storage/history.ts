/**
 * Public API for session history.
 *
 * Wraps the IndexedDB store with the legacy-data migration and a safe
 * degradation path: if storage is unavailable (private browsing, blocked
 * origin), reads return empty and writes no-op rather than throwing into the
 * middle of a workout.
 */

import * as db from './db';
import type { SessionRecord } from './types';

/** The single-session key used before history existed. */
const LEGACY_KEY = 'mavis_last_session';
/** Set once the legacy record has been imported, so it happens exactly once. */
const MIGRATED_KEY = 'mavis_history_migrated';

export function newSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

interface LegacyPayload {
  total_reps?: number;
  avg_quality?: number;
  good_reps?: number;
  most_common_fault?: string;
  exercise?: string;
  mode?: string;
  date?: string;
}

/**
 * Import the pre-history localStorage record, once.
 *
 * Returning users would otherwise open the new history page to an empty list
 * despite having just trained. Idempotent via MIGRATED_KEY, and the legacy key
 * is left in place rather than deleted — it costs nothing and means a rolled
 * back deploy doesn't lose the record.
 */
export async function migrateLegacySession(): Promise<boolean> {
  if (typeof localStorage === 'undefined') return false;
  if (localStorage.getItem(MIGRATED_KEY)) return false;

  const raw = localStorage.getItem(LEGACY_KEY);
  if (!raw) {
    localStorage.setItem(MIGRATED_KEY, '1');
    return false;
  }

  try {
    const legacy = JSON.parse(raw) as LegacyPayload;
    const totalReps = legacy.total_reps ?? 0;
    const goodReps = legacy.good_reps ?? 0;

    // A session with no reps isn't a workout — usually the camera was denied
    // or the page was closed early. Importing it would put an empty row in the
    // history list and drag the averages down.
    if (totalReps <= 0) {
      localStorage.setItem(MIGRATED_KEY, '1');
      return false;
    }

    await db.saveSession({
      id: `legacy_${legacy.date ?? Date.now()}`,
      date: legacy.date ?? new Date().toISOString(),
      exercise: legacy.exercise === 'shoulder' ? 'shoulder' : 'bicep',
      mode: legacy.mode === 'program' ? 'program' : 'free',
      totalReps,
      goodReps,
      badReps: Math.max(0, totalReps - goodReps),
      avgQuality: legacy.avg_quality ?? 0,
      topFault:
        legacy.most_common_fault && legacy.most_common_fault !== 'None'
          ? legacy.most_common_fault
          : null,
      durationSec: 0,
      // Per-rep detail predates this record; stats fall back to rep-weighted
      // session means when this is empty.
      repQualities: [],
    });

    localStorage.setItem(MIGRATED_KEY, '1');
    return true;
  } catch {
    // Corrupt legacy payload is not worth blocking history over.
    localStorage.setItem(MIGRATED_KEY, '1');
    return false;
  }
}

export async function recordSession(record: SessionRecord): Promise<void> {
  if (!db.isAvailable()) return;
  try {
    await db.saveSession(record);
  } catch {
    // A failed write must never interrupt the end-of-session flow.
  }
}

export async function loadHistory(): Promise<SessionRecord[]> {
  if (!db.isAvailable()) return [];
  try {
    await migrateLegacySession();
    return await db.listSessions();
  } catch {
    return [];
  }
}

export async function removeSession(id: string): Promise<void> {
  if (!db.isAvailable()) return;
  try {
    await db.deleteSession(id);
  } catch {
    /* ignore */
  }
}

export async function clearHistory(): Promise<void> {
  if (!db.isAvailable()) return;
  try {
    await db.clearAll();
  } catch {
    /* ignore */
  }
}

export type { SessionRecord } from './types';
