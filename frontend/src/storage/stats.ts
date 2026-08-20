/**
 * Aggregations over session history.
 *
 * Pure functions over a SessionRecord[] — no IndexedDB, no React — so the
 * arithmetic that answers "am I actually improving?" is directly testable.
 */

import type { HistorySummary, SessionRecord, TrendPoint } from './types';

/** Local-time YYYY-MM-DD. Grouping by UTC would shift evening sessions a day. */
export function dayKey(iso: string): string {
  const d = new Date(iso);
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${month}-${day}`;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/**
 * Mean quality weighted by rep count.
 *
 * Averaging session averages would let a 2-rep warmup swing the number as hard
 * as a 40-rep session.
 */
export function overallAvgQuality(sessions: SessionRecord[]): number {
  const reps = sessions.flatMap((s) => s.repQualities);
  if (reps.length) return Math.round(mean(reps));

  // Older records may predate per-rep storage; fall back to rep-weighted means.
  const totalReps = sessions.reduce((n, s) => n + s.totalReps, 0);
  if (!totalReps) return 0;
  const weighted = sessions.reduce((n, s) => n + s.avgQuality * s.totalReps, 0);
  return Math.round(weighted / totalReps);
}

/** One point per day that has at least one session, oldest first. */
export function dailyTrend(sessions: SessionRecord[]): TrendPoint[] {
  const byDay = new Map<string, SessionRecord[]>();
  for (const s of sessions) {
    const key = dayKey(s.date);
    const bucket = byDay.get(key);
    if (bucket) bucket.push(s);
    else byDay.set(key, [s]);
  }

  return [...byDay.entries()]
    .map(([day, group]) => ({
      day,
      avgQuality: overallAvgQuality(group),
      totalReps: group.reduce((n, s) => n + s.totalReps, 0),
      sessions: group.length,
    }))
    .sort((a, b) => a.day.localeCompare(b.day));
}

/**
 * Consecutive days with a session, counting back from today.
 *
 * Training today is not required to hold a streak — a session yesterday still
 * counts, so the number doesn't reset just because you haven't trained yet.
 */
export function currentStreakDays(sessions: SessionRecord[], now = new Date()): number {
  if (!sessions.length) return 0;

  const days = new Set(sessions.map((s) => dayKey(s.date)));
  const cursor = new Date(now);
  const key = (d: Date) => dayKey(d.toISOString());

  if (!days.has(key(cursor))) {
    cursor.setDate(cursor.getDate() - 1);
    if (!days.has(key(cursor))) return 0;
  }

  let streak = 0;
  while (days.has(key(cursor))) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

export function mostCommonFault(sessions: SessionRecord[]): string | null {
  const counts = new Map<string, number>();
  for (const s of sessions) {
    if (!s.topFault) continue;
    counts.set(s.topFault, (counts.get(s.topFault) ?? 0) + 1);
  }
  let best: string | null = null;
  let bestCount = 0;
  for (const [fault, count] of counts) {
    if (count > bestCount) {
      best = fault;
      bestCount = count;
    }
  }
  return best;
}

export function summarise(sessions: SessionRecord[], now = new Date()): HistorySummary {
  return {
    totalSessions: sessions.length,
    totalReps: sessions.reduce((n, s) => n + s.totalReps, 0),
    avgQuality: overallAvgQuality(sessions),
    currentStreakDays: currentStreakDays(sessions, now),
    bestQuality: sessions.reduce((best, s) => Math.max(best, s.avgQuality), 0),
    topFault: mostCommonFault(sessions),
  };
}

/**
 * Quality change between the two halves of the recorded history.
 *
 * Returns null until there is enough data for the comparison to mean anything.
 */
export function improvement(sessions: SessionRecord[]): number | null {
  const trend = dailyTrend(sessions);
  if (trend.length < 4) return null;
  const mid = Math.floor(trend.length / 2);
  const earlier = mean(trend.slice(0, mid).map((p) => p.avgQuality));
  const later = mean(trend.slice(mid).map((p) => p.avgQuality));
  return Math.round(later - earlier);
}
