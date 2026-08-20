import { describe, expect, it } from 'vitest';

import {
  currentStreakDays,
  dailyTrend,
  dayKey,
  improvement,
  mostCommonFault,
  overallAvgQuality,
  summarise,
} from '../stats';
import type { SessionRecord } from '../types';

function session(overrides: Partial<SessionRecord> = {}): SessionRecord {
  const repQualities = overrides.repQualities ?? [80, 80];
  return {
    id: Math.random().toString(36).slice(2),
    date: '2026-07-01T10:00:00.000Z',
    exercise: 'bicep',
    mode: 'free',
    totalReps: repQualities.length,
    goodReps: repQualities.filter((q) => q >= 60).length,
    badReps: repQualities.filter((q) => q < 60).length,
    avgQuality: Math.round(repQualities.reduce((a, b) => a + b, 0) / repQualities.length),
    topFault: null,
    durationSec: 120,
    repQualities,
    ...overrides,
  };
}

/** Builds an ISO timestamp `daysAgo` days before `from`, at local midday. */
function daysBefore(daysAgo: number, from = new Date()): string {
  const d = new Date(from);
  d.setDate(d.getDate() - daysAgo);
  d.setHours(12, 0, 0, 0);
  return d.toISOString();
}

describe('dayKey', () => {
  it('groups by local calendar day', () => {
    const evening = new Date(2026, 6, 15, 22, 30);
    expect(dayKey(evening.toISOString())).toBe('2026-07-15');
  });

  it('does not roll a late-evening session into the next day', () => {
    // Grouping on the UTC date would push an 11pm session in a positive-offset
    // timezone onto tomorrow, splitting one workout across two days.
    const late = new Date(2026, 6, 15, 23, 45);
    expect(dayKey(late.toISOString())).toBe('2026-07-15');
  });
});

describe('overallAvgQuality', () => {
  it('is zero with no sessions', () => {
    expect(overallAvgQuality([])).toBe(0);
  });

  it('weights by rep count, not by session', () => {
    // A 2-rep warmup must not swing the average as hard as a 40-rep session.
    const warmup = session({ repQualities: [40, 40] });
    const real = session({ repQualities: Array(40).fill(90) });
    const avg = overallAvgQuality([warmup, real]);
    expect(avg).toBeGreaterThan(85);
  });

  it('falls back to rep-weighted session means for records without per-rep data', () => {
    const legacy = session({ repQualities: [], totalReps: 10, avgQuality: 70 });
    expect(overallAvgQuality([legacy])).toBe(70);
  });

  it('ignores zero-rep sessions in the fallback rather than dividing by zero', () => {
    const empty = session({ repQualities: [], totalReps: 0, avgQuality: 0 });
    expect(overallAvgQuality([empty])).toBe(0);
  });
});

describe('dailyTrend', () => {
  it('is empty with no sessions', () => {
    expect(dailyTrend([])).toEqual([]);
  });

  it('collapses multiple sessions on one day into a single point', () => {
    const trend = dailyTrend([
      session({ date: daysBefore(1) }),
      session({ date: daysBefore(1) }),
    ]);
    expect(trend).toHaveLength(1);
    expect(trend[0].sessions).toBe(2);
  });

  it('returns points oldest first', () => {
    const trend = dailyTrend([
      session({ date: daysBefore(1) }),
      session({ date: daysBefore(5) }),
      session({ date: daysBefore(3) }),
    ]);
    const days = trend.map((p) => p.day);
    expect([...days].sort()).toEqual(days);
  });

  it('sums reps across the day', () => {
    const trend = dailyTrend([
      session({ date: daysBefore(1), repQualities: [90, 90, 90] }),
      session({ date: daysBefore(1), repQualities: [80, 80] }),
    ]);
    expect(trend[0].totalReps).toBe(5);
  });
});

describe('currentStreakDays', () => {
  const now = new Date(2026, 6, 20, 18, 0);

  it('is zero with no sessions', () => {
    expect(currentStreakDays([], now)).toBe(0);
  });

  it('counts consecutive days ending today', () => {
    const sessions = [0, 1, 2].map((d) => session({ date: daysBefore(d, now) }));
    expect(currentStreakDays(sessions, now)).toBe(3);
  });

  it('still counts a streak that ends yesterday', () => {
    // Not having trained yet today shouldn't zero out a live streak.
    const sessions = [1, 2, 3].map((d) => session({ date: daysBefore(d, now) }));
    expect(currentStreakDays(sessions, now)).toBe(3);
  });

  it('breaks on a missed day', () => {
    const sessions = [0, 1, 3, 4].map((d) => session({ date: daysBefore(d, now) }));
    expect(currentStreakDays(sessions, now)).toBe(2);
  });

  it('is zero when the most recent session is too old', () => {
    expect(currentStreakDays([session({ date: daysBefore(5, now) })], now)).toBe(0);
  });

  it('does not double-count two sessions on the same day', () => {
    const sessions = [
      session({ date: daysBefore(0, now) }),
      session({ date: daysBefore(0, now) }),
    ];
    expect(currentStreakDays(sessions, now)).toBe(1);
  });
});

describe('mostCommonFault', () => {
  it('is null when every session was clean', () => {
    expect(mostCommonFault([session(), session()])).toBeNull();
  });

  it('picks the most frequent fault', () => {
    const sessions = [
      session({ topFault: 'Elbow drifting forward' }),
      session({ topFault: 'Elbow drifting forward' }),
      session({ topFault: 'Torso swinging' }),
    ];
    expect(mostCommonFault(sessions)).toBe('Elbow drifting forward');
  });
});

describe('improvement', () => {
  it('is null until there is enough history to compare', () => {
    expect(improvement([session()])).toBeNull();
    expect(improvement([])).toBeNull();
  });

  it('is positive when later sessions score better', () => {
    const sessions = [
      session({ date: daysBefore(8), repQualities: [60, 60] }),
      session({ date: daysBefore(7), repQualities: [62, 62] }),
      session({ date: daysBefore(2), repQualities: [90, 90] }),
      session({ date: daysBefore(1), repQualities: [92, 92] }),
    ];
    expect(improvement(sessions)!).toBeGreaterThan(0);
  });

  it('is negative when form is regressing', () => {
    const sessions = [
      session({ date: daysBefore(8), repQualities: [95, 95] }),
      session({ date: daysBefore(7), repQualities: [93, 93] }),
      session({ date: daysBefore(2), repQualities: [65, 65] }),
      session({ date: daysBefore(1), repQualities: [60, 60] }),
    ];
    expect(improvement(sessions)!).toBeLessThan(0);
  });
});

describe('summarise', () => {
  it('handles an empty history without dividing by zero', () => {
    expect(summarise([])).toEqual({
      totalSessions: 0,
      totalReps: 0,
      avgQuality: 0,
      currentStreakDays: 0,
      bestQuality: 0,
      topFault: null,
    });
  });

  it('totals reps across sessions', () => {
    const summary = summarise([
      session({ repQualities: [90, 90, 90] }),
      session({ repQualities: [80, 80] }),
    ]);
    expect(summary.totalSessions).toBe(2);
    expect(summary.totalReps).toBe(5);
  });
});
