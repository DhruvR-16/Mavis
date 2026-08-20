/**
 * Shape of a persisted workout session.
 *
 * Everything here is derived from what the engine already computes at the end
 * of a session — no new tracking is required to record it.
 */

export type ExerciseKey = 'bicep' | 'shoulder';
export type SessionMode = 'free' | 'program';

export interface SessionRecord {
  /** Stable id; also the IndexedDB primary key. */
  id: string;
  /** ISO-8601 timestamp of when the session ended. */
  date: string;
  exercise: ExerciseKey;
  mode: SessionMode;

  totalReps: number;
  goodReps: number;
  badReps: number;
  /** Mean rep quality, 0–100. */
  avgQuality: number;
  /** Most frequent fault this session, or null if the session was clean. */
  topFault: string | null;
  /** Wall-clock length of the session in seconds. */
  durationSec: number;
  /** Per-rep quality scores, in order — drives the trend sparkline. */
  repQualities: number[];

  /** Program mode only. */
  setsCompleted?: number;
  setsPlanned?: number;
}

/** A single point on the progress-over-time chart. */
export interface TrendPoint {
  /** ISO date (day granularity). */
  day: string;
  avgQuality: number;
  totalReps: number;
  sessions: number;
}

export interface HistorySummary {
  totalSessions: number;
  totalReps: number;
  /** Mean quality across every recorded rep, not a mean of session means. */
  avgQuality: number;
  /** Consecutive days with at least one session, counting back from today. */
  currentStreakDays: number;
  bestQuality: number;
  /** Most common fault across all sessions, or null. */
  topFault: string | null;
}
