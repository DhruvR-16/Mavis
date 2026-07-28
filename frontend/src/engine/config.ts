/**
 * Typed access to the shared exercise definitions in exercises.json.
 *
 * The same file is read by the Python engine (analyzer/exercise_config.py), so
 * a threshold only ever exists in one place. Before this, each runtime carried
 * its own hand-maintained copy and they had already drifted apart — most
 * seriously, this engine measured the shoulder joint for a press while Python
 * measured the elbow, applying identical thresholds to both.
 */

import shared from '../../../exercises.json';

export type ExerciseKey = 'bicep' | 'shoulder';

export interface FaultDef {
  message: string;
  severity: number;
  toleranceDeg?: number;
  rampDeg?: number;
}

export interface ExerciseDef {
  displayName: string;
  accent: string;
  angle: {
    triplets: { left: [number, number, number]; right: [number, number, number] };
    armSelection: 'active' | 'bilateral';
  };
  thresholds: { up: number; down: number };
  direction: 'flexion' | 'extension';
  /** How far past tolerance counts as a full-weight miss, per category. */
  ramps: Partial<Record<
    'peakDeg' | 'bottomDeg' | 'driftRatio' | 'lockoutDeg' | 'depthDeg' | 'symmetryDeg',
    number
  >>;
  scoring: Record<string, number>;
  faults: Record<string, FaultDef>;
  requiredRegions: string[];
}

export const TOLERANCES = shared.tolerances;
export const CALIBRATION = shared.calibration;

/**
 * Scale a deduction by how badly the athlete actually missed.
 *
 * A binary threshold punishes missing by 1° exactly as hard as missing by 40°,
 * which makes scores feel arbitrary and unforgiving. Returns no penalty while
 * within tolerance, then ramps linearly to the full weight once `ramp` units
 * past it. Mirrors graded_penalty() in analyzer/base_analyzer.py.
 */
export function gradedPenalty(excess: number, ramp: number, weight: number): number {
  if (excess <= 0) return 0;
  return Math.round(weight * Math.min(1, excess / ramp));
}

export function getExercise(key: ExerciseKey): ExerciseDef {
  const def = (shared.exercises as Record<string, unknown>)[key];
  if (!def) throw new Error(`Unknown exercise '${key}'`);
  return def as unknown as ExerciseDef;
}
