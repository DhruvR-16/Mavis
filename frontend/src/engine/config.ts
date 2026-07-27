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
  scoring: Record<string, number>;
  faults: Record<string, FaultDef>;
  requiredRegions: string[];
}

export const TOLERANCES = shared.tolerances;
export const CALIBRATION = shared.calibration;

export function getExercise(key: ExerciseKey): ExerciseDef {
  const def = (shared.exercises as Record<string, unknown>)[key];
  if (!def) throw new Error(`Unknown exercise '${key}'`);
  return def as unknown as ExerciseDef;
}
