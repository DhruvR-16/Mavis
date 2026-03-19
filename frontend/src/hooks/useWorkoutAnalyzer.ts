import { useRef, useCallback } from 'react';

// ─── Types ──────────────────────────────────────────────────
export type WorkoutType = 'bicep' | 'shoulder';
export type FeedbackType = 'good' | 'bad' | 'info' | 'neutral';

const Stage = { DOWN: 'DOWN', UP: 'UP' } as const;
type StageType = (typeof Stage)[keyof typeof Stage];

export interface WorkoutState {
  repCount: number;
  badRepCount: number;
  elbowAngle: number;
  stage: StageType;
  feedback: string;
  cueTitle: string;
  feedbackType: FeedbackType;
  formColor: string;
  activeSide: 'left' | 'right';
}

// ─── Constants ──────────────────────────────────────────────
const BICEP_UP = 50;
const BICEP_DOWN = 155;
const BICEP_DRIFT = 0.08;
const SHOULDER_UP = 150;
const SHOULDER_DOWN = 80;
const MIN_REP_TIME_MS = 800;

// EMA smoothing factor (0 = full smoothing, 1 = no smoothing)
const EMA_ALPHA = 0.35;

// ─── Helpers ────────────────────────────────────────────────
interface Point3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

function calculateAngle(a: Point3D, b: Point3D, c: Point3D): number {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

function isVisible(l: Point3D | undefined, threshold = 0.5): boolean {
  return !!l && (l.visibility ?? 0) > threshold;
}

const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
};

// ─── Hook ───────────────────────────────────────────────────
export function useWorkoutAnalyzer(workoutType: WorkoutType) {
  const stateRef = useRef<WorkoutState>({
    repCount: 0,
    badRepCount: 0,
    elbowAngle: 0,
    stage: Stage.DOWN,
    feedback: 'Start Workout',
    cueTitle: 'Positioning',
    feedbackType: 'info',
    formColor: '#2563EB',
    activeSide: 'left',
  });

  const repStartTimeRef = useRef(0);
  const anchorElbowXRef = useRef(0);
  const smoothedAngleRef = useRef(0);
  // Debounce: require N consecutive frames before switching feedback
  const feedbackCounterRef = useRef(0);
  const pendingFeedbackRef = useRef<FeedbackType>('info');

  const debounceFeedback = (type: FeedbackType, threshold = 2): FeedbackType => {
    if (type === pendingFeedbackRef.current) {
      feedbackCounterRef.current++;
    } else {
      pendingFeedbackRef.current = type;
      feedbackCounterRef.current = 1;
    }
    return feedbackCounterRef.current >= threshold ? type : stateRef.current.feedbackType;
  };

  const analyze = useCallback(
    (worldLandmarks: Point3D[], normalizedLandmarks: Point3D[]): WorkoutState => {
      const s = stateRef.current;

      if (workoutType === 'bicep') {
        analyzeBicep(s, worldLandmarks, normalizedLandmarks);
      } else {
        analyzeShoulder(s, worldLandmarks, normalizedLandmarks);
      }

      stateRef.current = { ...s };
      return stateRef.current;
    },
    [workoutType],
  );

  // ─── Bicep Curl Analysis ────────────────────────────────
  function analyzeBicep(s: WorkoutState, world: Point3D[], norm: Point3D[]) {
    // Detect active arm
    const lS = world[LM.LEFT_SHOULDER], lE = world[LM.LEFT_ELBOW], lW = world[LM.LEFT_WRIST];
    const rS = world[LM.RIGHT_SHOULDER], rE = world[LM.RIGHT_ELBOW], rW = world[LM.RIGHT_WRIST];
    const leftVis = isVisible(lS) && isVisible(lE) && isVisible(lW);
    const rightVis = isVisible(rS) && isVisible(rE) && isVisible(rW);

    if (leftVis && rightVis) {
      s.activeSide = calculateAngle(lS, lE, lW) < calculateAngle(rS, rE, rW) ? 'left' : 'right';
    } else if (leftVis) s.activeSide = 'left';
    else if (rightVis) s.activeSide = 'right';

    const shoulder = s.activeSide === 'left' ? world[LM.LEFT_SHOULDER] : world[LM.RIGHT_SHOULDER];
    const elbow = s.activeSide === 'left' ? world[LM.LEFT_ELBOW] : world[LM.RIGHT_ELBOW];
    const wrist = s.activeSide === 'left' ? world[LM.LEFT_WRIST] : world[LM.RIGHT_WRIST];
    const hip = s.activeSide === 'left' ? world[LM.LEFT_HIP] : world[LM.RIGHT_HIP];
    // Use normalized landmarks for drift check (screen-space)
    const elbowNorm = s.activeSide === 'left' ? norm[LM.LEFT_ELBOW] : norm[LM.RIGHT_ELBOW];

    if (!isVisible(shoulder) || !isVisible(elbow) || !isVisible(wrist) || !isVisible(hip)) {
      s.cueTitle = 'Positioning'; s.feedback = 'Step into frame'; s.feedbackType = 'neutral'; s.formColor = '#71717A'; return;
    }

    const rawAngle = calculateAngle(shoulder, elbow, wrist);
    // EMA smooth the angle
    smoothedAngleRef.current = EMA_ALPHA * rawAngle + (1 - EMA_ALPHA) * smoothedAngleRef.current;
    s.elbowAngle = smoothedAngleRef.current;
    const shoulderAngle = calculateAngle(elbow, shoulder, hip);
    const now = Date.now();

    if (s.elbowAngle > BICEP_DOWN) {
      if (s.stage === Stage.UP) {
        if (now - repStartTimeRef.current > MIN_REP_TIME_MS) {
          s.repCount++; s.cueTitle = 'Correct Form'; s.feedback = 'Excellent Rep!'; s.feedbackType = debounceFeedback('good'); s.formColor = '#10B981';
        } else {
          s.badRepCount++; s.cueTitle = 'Tempo Control'; s.feedback = 'Too Fast! Slow down'; s.feedbackType = debounceFeedback('bad'); s.formColor = '#EF4444';
        }
        s.stage = Stage.DOWN;
      } else {
        s.formColor = '#2563EB';
        if (!s.feedback.includes('Fast') && !s.feedback.includes('Excellent')) {
          s.cueTitle = 'Ready'; s.feedback = 'Curl upwards!'; s.feedbackType = debounceFeedback('info');
        }
        anchorElbowXRef.current = elbowNorm.x;
      }
    } else if (s.elbowAngle < BICEP_UP) {
      if (s.stage === Stage.DOWN) {
        s.stage = Stage.UP; repStartTimeRef.current = now; anchorElbowXRef.current = elbowNorm.x;
        s.cueTitle = 'Contraction'; s.feedback = 'Squeeze the bicep!'; s.feedbackType = debounceFeedback('good'); s.formColor = '#10B981';
      }
      if (Math.abs(elbowNorm.x - anchorElbowXRef.current) > BICEP_DRIFT) {
        s.cueTitle = 'Form Warning'; s.feedback = 'Fix Elbow! Keep it pinned.'; s.feedbackType = debounceFeedback('bad'); s.formColor = '#EF4444';
      }
    } else {
      s.formColor = '#2563EB';
      if (s.stage === Stage.UP) {
        if (shoulderAngle > 45) { s.cueTitle = 'Torso Swing'; s.feedback = "Don't swing back!"; s.feedbackType = debounceFeedback('bad'); s.formColor = '#EF4444'; }
        else { s.cueTitle = 'Eccentric'; s.feedback = 'Lower slowly...'; s.feedbackType = debounceFeedback('info'); }
      } else if (s.stage === Stage.DOWN) {
        s.cueTitle = 'Concentric'; s.feedback = 'Keep going up!'; s.feedbackType = debounceFeedback('info');
      }
    }

    if (s.stage === Stage.DOWN && s.elbowAngle < BICEP_DOWN && s.elbowAngle > BICEP_UP && s.elbowAngle < 120) {
      s.cueTitle = 'ROM Check'; s.feedback = 'Extend arm fully!'; s.feedbackType = debounceFeedback('info'); s.formColor = '#F59E0B';
    }
  }

  // ─── Shoulder Press Analysis ────────────────────────────
  function analyzeShoulder(s: WorkoutState, world: Point3D[], _norm: Point3D[]) {
    const lS = world[LM.LEFT_SHOULDER], lE = world[LM.LEFT_ELBOW], lW = world[LM.LEFT_WRIST];
    const rS = world[LM.RIGHT_SHOULDER], rE = world[LM.RIGHT_ELBOW], rW = world[LM.RIGHT_WRIST];
    const leftVis = isVisible(lS) && isVisible(lE) && isVisible(lW);
    const rightVis = isVisible(rS) && isVisible(rE) && isVisible(rW);

    if (!leftVis || !rightVis) {
      s.cueTitle = 'Positioning'; s.feedback = 'Both arms must be visible'; s.feedbackType = 'neutral'; s.formColor = '#71717A'; return;
    }

    const leftAngle = calculateAngle(lS, lE, lW);
    const rightAngle = calculateAngle(rS, rE, rW);
    const rawAngle = (leftAngle + rightAngle) / 2;
    smoothedAngleRef.current = EMA_ALPHA * rawAngle + (1 - EMA_ALPHA) * smoothedAngleRef.current;
    s.elbowAngle = smoothedAngleRef.current;
    const now = Date.now();

    if (Math.abs(leftAngle - rightAngle) > 25) {
      s.cueTitle = 'Imbalance'; s.feedback = 'Press evenly!'; s.feedbackType = debounceFeedback('bad'); s.formColor = '#EF4444';
    } else {
      if (s.elbowAngle < SHOULDER_DOWN) {
        if (s.stage === Stage.UP) {
          if (now - repStartTimeRef.current > MIN_REP_TIME_MS) {
            s.repCount++; s.cueTitle = 'Correct Form'; s.feedback = 'Great Press!'; s.feedbackType = debounceFeedback('good'); s.formColor = '#10B981';
          } else {
            s.badRepCount++; s.cueTitle = 'Control Drop'; s.feedback = "Don't drop the weight abruptly"; s.feedbackType = debounceFeedback('bad'); s.formColor = '#EF4444';
          }
          s.stage = Stage.DOWN;
        } else {
          s.formColor = '#2563EB';
          if (!s.feedback.includes('Great') && !s.feedback.includes('drop')) {
            s.cueTitle = 'Ready'; s.feedback = 'Press overhead!'; s.feedbackType = debounceFeedback('info');
          }
        }
      } else if (s.elbowAngle > SHOULDER_UP) {
        if (s.stage === Stage.DOWN) {
          s.stage = Stage.UP; repStartTimeRef.current = now;
          s.cueTitle = 'Lockout'; s.feedback = 'Hold at the top!'; s.feedbackType = debounceFeedback('good'); s.formColor = '#10B981';
        }
      } else {
        s.formColor = '#2563EB';
        if (s.stage === Stage.UP) { s.cueTitle = 'Lowering'; s.feedback = 'Control descent...'; s.feedbackType = debounceFeedback('info'); }
        else if (s.stage === Stage.DOWN) { s.cueTitle = 'Pressing'; s.feedback = 'Push up!'; s.feedbackType = debounceFeedback('info'); }
      }
    }
  }

  const reset = useCallback(() => {
    stateRef.current = {
      repCount: 0, badRepCount: 0, elbowAngle: 0, stage: Stage.DOWN,
      feedback: 'Start Workout', cueTitle: 'Positioning', feedbackType: 'info',
      formColor: '#2563EB', activeSide: 'left',
    };
    smoothedAngleRef.current = 0;
    feedbackCounterRef.current = 0;
  }, []);

  return { analyze, reset, getState: () => stateRef.current };
}
