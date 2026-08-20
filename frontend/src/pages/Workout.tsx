import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Chart } from 'chart.js/auto';
import { usePoseDetection, type PoseLandmark } from '../hooks/usePoseDetection';
import { CALIBRATION, TOLERANCES, getExercise, gradedPenalty } from '../engine/config';
import { newSessionId, recordSession } from '../storage/history';
import './Workout.css';

type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility: number;
};

type SetResult = {
  set: number;
  valid: boolean;
  reps: RepLogItem[];
};

type RepLogItem = {
  n: number;
  quality: number;
  faults: string[];
};

type SessionState = {
  reps: number;
  goodReps: number;
  badReps: number;
  stage: 'down' | 'up';
  activeSide: 'left' | 'right';
  calibrated: boolean;
  calibStart: number;
  calib: {
    shoulderWidth: number;
    /** Neutral elbow offset from its own shoulder, in shoulder-widths. */
    leftElbowOffset: number;
    rightElbowOffset: number;
  };
  repQualities: number[];
  repLog: RepLogItem[];
  /** Top of the movement — start of the eccentric. */
  repStartTime: number;
  /** When the athlete first left the bottom: the true start of the rep. */
  repCycleStart: number;
  repFaults: string[];
  repQualityLive: number;
  peakAngle: number;
  bottomAngle: number;
  maxDrift: number;
  maxLean: number;
  currentSet: number;
  setHistory: SetResult[];
  currentSetReps: RepLogItem[];
  resting: boolean;
  restStart: number;
  programComplete: boolean;
  sessionStart: number;
  lastVoiceTime: number;
  voiceCooldown: number;
  lastFeedback: string;
};

// Thresholds and tolerances come from the repo-root exercises.json, which the
// Python engine reads too — see frontend/src/engine/config.ts.
const TOLERANCE = TOLERANCES.angleDeg;
const DRIFT_TOL = TOLERANCES.driftBodyRatio;
const MIN_TEMPO_MS = TOLERANCES.tempoMinSeconds * 1000;
const CALIB_HOLD_MS = CALIBRATION.holdSeconds * 1000;

// Calibration must be held genuinely still: if these key points move more
// than this (normalised coordinate units) between frames, the countdown
// restarts rather than completing against whatever pose is held at the
// deadline (e.g. mid-curl).
const CALIB_STABILITY_TOL = CALIBRATION.stabilityTolerance;
const CALIB_KEY_POINTS = CALIBRATION.keyPoints;

// peakAngle/bottomAngle track opposite extremes depending on exercise: bicep
// tracks a MIN (deepest contraction) and a MAX (fullest extension) of the
// elbow angle, so the trackers must start at the opposite ends (180/0).
// Shoulder press tracks a MAX (lockout) and a MIN (depth) of the same pair,
// so it needs the reverse starting values — get this backwards and
// Math.max/min against the wrong bound never fires, silently disabling
// lockout/depth scoring.
const initialState = (isBicep: boolean): SessionState => ({
  reps: 0,
  goodReps: 0,
  badReps: 0,
  stage: 'down',
  activeSide: 'left',
  calibrated: false,
  calibStart: 0,
  calib: { shoulderWidth: 0.2, leftElbowOffset: 0, rightElbowOffset: 0 },
  repQualities: [],
  repLog: [],
  repStartTime: 0,
  repCycleStart: 0,
  repFaults: [],
  repQualityLive: 100,
  peakAngle: isBicep ? 180 : 0,
  bottomAngle: isBicep ? 0 : 180,
  maxDrift: 0,
  maxLean: 0,
  currentSet: 1,
  setHistory: [],
  currentSetReps: [],
  resting: false,
  restStart: 0,
  programComplete: false,
  sessionStart: Date.now(),
  lastVoiceTime: 0,
  voiceCooldown: 2500,
  lastFeedback: '',
});

function angle3(a: Landmark, b: Landmark, c: Landmark) {
  const ax = a.x - b.x;
  const ay = a.y - b.y;
  const cx = c.x - b.x;
  const cy = c.y - b.y;
  const dot = ax * cx + ay * cy;
  const mag = (Math.sqrt(ax * ax + ay * ay) + 1e-9) * (Math.sqrt(cx * cx + cy * cy) + 1e-9);
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

function shoulderWidth(lms: Landmark[]) {
  const ls = lms[11];
  const rs = lms[12];
  if (!ls || !rs || ls.visibility < 0.55 || rs.visibility < 0.55) return 0.2;
  return Math.max(0.05, Math.hypot(ls.x - rs.x, ls.y - rs.y));
}

export default function Workout() {
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const EXERCISE = params.get('exercise') === 'shoulder' ? 'shoulder' : 'bicep';
  const MODE = params.get('mode') === 'program' ? 'program' : 'free';
  const SETS = Number.parseInt(params.get('sets') || '3', 10);
  const REPS = Number.parseInt(params.get('reps') || '10', 10);
  const REST = Number.parseInt(params.get('rest') || '60', 10);
  const VOICE_ON = params.get('voice') !== '0';

  const IS_BICEP = EXERCISE === 'bicep';
  const DEF = useMemo(() => getExercise(EXERCISE), [EXERCISE]);
  const ACCENT = DEF.accent;

  const THRESH = useMemo(
    () => ({
      upIdeal: DEF.thresholds.up,
      downIdeal: DEF.thresholds.down,
      // Bicep curls are single-arm, so bilateral symmetry is not scored;
      // an unreachable threshold disables the check.
      asym: IS_BICEP ? Number.POSITIVE_INFINITY : TOLERANCES.symmetryDeg,
    }),
    [DEF, IS_BICEP],
  );

  const [state, setState] = useState<SessionState>(() => initialState(IS_BICEP));
  const stateRef = useRef<SessionState>(state);
  const [sessionTimer, setSessionTimer] = useState('00:00');
  const [feedback, setFeedbackState] = useState<{ text: string; type: 'good' | 'bad' | 'info' }>({ text: 'Ready', type: 'info' });
  const [calibProgress, setCalibProgress] = useState({ width: 0, text: 'Hold still for 3 seconds' });
  const [camPermissionVisible, setCamPermissionVisible] = useState(true);
  const [camError, setCamError] = useState('');
  const [calibVisible, setCalibVisible] = useState(false);
  const [warningText, setWarningText] = useState('');
  const [showWarning, setShowWarning] = useState(false);
  const [showSummary, setShowSummary] = useState(false);

  const [summary, setSummary] = useState({
    reps: 0,
    quality: 0,
    good: 0,
    fault: '',
  });

  const [restUi, setRestUi] = useState({
    show: false,
    seconds: REST,
    subtitle: `Next: Set 1 of ${SETS}`,
    dashOffset: 377,
  });

  const qualityChartRef = useRef<Chart | null>(null);
  const summaryChartRef = useRef<Chart | null>(null);

  const restRafRef = useRef<number | null>(null);
  const lowVisCounterRef = useRef(0);
  const calibRefPointsRef = useRef<{ x: number; y: number }[] | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const qualityCanvasRef = useRef<HTMLCanvasElement>(null);
  const summaryCanvasRef = useRef<HTMLCanvasElement>(null);

  const updateState = (updater: (prev: SessionState) => SessionState) => {
    setState((prev) => {
      const next = updater(prev);
      stateRef.current = next;
      return next;
    });
  };

  // Mutates stateRef.current directly, WITHOUT going through React setState.
  // Use this for fields nothing in JSX reads directly — the pose callback
  // runs at 30-60Hz, and routing every per-frame update through setState
  // re-renders the whole tree (chart, rep log, sidebar) every frame.
  // completeRep/updateProgramAfterRep still use updateState above, because
  // their fields (reps, currentSet, repLog, ...) genuinely drive the UI —
  // and they only fire once per rep, not once per frame, so there's no
  // storm to fix there.
  const mutateRef = (patch: Partial<SessionState>) => {
    Object.assign(stateRef.current, patch);
  };

  const speak = (text: string, priority = false) => {
    if (!VOICE_ON) return;

    const now = Date.now();
    const snapshot = stateRef.current;
    if (!priority && now - snapshot.lastVoiceTime < snapshot.voiceCooldown) return;
    if (window.speechSynthesis.speaking && !priority) return;
    if (priority) window.speechSynthesis.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.15;
    utter.pitch = 1;
    utter.volume = 0.9;
    window.speechSynthesis.speak(utter);

    mutateRef({ lastVoiceTime: now });
  };

  const setFeedback = (text: string, type: 'good' | 'bad' | 'info') => {
    const snapshot = stateRef.current;
    if (snapshot.lastFeedback === text) return;

    setFeedbackState({ text, type });
    mutateRef({ lastFeedback: text });
  };

  const updateQualityChart = (quality: number) => {
    const chart = qualityChartRef.current;
    if (!chart) return;

    if (!chart.data.labels) chart.data.labels = [];

    const labels = chart.data.labels;
    const nextN = labels.length + 1;
    labels.push(`Rep ${nextN}`);
    chart.data.datasets[0].data.push(quality);
    if (labels.length > 20) {
      labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update();
  };

  const addFault = (text: string, severity: number) => {
    const s = stateRef.current;
    // Only deduct once per rep — the caller re-checks the fault condition
    // every frame while it holds, and without this gate a sustained fault
    // (e.g. elbow drifted for half a second) deducted severity on every one
    // of those frames instead of once, crashing the live score to 0 almost
    // instantly. Matches the Python engine's _fault_seen design.
    if (s.repFaults.includes(text)) return;
    mutateRef({
      repFaults: [...s.repFaults, text],
      repQualityLive: Math.max(0, s.repQualityLive - severity),
    });
  };

  /**
   * Debounce a fault condition: it must hold for faultConfirmFrames
   * consecutive frames before counting. Per-rep extremes are tracked with
   * Math.max/min, which by construction latch onto the worst single frame in
   * the rep, so without this one noisy landmark estimate is enough to fault an
   * otherwise clean rep. Mirrors confirm_fault() in the Python engine.
   */
  const faultStreaksRef = useRef<Record<string, number>>({});

  const confirmFault = (key: string, active: boolean): boolean => {
    if (!active) {
      faultStreaksRef.current[key] = 0;
      return false;
    }
    const streak = (faultStreaksRef.current[key] ?? 0) + 1;
    faultStreaksRef.current[key] = streak;
    return streak >= TOLERANCES.faultConfirmFrames;
  };

  /** Elapsed time for the current rep, covering the full cycle. */
  const repDuration = () => {
    const s = stateRef.current;
    const start = s.repCycleStart || s.repStartTime;
    return start ? Date.now() - start : 0;
  };

  /** First frame the athlete leaves the bottom — the true start of a rep. */
  const markCycleStart = () => {
    if (!stateRef.current.repCycleStart) mutateRef({ repCycleStart: Date.now() });
  };

  const beginRep = () => {
    mutateRef({
      repStartTime: Date.now(),
      repFaults: [],
      repQualityLive: 100,
      peakAngle: IS_BICEP ? 180 : 0,
      bottomAngle: IS_BICEP ? 0 : 180,
      maxDrift: 0,
      maxLean: 0,
    });
    faultStreaksRef.current = {};
  };

  /**
   * Deductions ramp with how far past tolerance the athlete was, rather than
   * dropping the full weight the instant a line is crossed — missing peak
   * contraction by 2° should not cost the same as missing it by 40°.
   */
  const scoreRep = () => {
    const snapshot = stateRef.current;
    let q = 100;

    const w = DEF.scoring;
    const r = DEF.ramps;
    const dur = repDuration();

    if (IS_BICEP) {
      q -= gradedPenalty(snapshot.peakAngle - (THRESH.upIdeal + TOLERANCE), r.peakDeg!, w.range);
      q -= gradedPenalty(
        THRESH.downIdeal - TOLERANCE * 2 - snapshot.bottomAngle,
        r.bottomDeg!,
        Math.round(w.range * 0.5),
      );
      q -= gradedPenalty(snapshot.maxDrift - DRIFT_TOL, r.driftRatio!, w.drift);
      q -= gradedPenalty(
        snapshot.maxLean - DEF.faults.torso.toleranceDeg!,
        DEF.faults.torso.rampDeg!,
        w.torso,
      );
      if (dur < MIN_TEMPO_MS) q -= w.tempo;
    } else {
      q -= gradedPenalty(THRESH.upIdeal - TOLERANCE - snapshot.peakAngle, r.lockoutDeg!, w.lockout);
      q -= gradedPenalty(snapshot.bottomAngle - (THRESH.downIdeal + TOLERANCE), r.depthDeg!, w.depth);
      q -= gradedPenalty(snapshot.maxDrift - (THRESH.asym + TOLERANCE), r.symmetryDeg!, w.symmetry);
      if (dur < MIN_TEMPO_MS) q -= w.tempo;
    }

    return Math.max(0, Math.min(100, q));
  };

  const renderSetPips = () => {
    const snapshot = stateRef.current;
    return Array.from({ length: SETS }, (_, idx) => {
      const i = idx + 1;
      const existing = snapshot.setHistory.find((s) => s.set === i);
      let cls = '';
      if (existing) cls = existing.valid ? 'done' : 'failed';
      else if (i === snapshot.currentSet) cls = 'current';
      return (
        <div key={i} className={`set-pip ${cls}`.trim()}>
          {i}
        </div>
      );
    });
  };

  const animateRest = () => {
    const snapshot = stateRef.current;
    if (!snapshot.resting) return;

    const elapsed = (Date.now() - snapshot.restStart) / 1000;
    const remaining = Math.max(0, REST - elapsed);
    const pct = remaining / REST;

    setRestUi({
      show: true,
      seconds: Math.ceil(remaining),
      subtitle: `Next: Set ${snapshot.currentSet} of ${SETS}`,
      dashOffset: (1 - pct) * 377,
    });

    if (remaining <= 0) {
      mutateRef({ resting: false });
      setRestUi((prev) => ({ ...prev, show: false }));
      setFeedback(`Set ${stateRef.current.currentSet} - ${REPS} reps`, 'info');
      speak(`Rest over. Start set ${stateRef.current.currentSet}.`, true);
      return;
    }

    restRafRef.current = requestAnimationFrame(animateRest);
  };

  const startRest = () => {
    mutateRef({ resting: true, restStart: Date.now() });
    setRestUi((prev) => ({ ...prev, show: true }));
    speak(`Set complete. Rest for ${REST} seconds.`, true);
    restRafRef.current = requestAnimationFrame(animateRest);
  };

  const updateProgramAfterRep = (rep: RepLogItem) => {
    if (MODE !== 'program') return;

    updateState((prev) => {
      const currentSetReps = [...prev.currentSetReps, rep];
      const done = currentSetReps.length;
      const badInSet = currentSetReps.filter((r) => r.quality < 60).length;

      if (badInSet >= 3 && done < REPS) {
        setFeedback('Too many bad reps - focus!', 'bad');
        speak('Too many bad reps. Focus on form.', true);
      }

      if (done < REPS) {
        return { ...prev, currentSetReps };
      }

      const valid = badInSet < 3;
      const setHistory = [...prev.setHistory, { set: prev.currentSet, valid, reps: [...currentSetReps] }];

      if (!valid) {
        setFeedback(`Set ${prev.currentSet} invalid - repeat`, 'bad');
        speak(`Set ${prev.currentSet} does not count. Too many bad reps. Repeat it.`, true);
        return {
          ...prev,
          currentSetReps: [],
          setHistory,
        };
      }

      if (prev.currentSet >= SETS) {
        setFeedback('Program complete!', 'good');
        speak('Program complete. Excellent work.', true);
        setTimeout(() => endSession(), 900);
        return {
          ...prev,
          currentSetReps: [],
          setHistory,
          programComplete: true,
        };
      }

      const nextSet = prev.currentSet + 1;
      setTimeout(() => startRest(), 300);
      return {
        ...prev,
        currentSetReps: [],
        setHistory,
        currentSet: nextSet,
      };
    });
  };

  const voiceForRep = (quality: number) => {
    const snapshot = stateRef.current;
    const n = snapshot.reps;
    if (quality >= 85) speak(`${n}. Great rep!`);
    else if (quality >= 60) speak(`${n}.`);
    else speak(snapshot.repFaults[0] || 'bad form', true);

    if (n > 0 && n % 5 === 0) speak(`${n} reps - keep it up!`);
  };

  const completeRep = () => {
    const quality = Math.min(scoreRep(), stateRef.current.repQualityLive);
    // Built inside the updater below, where prev.repFaults still holds this
    // rep's faults (the same updater resets repFaults to [] for the next
    // rep) and nextReps is the correct just-completed rep number. Reusing
    // this object for updateProgramAfterRep avoids reconstructing it from
    // stateRef.current afterward, which previously read post-reset values —
    // repFaults was already [], and reps had already been incremented, so
    // the reconstructed rep silently had the wrong number and no faults.
    let rep: RepLogItem | null = null;

    updateState((prev) => {
      const nextReps = prev.reps + 1;
      const isGood = quality >= 60;
      rep = { n: nextReps, quality, faults: [...prev.repFaults] };
      return {
        ...prev,
        reps: nextReps,
        goodReps: prev.goodReps + (isGood ? 1 : 0),
        badReps: prev.badReps + (isGood ? 0 : 1),
        repQualities: [...prev.repQualities, quality],
        repLog: [...prev.repLog, rep],
        repFaults: [],
        repQualityLive: 100,
        // Cleared so the next rep is timed from when the lifter leaves the
        // bottom again, not from this rep's start.
        repCycleStart: 0,
        peakAngle: IS_BICEP ? 180 : 0,
        bottomAngle: IS_BICEP ? 0 : 180,
        maxDrift: 0,
        maxLean: 0,
      };
    });

    faultStreaksRef.current = {};
    updateQualityChart(quality);
    voiceForRep(quality);
    if (rep) updateProgramAfterRep(rep);
  };

  const analyzeBicep = (lms: Landmark[], sw: number) => {
    const ls = lms[11];
    const le = lms[13];
    const lw = lms[15];
    const lh = lms[23];
    const rs = lms[12];
    const re = lms[14];
    const rw = lms[16];
    const rh = lms[24];

    const leftAngle = angle3(ls, le, lw);
    const rightAngle = angle3(rs, re, rw);

    // Latch which arm is curling while at rest (not mid-rep) — locking it
    // for the duration of a rep avoids flip-flopping if the two angles
    // happen to cross due to landmark noise once a curl is underway.
    // Tracking only the left arm meant curling with the right arm counted
    // zero reps; calib.rightElbowAnchorX was already captured during
    // calibration but never consumed until now.
    if (stateRef.current.stage !== 'up') {
      mutateRef({ activeSide: leftAngle <= rightAngle ? 'left' : 'right' });
    }
    const active = stateRef.current.activeSide;
    const elbowAngle = active === 'left' ? leftAngle : rightAngle;

    // Drift is the elbow's offset FROM ITS OWN SHOULDER, compared against the
    // same quantity at calibration. Comparing against an absolute calibrated
    // x-coordinate made any whole-body movement register as elbow drift, so a
    // lifter who shifted stance was faulted for an elbow that never left their
    // side. Offset-from-shoulder is invariant to translation; dividing by
    // shoulder width makes it invariant to distance from the camera.
    const elbowOffset = active === 'left' ? (le.x - ls.x) / sw : (re.x - rs.x) / sw;
    const anchorOffset =
      active === 'left'
        ? stateRef.current.calib.leftElbowOffset
        : stateRef.current.calib.rightElbowOffset;
    const drift = Math.abs(elbowOffset - anchorOffset);

    const torsoLean =
      Math.abs(
        Math.atan2(
          Math.abs((ls.x + rs.x) / 2 - (lh.x + rh.x) / 2),
          Math.abs((ls.y + rs.y) / 2 - (lh.y + rh.y) / 2) + 1e-9,
        ),
      ) *
      (180 / Math.PI);

    const s = stateRef.current;
    mutateRef({
      peakAngle: Math.min(s.peakAngle, elbowAngle),
      bottomAngle: Math.max(s.bottomAngle, elbowAngle),
      maxDrift: Math.max(s.maxDrift, drift),
      maxLean: Math.max(s.maxLean, torsoLean),
    });

    if (elbowAngle > THRESH.downIdeal - TOLERANCE) {
      if (stateRef.current.stage === 'up') completeRep();
      mutateRef({ stage: 'down' });
      if (stateRef.current.reps === 0) setFeedback('Curl up - full range', 'info');
      return;
    }

    if (elbowAngle < THRESH.upIdeal + TOLERANCE) {
      if (stateRef.current.stage === 'down') {
        beginRep();
        mutateRef({ stage: 'up' });
      }
      // Debounced: one noisy frame is jitter, not bad form.
      if (confirmFault('drift', drift > DRIFT_TOL)) {
        addFault(DEF.faults.drift.message, DEF.faults.drift.severity);
      }
      if (confirmFault('torso', torsoLean > DEF.faults.torso.toleranceDeg!)) {
        addFault(DEF.faults.torso.message, DEF.faults.torso.severity);
      }
      setFeedback(stateRef.current.repFaults[0] || 'Squeeze at top', stateRef.current.repFaults.length ? 'bad' : 'good');
      return;
    }

    if (stateRef.current.stage === 'down') {
      markCycleStart();
      setFeedback('Curl up', 'info');
    } else setFeedback('Lower slowly', 'info');
  };

  const analyzeShoulder = (lms: Landmark[]) => {
    // Elbow extension (shoulder -> elbow -> wrist), per the shared definition.
    // This used to measure elbow -> shoulder -> hip — a different joint —
    // while applying the same thresholds the Python engine used for the elbow.
    const [lsIdx, leIdx, lwIdx] = DEF.angle.triplets.left;
    const [rsIdx, reIdx, rwIdx] = DEF.angle.triplets.right;

    const leftA = angle3(lms[lsIdx], lms[leIdx], lms[lwIdx]);
    const rightA = angle3(lms[rsIdx], lms[reIdx], lms[rwIdx]);
    const avg = (leftA + rightA) / 2;
    const asym = Math.abs(leftA - rightA);

    const s = stateRef.current;
    mutateRef({
      peakAngle: Math.max(s.peakAngle, avg),
      bottomAngle: Math.min(s.bottomAngle, avg),
      maxDrift: Math.max(s.maxDrift, asym),
    });

    if (avg < THRESH.downIdeal + TOLERANCE) {
      if (stateRef.current.stage === 'up') completeRep();
      mutateRef({ stage: 'down' });
      if (stateRef.current.reps === 0) setFeedback('Press overhead', 'info');
      return;
    }

    if (avg > THRESH.upIdeal - TOLERANCE) {
      if (stateRef.current.stage === 'down') {
        beginRep();
        mutateRef({ stage: 'up' });
      }
      // Debounced: the two arms drift in and out of sync frame to frame, so
      // an instantaneous difference is not an imbalance.
      if (confirmFault('asym', asym > THRESH.asym + TOLERANCE)) {
        addFault(DEF.faults.symmetry.message, DEF.faults.symmetry.severity);
      }
      setFeedback(stateRef.current.repFaults[0] || 'Lockout overhead', stateRef.current.repFaults.length ? 'bad' : 'good');
      return;
    }

    if (stateRef.current.stage === 'down') {
      markCycleStart();
      setFeedback('Press up - full range', 'info');
    } else setFeedback('Lower under control', 'info');
  };

  const checkVisibility = (lms: Landmark[]) => {
    // Both arms are always required now, even for bicep: active-arm
    // detection needs to compare both elbow angles to know which one is
    // curling (see analyzeBicep).
    const required = [11, 12, 13, 14, 15, 16];
    const occluded = required.filter((i) => (lms[i]?.visibility || 0) < 0.55);

    if (occluded.length > 0) lowVisCounterRef.current += 1;
    else lowVisCounterRef.current = 0;

    if (lowVisCounterRef.current >= 12) {
      const names: Record<number, string> = {
        11: 'left shoulder',
        12: 'right shoulder',
        13: 'left elbow',
        14: 'right elbow',
        15: 'left wrist',
        16: 'right wrist',
      };
      setWarningText(`Can't see your ${names[occluded[0]] || 'body'} - adjust camera`);
      setShowWarning(true);
    } else {
      setShowWarning(false);
    }
  };

  const drawSkeleton = (lms: Landmark[], color: string) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const connections = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
      [23, 25], [25, 27], [24, 26], [26, 28],
    ];

    const px = (i: number) => (1 - lms[i].x) * W;
    const py = (i: number) => lms[i].y * H;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.85;

    for (const [a, b] of connections) {
      if (!lms[a] || !lms[b]) continue;
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }

    ctx.fillStyle = color;
    ctx.globalAlpha = 1;
    for (let i = 0; i < lms.length; i += 1) {
      if ((lms[i].visibility || 0) < 0.35) continue;
      ctx.beginPath();
      ctx.arc(px(i), py(i), 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  };

  const onPoseResults = (_worldLandmarks: PoseLandmark[], normalizedLandmarks: PoseLandmark[]) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!video || !canvas || !ctx) return;

    canvas.width = video.videoWidth || canvas.offsetWidth;
    canvas.height = video.videoHeight || canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // <video> is mirrored via CSS (see Workout.css); landmark coordinates are
    // relative to the unmirrored source frame regardless of on-screen
    // presentation, so drawSkeleton's manual x-flip below is still correct.
    const lms: Landmark[] = normalizedLandmarks.map((l) => ({
      x: l.x, y: l.y, z: l.z, visibility: l.visibility ?? 0,
    }));

    if (!stateRef.current.calibrated) {
      const snapshot = stateRef.current;
      if (!snapshot.calibStart) {
        mutateRef({ calibStart: Date.now() });
        calibRefPointsRef.current = null;
      }

      // Require the pose to actually be held still — reset the countdown if
      // the user moves more than a small amount, instead of calibrating
      // against whatever pose happens to be held at the 3-second mark (e.g.
      // mid-curl), which would silently corrupt every drift measurement for
      // the rest of the session.
      const keyPoints = CALIB_KEY_POINTS.map((i) => ({ x: lms[i].x, y: lms[i].y }));
      if (calibRefPointsRef.current) {
        const moved = keyPoints.some((p, idx) => {
          const ref = calibRefPointsRef.current![idx];
          return Math.hypot(p.x - ref.x, p.y - ref.y) > CALIB_STABILITY_TOL;
        });
        if (moved) mutateRef({ calibStart: Date.now() });
      }
      calibRefPointsRef.current = keyPoints;

      const elapsed = Date.now() - (stateRef.current.calibStart || Date.now());
      const pct = Math.min(100, (elapsed / CALIB_HOLD_MS) * 100);
      setCalibProgress({
        width: pct,
        text: `Hold still - ${((CALIB_HOLD_MS - elapsed) / 1000).toFixed(1)}s`,
      });

      if (elapsed >= CALIB_HOLD_MS) {
        const sw = shoulderWidth(lms);
        mutateRef({
          calibrated: true,
          calib: {
            shoulderWidth: sw,
            // Stored relative to each shoulder so drift survives the lifter
            // shifting position mid-set.
            leftElbowOffset: (lms[13].x - lms[11].x) / sw,
            rightElbowOffset: (lms[14].x - lms[12].x) / sw,
          },
        });
        setCalibVisible(false);
        setFeedback('Calibrated! Begin when ready.', 'good');
        speak('Calibrated. Begin when you are ready.', true);
      }

      drawSkeleton(lms, '#6b6b7e');
      return;
    }

    if (stateRef.current.resting) {
      drawSkeleton(lms, '#6b6b7e');
      return;
    }

    checkVisibility(lms);
    if (IS_BICEP) analyzeBicep(lms, stateRef.current.calib.shoulderWidth);
    else analyzeShoulder(lms);

    drawSkeleton(lms, stateRef.current.repQualityLive > 60 ? ACCENT : '#f87171');
  };

  const poseDetection = usePoseDetection({ onResults: onPoseResults });

  const startCamera = async () => {
    const video = videoRef.current;
    if (!video) return;

    setCamError('');

    // Reuse the existing landmarker across restarts instead of rebuilding
    // the whole detection graph every time — init() loads the wasm runtime
    // and model and only needs to run once per page load, not once per
    // session (the legacy MediaPipe Solutions setup used to reconstruct
    // everything on every restart, leaking a graph each time).
    const initPromise = poseDetection.landmarkerRef.current
      ? Promise.resolve()
      : poseDetection.init();
    const [ok] = await Promise.all([poseDetection.startCamera(video), initPromise]);

    if (ok) {
      setCamPermissionVisible(false);
      setCalibVisible(true);
    } else {
      // Leave the permission prompt visible so the user can retry — don't
      // hide it before we know the camera actually started — but now say
      // why, instead of silently doing nothing.
      setCamError('Camera access failed. Check your browser’s site permissions (or that no other app is using the camera) and try again.');
    }
  };

  const endSession = () => {
    poseDetection.stopCamera();
    if (window.speechSynthesis) window.speechSynthesis.cancel();

    const qualities = stateRef.current.repQualities;
    const avg = qualities.length > 0 ? Math.round(qualities.reduce((a, b) => a + b, 0) / qualities.length) : 0;

    const allFaults = stateRef.current.repLog.flatMap((r) => r.faults);
    const faultCounts = allFaults.reduce<Record<string, number>>((acc, fault) => {
      acc[fault] = (acc[fault] || 0) + 1;
      return acc;
    }, {});
    const topFault = Object.entries(faultCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '';

    setSummary({
      reps: stateRef.current.reps,
      quality: avg,
      good: stateRef.current.goodReps,
      fault: topFault,
    });

    // Persist the full session to history. Fire-and-forget: a storage failure
    // must never block the results screen.
    if (stateRef.current.reps > 0) {
      void recordSession({
        id: newSessionId(),
        date: new Date().toISOString(),
        exercise: EXERCISE,
        mode: MODE,
        totalReps: stateRef.current.reps,
        goodReps: stateRef.current.goodReps,
        badReps: stateRef.current.badReps,
        avgQuality: avg,
        topFault: topFault || null,
        durationSec: Math.round((Date.now() - stateRef.current.sessionStart) / 1000),
        repQualities: [...qualities],
        ...(MODE === 'program'
          ? {
              setsCompleted: stateRef.current.setHistory.filter((s) => s.valid).length,
              setsPlanned: SETS,
            }
          : {}),
      });
    }

    setShowSummary(true);
  };

  const resetFullSession = () => {
    const fresh = initialState(IS_BICEP);
    setState(fresh);
    stateRef.current = fresh;
    setFeedbackState({ text: 'Ready', type: 'info' });
    setWarningText('');
    setShowWarning(false);
    setCamError('');
    setSessionTimer('00:00');
    calibRefPointsRef.current = null;
    qualityChartRef.current?.destroy();
    qualityChartRef.current = null;
    if (qualityCanvasRef.current) {
      qualityChartRef.current = new Chart(qualityCanvasRef.current, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{ data: [], borderColor: ACCENT, borderWidth: 2, tension: 0.35, pointRadius: 0 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: { min: 0, max: 100, ticks: { color: '#6b6b7e' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          },
          animation: { duration: 250 },
        },
      });
    }
  };

  useEffect(() => {
    document.documentElement.style.setProperty('--accent-ex', ACCENT);

    if (qualityCanvasRef.current) {
      qualityChartRef.current = new Chart(qualityCanvasRef.current, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{ data: [], borderColor: ACCENT, borderWidth: 2, tension: 0.35, pointRadius: 0 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: { min: 0, max: 100, ticks: { color: '#6b6b7e' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          },
          animation: { duration: 250 },
        },
      });
    }

    const timerId = window.setInterval(() => {
      const s = Math.floor((Date.now() - stateRef.current.sessionStart) / 1000);
      const mm = String(Math.floor(s / 60)).padStart(2, '0');
      const ss = String(s % 60).padStart(2, '0');
      setSessionTimer(`${mm}:${ss}`);
    }, 1000);

    const boot = window.setTimeout(() => {
      void startCamera();
    }, 300);

    return () => {
      window.clearInterval(timerId);
      window.clearTimeout(boot);
      if (restRafRef.current) cancelAnimationFrame(restRafRef.current);
      poseDetection.stopCamera();
      qualityChartRef.current?.destroy();
      summaryChartRef.current?.destroy();
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ACCENT]);

  useEffect(() => {
    if (!showSummary || !summaryCanvasRef.current) return;

    summaryChartRef.current?.destroy();
    summaryChartRef.current = new Chart(summaryCanvasRef.current, {
      type: 'line',
      data: {
        labels: state.repQualities.map((_, idx) => `${idx + 1}`),
        datasets: [
          {
            data: state.repQualities,
            borderColor: ACCENT,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { min: 0, max: 100, ticks: { color: '#6b6b7e' }, grid: { color: 'rgba(255,255,255,0.07)' } },
        },
      },
    });
  }, [showSummary, state.repQualities, ACCENT]);

  const sessionTitle = IS_BICEP ? 'Bicep Curl' : 'Shoulder Press';
  const sessionBadge = MODE === 'program' ? `${SETS}x${REPS}` : 'FREE';
  const quality = state.repQualities[state.repQualities.length - 1] ?? 0;

  return (
    <div className="session-root">
      <div className="session-layout">
        <div className="topbar">
          <div className="topbar-left">
            <Link className="back-btn" to="/">← Home</Link>
            <div className="session-title" id="session-title">{sessionTitle}</div>
            <div className="session-badge" id="session-badge">{sessionBadge}</div>
          </div>
          <div className="topbar-right">
            <div className="session-timer">{sessionTimer}</div>
            <button type="button" className="end-btn" onClick={endSession}>End Session</button>
          </div>
        </div>

        <div className="camera-panel">
          <video ref={videoRef} id="video-el" playsInline />
          <canvas ref={canvasRef} id="pose-canvas" />

          <div className={`cam-permission ${camPermissionVisible ? '' : 'hidden'}`}>
            <div className="cam-icon">📷</div>
            <h3>Allow camera access</h3>
            <p>We need your webcam to analyze movement and give rep-by-rep feedback.</p>
            {camError && <p className="cam-error">{camError}</p>}
            <button type="button" className="cam-btn" onClick={() => void startCamera()}>Enable Camera</button>
          </div>

          <div className={`calib-overlay ${calibVisible ? '' : 'hidden'}`}>
            <div className="calib-icon">🧭</div>
            <h3>Calibrating your baseline</h3>
            <p>Stand still in frame with your upper body visible. Hold for 3 seconds.</p>
            <div className="calib-progress">
              <div className="calib-bar" style={{ width: `${calibProgress.width}%` }} />
            </div>
            <p id="calib-text">{calibProgress.text}</p>
          </div>

          <div className={`rest-overlay ${restUi.show ? 'show' : ''}`}>
            <div className="rest-label">Rest</div>
            <div className="rest-ring">
              <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="60" stroke="rgba(255,255,255,0.12)" strokeWidth="8" fill="none" />
                <circle
                  id="rest-ring-circle"
                  cx="70"
                  cy="70"
                  r="60"
                  stroke="var(--accent-ex)"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray="377"
                  strokeDashoffset={restUi.dashOffset}
                />
              </svg>
              <div className="rest-ring-num" id="rest-num">{restUi.seconds}</div>
            </div>
            <div className="rest-sub" id="rest-sub">{restUi.subtitle}</div>
          </div>

          <div className={`vis-warning ${showWarning ? 'show' : ''}`}>
            <span>⚠</span>
            <span id="vis-warn-text">{warningText}</span>
          </div>

          <div className={`feedback-pill ${feedback.type}`} id="feedback-pill">{feedback.text}</div>
        </div>

        <div className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-label">Live Metrics</div>
            <div className="metrics-grid">
              <div className="metric-box">
                <div className="metric-label">Reps</div>
                <div className="metric-value">{state.reps}</div>
                <div className="metric-sub">total</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Good</div>
                <div className="metric-value good">{state.goodReps}</div>
                <div className="metric-sub">quality &gt;= 60</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Bad</div>
                <div className="metric-value danger">{state.badReps}</div>
                <div className="metric-sub">quality &lt; 60</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Last Q</div>
                <div className={`metric-value ${quality >= 80 ? 'good' : quality >= 60 ? 'warn' : 'danger'}`}>{quality}</div>
                <div className="metric-sub">out of 100</div>
              </div>
            </div>
          </div>

          {MODE === 'program' && (
            <div className="sidebar-section" id="program-tracker">
              <div className="sidebar-label">Program</div>
              <div className="set-tracker" id="set-tracker">{renderSetPips()}</div>
              <div className="metric-sub" id="program-status">Set {state.currentSet} of {SETS} - {REPS} reps</div>
            </div>
          )}

          <div className="sidebar-section">
            <div className="sidebar-label">Rep Quality Trend</div>
            <div className="chart-wrap">
              <canvas ref={qualityCanvasRef} id="quality-chart" />
            </div>
          </div>

          <div className="sidebar-section" style={{ flex: 1 }}>
            <div className="sidebar-label">Rep Log</div>
            <div className="rep-log" id="rep-log">
              {state.repLog.length === 0 ? (
                <div className="rep-empty">No reps yet</div>
              ) : (
                [...state.repLog]
                  .reverse()
                  .map((rep) => (
                    <div key={rep.n} className="rep-row">
                      <span className="rep-num">Rep {rep.n}</span>
                      <span className="rep-fault">{rep.faults[0] || ''}</span>
                      <span className={`rep-score ${rep.quality >= 80 ? 'great' : rep.quality >= 60 ? 'ok' : 'bad'}`}>{rep.quality}</span>
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>
      </div>

      <div className={`summary-backdrop ${showSummary ? 'show' : ''}`}>
        <div className="summary-card">
          <div className="summary-eyebrow">Session complete</div>
          <h2 id="sum-title">{sessionTitle} - Summary</h2>

          <div className="summary-stats">
            <div className="sumstat">
              <div className="sumstat-val" id="sum-reps">{summary.reps}</div>
              <div className="sumstat-lbl">Total Reps</div>
            </div>
            <div className="sumstat">
              <div className="sumstat-val" id="sum-quality">{summary.quality}</div>
              <div className="sumstat-lbl">Avg Quality</div>
            </div>
            <div className="sumstat">
              <div className="sumstat-val" id="sum-good">{summary.good}</div>
              <div className="sumstat-lbl">Good Reps</div>
            </div>
          </div>

          <div className="summary-chart-wrap">
            <canvas ref={summaryCanvasRef} id="summary-chart" />
          </div>

          <div className="summary-fault" id="sum-fault-box" style={{ display: summary.fault ? 'block' : 'none' }}>
            <strong>TOP FAULT</strong>
            <span id="sum-fault-text">{summary.fault}</span>
          </div>

          <div className="summary-actions">
            <button
              type="button"
              className="summary-btn secondary"
              onClick={() => {
                setShowSummary(false);
                void startCamera();
                resetFullSession();
              }}
            >
              Restart
            </button>
            <button type="button" className="summary-btn primary" onClick={() => navigate('/')}>Back Home</button>
          </div>
        </div>
      </div>
    </div>
  );
}
