import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Chart } from 'chart.js/auto';
import '@mediapipe/camera_utils';
import '@mediapipe/pose';
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
  calibrated: boolean;
  calibStart: number;
  calib: {
    shoulderWidth: number;
    leftElbowAnchorX: number;
    rightElbowAnchorX: number;
  };
  repQualities: number[];
  repLog: RepLogItem[];
  repStartTime: number;
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

const TOLERANCE = 15;
const DRIFT_TOL = 0.18;
const MIN_TEMPO_MS = 800;

const initialState = (): SessionState => ({
  reps: 0,
  goodReps: 0,
  badReps: 0,
  stage: 'down',
  calibrated: false,
  calibStart: 0,
  calib: { shoulderWidth: 0.2, leftElbowAnchorX: 0.5, rightElbowAnchorX: 0.5 },
  repQualities: [],
  repLog: [],
  repStartTime: 0,
  repFaults: [],
  repQualityLive: 100,
  peakAngle: 180,
  bottomAngle: 0,
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
  const ACCENT = IS_BICEP ? '#4ade80' : '#22d3ee';

  const THRESH = useMemo(
    () => (IS_BICEP ? { upIdeal: 45, downIdeal: 160, asym: 999 } : { upIdeal: 165, downIdeal: 75, asym: 20 }),
    [IS_BICEP],
  );

  const [state, setState] = useState<SessionState>(initialState);
  const stateRef = useRef<SessionState>(state);
  const [sessionTimer, setSessionTimer] = useState('00:00');
  const [feedback, setFeedbackState] = useState<{ text: string; type: 'good' | 'bad' | 'info' }>({ text: 'Ready', type: 'info' });
  const [calibProgress, setCalibProgress] = useState({ width: 0, text: 'Hold still for 3 seconds' });
  const [camPermissionVisible, setCamPermissionVisible] = useState(true);
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

  const poseRef = useRef<{ send: (input: { image: HTMLVideoElement }) => Promise<void> } | null>(null);
  const cameraRef = useRef<{ start: () => Promise<void>; stop: () => void } | null>(null);
  const restRafRef = useRef<number | null>(null);
  const lowVisCounterRef = useRef(0);

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

    updateState((prev) => ({ ...prev, lastVoiceTime: now }));
  };

  const setFeedback = (text: string, type: 'good' | 'bad' | 'info') => {
    const snapshot = stateRef.current;
    if (snapshot.lastFeedback === text) return;

    setFeedbackState({ text, type });
    updateState((prev) => ({ ...prev, lastFeedback: text }));
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
    updateState((prev) => {
      const faults = prev.repFaults.includes(text) ? prev.repFaults : [...prev.repFaults, text];
      return {
        ...prev,
        repFaults: faults,
        repQualityLive: Math.max(0, prev.repQualityLive - severity),
      };
    });
  };

  const beginRep = () => {
    updateState((prev) => ({
      ...prev,
      repStartTime: Date.now(),
      repFaults: [],
      repQualityLive: 100,
      peakAngle: 180,
      bottomAngle: 0,
      maxDrift: 0,
      maxLean: 0,
    }));
  };

  const scoreRep = () => {
    const snapshot = stateRef.current;
    let q = 100;

    if (IS_BICEP) {
      if (snapshot.peakAngle > THRESH.upIdeal + TOLERANCE) q -= 40;
      if (snapshot.bottomAngle < THRESH.downIdeal - TOLERANCE * 2) q -= 20;
      if (snapshot.maxDrift > DRIFT_TOL) {
        q -= Math.min(20, Math.round((20 * (snapshot.maxDrift - DRIFT_TOL)) / DRIFT_TOL));
      }
      if (snapshot.maxLean > 20 + TOLERANCE) q -= 15;
      const dur = (Date.now() - snapshot.repStartTime) / 1000;
      if (dur < MIN_TEMPO_MS / 1000) q -= 25;
    } else {
      if (snapshot.peakAngle < THRESH.upIdeal - TOLERANCE) q -= 35;
      if (snapshot.bottomAngle > THRESH.downIdeal + TOLERANCE) q -= 30;
      if (snapshot.maxDrift > THRESH.asym + TOLERANCE) {
        q -= Math.min(20, Math.round((20 * (snapshot.maxDrift - THRESH.asym)) / THRESH.asym));
      }
      const dur = (Date.now() - snapshot.repStartTime) / 1000;
      if (dur < MIN_TEMPO_MS / 1000) q -= 15;
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
      updateState((prev) => ({ ...prev, resting: false }));
      setRestUi((prev) => ({ ...prev, show: false }));
      setFeedback(`Set ${stateRef.current.currentSet} - ${REPS} reps`, 'info');
      speak(`Rest over. Start set ${stateRef.current.currentSet}.`, true);
      return;
    }

    restRafRef.current = requestAnimationFrame(animateRest);
  };

  const startRest = () => {
    updateState((prev) => ({ ...prev, resting: true, restStart: Date.now() }));
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

    updateState((prev) => {
      const nextReps = prev.reps + 1;
      const isGood = quality >= 60;
      const rep: RepLogItem = { n: nextReps, quality, faults: [...prev.repFaults] };
      const next = {
        ...prev,
        reps: nextReps,
        goodReps: prev.goodReps + (isGood ? 1 : 0),
        badReps: prev.badReps + (isGood ? 0 : 1),
        repQualities: [...prev.repQualities, quality],
        repLog: [...prev.repLog, rep],
        repFaults: [],
        repQualityLive: 100,
        peakAngle: 180,
        bottomAngle: 0,
        maxDrift: 0,
        maxLean: 0,
      };

      return next;
    });

    updateQualityChart(quality);
    voiceForRep(quality);
    updateProgramAfterRep({ n: stateRef.current.reps + 1, quality, faults: [...stateRef.current.repFaults] });
  };

  const analyzeBicep = (lms: Landmark[], sw: number) => {
    const ls = lms[11];
    const le = lms[13];
    const lw = lms[15];
    const lh = lms[23];
    const rs = lms[12];
    const rh = lms[24];

    const elbowAngle = angle3(ls, le, lw);
    const drift = Math.abs(le.x - stateRef.current.calib.leftElbowAnchorX) / sw;
    const torsoLean =
      Math.abs(
        Math.atan2(
          Math.abs((ls.x + rs.x) / 2 - (lh.x + rh.x) / 2),
          Math.abs((ls.y + rs.y) / 2 - (lh.y + rh.y) / 2) + 1e-9,
        ),
      ) *
      (180 / Math.PI);

    updateState((prev) => ({
      ...prev,
      peakAngle: Math.min(prev.peakAngle, elbowAngle),
      bottomAngle: Math.max(prev.bottomAngle, elbowAngle),
      maxDrift: Math.max(prev.maxDrift, drift),
      maxLean: Math.max(prev.maxLean, torsoLean),
    }));

    if (elbowAngle > THRESH.downIdeal - TOLERANCE) {
      if (stateRef.current.stage === 'up') completeRep();
      updateState((prev) => ({ ...prev, stage: 'down' }));
      if (stateRef.current.reps === 0) setFeedback('Curl up - full range', 'info');
      return;
    }

    if (elbowAngle < THRESH.upIdeal + TOLERANCE) {
      if (stateRef.current.stage === 'down') {
        beginRep();
        updateState((prev) => ({ ...prev, stage: 'up' }));
      }
      if (drift > DRIFT_TOL) addFault('Elbow drifting forward', 18);
      if (torsoLean > 20 + TOLERANCE) addFault('Too much torso swing', 15);
      setFeedback(stateRef.current.repFaults[0] || 'Squeeze at top', stateRef.current.repFaults.length ? 'bad' : 'good');
      return;
    }

    if (stateRef.current.stage === 'down') setFeedback('Curl up', 'info');
    else setFeedback('Lower slowly', 'info');
  };

  const analyzeShoulder = (lms: Landmark[]) => {
    const ls = lms[11];
    const rs = lms[12];
    const le = lms[13];
    const re = lms[14];
    const lh = lms[23];
    const rh = lms[24];

    const leftA = angle3(le, ls, lh);
    const rightA = angle3(re, rs, rh);
    const avg = (leftA + rightA) / 2;
    const asym = Math.abs(leftA - rightA);

    updateState((prev) => ({
      ...prev,
      peakAngle: Math.max(prev.peakAngle, avg),
      bottomAngle: Math.min(prev.bottomAngle, avg),
      maxDrift: Math.max(prev.maxDrift, asym),
    }));

    if (avg < THRESH.downIdeal + TOLERANCE) {
      if (stateRef.current.stage === 'up') completeRep();
      updateState((prev) => ({ ...prev, stage: 'down' }));
      if (stateRef.current.reps === 0) setFeedback('Press overhead', 'info');
      return;
    }

    if (avg > THRESH.upIdeal - TOLERANCE) {
      if (stateRef.current.stage === 'down') {
        beginRep();
        updateState((prev) => ({ ...prev, stage: 'up' }));
      }
      if (asym > THRESH.asym + TOLERANCE) addFault('Arms are uneven', 18);
      setFeedback(stateRef.current.repFaults[0] || 'Lockout overhead', stateRef.current.repFaults.length ? 'bad' : 'good');
      return;
    }

    if (stateRef.current.stage === 'down') setFeedback('Press up - full range', 'info');
    else setFeedback('Lower under control', 'info');
  };

  const checkVisibility = (lms: Landmark[]) => {
    const required = IS_BICEP ? [11, 13, 15] : [11, 12, 13, 14, 15, 16];
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

  const onPoseResults = (results: { image: CanvasImageSource; poseLandmarks?: unknown }) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!video || !canvas || !ctx) return;

    canvas.width = video.videoWidth || canvas.offsetWidth;
    canvas.height = video.videoHeight || canvas.offsetHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (!results.poseLandmarks) return;
    const lms = results.poseLandmarks as Landmark[];

    if (!stateRef.current.calibrated) {
      const snapshot = stateRef.current;
      if (!snapshot.calibStart) {
        updateState((prev) => ({ ...prev, calibStart: Date.now() }));
      }
      const elapsed = Date.now() - (stateRef.current.calibStart || Date.now());
      const pct = Math.min(100, (elapsed / 3000) * 100);
      setCalibProgress({ width: pct, text: `Hold still - ${((3000 - elapsed) / 1000).toFixed(1)}s` });

      if (elapsed >= 3000) {
        updateState((prev) => ({
          ...prev,
          calibrated: true,
          calib: {
            shoulderWidth: shoulderWidth(lms),
            leftElbowAnchorX: lms[13].x,
            rightElbowAnchorX: lms[14].x,
          },
        }));
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

  const startCamera = async () => {
    const video = videoRef.current;
    if (!video) return;

    setCamPermissionVisible(false);
    setCalibVisible(true);

    const PoseCtor = (window as unknown as { Pose: new (options: { locateFile: (file: string) => string }) => {
      setOptions: (options: Record<string, unknown>) => void;
      onResults: (cb: (results: { image: CanvasImageSource; poseLandmarks?: unknown }) => void) => void;
      send: (input: { image: HTMLVideoElement }) => Promise<void>;
    } }).Pose;
    const CameraCtor = (window as unknown as { Camera: new (video: HTMLVideoElement, options: {
      onFrame: () => Promise<void>;
      width: number;
      height: number;
    }) => { start: () => Promise<void>; stop: () => void } }).Camera;

    const pose = new PoseCtor({ locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}` });
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults(onPoseResults);
    poseRef.current = pose;

    const camera = new CameraCtor(video, {
      onFrame: async () => {
        if (!poseRef.current || !videoRef.current) return;
        await poseRef.current.send({ image: videoRef.current });
      },
      width: 1280,
      height: 720,
    });

    cameraRef.current = camera;
    await camera.start();
  };

  const endSession = () => {
    cameraRef.current?.stop();
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

    const payload = {
      total_reps: stateRef.current.reps,
      avg_quality: avg,
      good_reps: stateRef.current.goodReps,
      most_common_fault: topFault || 'None',
      exercise: EXERCISE,
      mode: MODE,
      date: new Date().toISOString(),
    };
    localStorage.setItem('mavis_last_session', JSON.stringify(payload));

    setShowSummary(true);
  };

  const resetFullSession = () => {
    const fresh = initialState();
    setState(fresh);
    stateRef.current = fresh;
    setFeedbackState({ text: 'Ready', type: 'info' });
    setWarningText('');
    setShowWarning(false);
    setSessionTimer('00:00');
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
      cameraRef.current?.stop();
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
