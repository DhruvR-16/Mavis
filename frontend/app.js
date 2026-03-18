// Mavis Frontend Logic — Bicep Curl Analyzer
// Uses MediaPipe Pose for real-time form detection

// ─── Configuration ───────────────────────────────────────────
const INACTIVITY_LIMIT_MS = 60000;

// Thresholds
const UP_THRESHOLD = 50;
const DOWN_THRESHOLD = 155;
const MIN_REP_TIME_MS = 800;
const ELBOW_DRIFT_TOLERANCE = 0.08;

// ─── State ───────────────────────────────────────────────────
let inactivityTimer;
let stream = null;
let isActive = true;
let poseReady = false;

// Bicep Curl State
const Stage = { DOWN: "DOWN", UP: "UP" };
let stage = Stage.DOWN;
let repCount = 0;
let badRepCount = 0;
let feedback = "Start Curls";
let cueTitle = "Positioning";
let feedbackType = "info"; // "good", "bad", "info", "neutral"
let elbowAngle = 0;
let repStartTime = 0;
let anchorElbowX = 0;
let currentFormColor = "#1D4ED8"; // Accent blue by default

// Active side detection
let activeSide = "left";
let lastLeftAngle = 180;
let lastRightAngle = 180;

// Recording Timer
let sessionStartTime = null;
let timerInterval = null;

// ─── DOM Elements ────────────────────────────────────────────
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("pose-canvas");
const ctx = canvasEl.getContext("2d");
const overlayEl = document.getElementById("timeout-overlay");
const resumeBtn = document.getElementById("resume-btn");
const cameraToggleBtn = document.getElementById("camera-toggle-btn");
const loadingEl = document.getElementById("loading-overlay");

// Top-left Overlays
const recTimeEl = document.getElementById("rec-time");
const recDotEl = document.getElementById("rec-dot");
const titleTagEl = document.getElementById("exercise-title-tag");

// Bottom-left Overlays
const repsCount = document.getElementById("reps-count");
const badRepsCount = document.getElementById("bad-reps-count");
const durationVal = document.getElementById("duration-val");

// Sidebar Feedbacks
const masterBox = document.getElementById("master-feedback-box");
const masterText = document.getElementById("master-feedback-text");
const cueIcon = document.getElementById("cue-icon");
const cueTitleEl = document.getElementById("cue-title");
const cueDescEl = document.getElementById("cue-desc");
const angleText = document.getElementById("angle-text");
const stageText = document.getElementById("stage-text");

if (cameraToggleBtn) {
  cameraToggleBtn.addEventListener("click", toggleCamera);
}

// ─── Session Timer ──────────────────────────────────────────
function formatTime(ms) {
  const totalSecs = Math.floor(ms / 1000);
  const h = Math.floor(totalSecs / 3600);
  const m = Math.floor((totalSecs % 3600) / 60);
  const s = totalSecs % 60;
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function startTimer() {
  if (!sessionStartTime) sessionStartTime = Date.now();
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    if (isActive) {
      const elapsed = Date.now() - sessionStartTime;
      recTimeEl.textContent = formatTime(elapsed);
      durationVal.textContent = Math.floor(elapsed / 1000) + "s";
      recDotEl.style.opacity = (Math.floor(elapsed / 500) % 2 === 0) ? "1" : "0.3"; // Blink
    }
  }, 500);
}

// ─── Angle Calculation ──────────────────────────────────────
function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

const LM = { LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_ELBOW: 13, RIGHT_ELBOW: 14, LEFT_WRIST: 15, RIGHT_WRIST: 16, LEFT_HIP: 23, RIGHT_HIP: 24 };
function lm(landmarks, idx) { return landmarks[idx]; }
function isLandmarkVisible(l, threshold = 0.5) { return l && l.visibility > threshold; }

function detectActiveSide(landmarks) {
  const lShoulder = lm(landmarks, LM.LEFT_SHOULDER); const lElbow = lm(landmarks, LM.LEFT_ELBOW); const lWrist = lm(landmarks, LM.LEFT_WRIST);
  const rShoulder = lm(landmarks, LM.RIGHT_SHOULDER); const rElbow = lm(landmarks, LM.RIGHT_ELBOW); const rWrist = lm(landmarks, LM.RIGHT_WRIST);
  const leftVisible = isLandmarkVisible(lShoulder) && isLandmarkVisible(lElbow) && isLandmarkVisible(lWrist);
  const rightVisible = isLandmarkVisible(rShoulder) && isLandmarkVisible(rElbow) && isLandmarkVisible(rWrist);

  if (leftVisible && rightVisible) {
    const leftAngle = calculateAngle(lShoulder, lElbow, lWrist);
    const rightAngle = calculateAngle(rShoulder, rElbow, rWrist);
    activeSide = leftAngle < rightAngle ? "left" : "right";
  } else if (leftVisible) activeSide = "left";
  else if (rightVisible) activeSide = "right";
  return activeSide;
}

function getActiveKeypoints(landmarks) {
  const side = detectActiveSide(landmarks);
  if (side === "left") {
    return { shoulder: lm(landmarks, LM.LEFT_SHOULDER), elbow: lm(landmarks, LM.LEFT_ELBOW), wrist: lm(landmarks, LM.LEFT_WRIST), hip: lm(landmarks, LM.LEFT_HIP) };
  } else {
    return { shoulder: lm(landmarks, LM.RIGHT_SHOULDER), elbow: lm(landmarks, LM.RIGHT_ELBOW), wrist: lm(landmarks, LM.RIGHT_WRIST), hip: lm(landmarks, LM.RIGHT_HIP) };
  }
}

// ─── Form Analysis ──────────────────────────────────────────
function analyzeForm(landmarks) {
  const kp = getActiveKeypoints(landmarks);
  const { shoulder, elbow, wrist, hip } = kp;

  if (!isLandmarkVisible(shoulder) || !isLandmarkVisible(elbow) || !isLandmarkVisible(wrist) || !isLandmarkVisible(hip)) {
    cueTitle = "Positioning"; feedback = "Position yourself in frame"; feedbackType = "neutral";
    return;
  }

  elbowAngle = calculateAngle(shoulder, elbow, wrist);
  const shoulderAngle = calculateAngle(elbow, shoulder, hip);
  const now = Date.now();

  // ── State Machine ──
  if (elbowAngle > DOWN_THRESHOLD) {
    if (stage === Stage.UP) {
      const duration = now - repStartTime;
      if (duration > MIN_REP_TIME_MS) {
        repCount++; cueTitle = "Correct Form"; feedback = "Excellent Rep!"; feedbackType = "good"; currentFormColor = "#10B981"; // success
      } else {
        badRepCount++; cueTitle = "Tempo Control"; feedback = "Too Fast! Slow down"; feedbackType = "bad"; currentFormColor = "#EF4444"; // error
      }
      stage = Stage.DOWN;
    } else {
      currentFormColor = "#1D4ED8"; // ready
      if (feedback === "Start Curls" || feedback === "Excellent Rep!" || feedback === "Too Fast! Slow down") {
        cueTitle = "Ready"; feedback = "Curl upwards!"; feedbackType = "info";
      }
      anchorElbowX = elbow.x;
    }
  } else if (elbowAngle < UP_THRESHOLD) {
    if (stage === Stage.DOWN) {
      stage = Stage.UP; repStartTime = now; anchorElbowX = elbow.x;
      cueTitle = "Contraction"; feedback = "Squeeze the bicep!"; feedbackType = "good"; currentFormColor = "#10B981";
    }
    const drift = Math.abs(elbow.x - anchorElbowX);
    if (drift > ELBOW_DRIFT_TOLERANCE) {
      cueTitle = "Shoulder Position"; feedback = "Fix Elbow! Keep it pinned."; feedbackType = "bad"; currentFormColor = "#EF4444";
    }
  } else {
    if (stage === Stage.UP) {
      if (shoulderAngle > 45) {
        cueTitle = "Torso Swing"; feedback = "Don't swing back!"; feedbackType = "bad"; currentFormColor = "#EF4444";
      } else {
        cueTitle = "Eccentric Phase"; feedback = "Lower slowly..."; feedbackType = "info"; currentFormColor = "#1D4ED8";
      }
    } else if (stage === Stage.DOWN) {
      if (elbowAngle < DOWN_THRESHOLD - 10) {
        cueTitle = "Concentric Phase"; feedback = "Keep going up!"; feedbackType = "info"; currentFormColor = "#1D4ED8";
      }
    }
  }

  if (stage === Stage.DOWN && elbowAngle < DOWN_THRESHOLD && elbowAngle > UP_THRESHOLD && elbowAngle < 120) {
    cueTitle = "Range of Motion"; feedback = "Extend arm fully!"; feedbackType = "info";
  }
}

// ─── Drawing ────────────────────────────────────────────────
const POSE_CONNECTIONS_BODY = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]
];

function drawSkeleton(landmarks, width, height) {
  const activeIndices = activeSide === "left" ? [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST] : [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST];
  for (const [i, j] of POSE_CONNECTIONS_BODY) {
    const a = landmarks[i]; const b = landmarks[j];
    if (!a || !b || a.visibility < 0.4 || b.visibility < 0.4) continue;
    const isActiveArm = (activeIndices.includes(i) || activeIndices.includes(j));
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height); ctx.lineTo(b.x * width, b.y * height);
    ctx.strokeStyle = isActiveArm ? currentFormColor : "rgba(255,255,255,0.15)";
    ctx.lineWidth = isActiveArm ? 5 : 2;
    ctx.stroke();
  }
  for (let i = 0; i <= 28; i++) {
    const l = landmarks[i];
    if (!l || l.visibility < 0.4) continue;
    const isActive = activeIndices.includes(i);
    ctx.beginPath();
    ctx.arc(l.x * width, l.y * height, isActive ? 8 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = isActive ? currentFormColor : "rgba(255,255,255,0.3)";
    ctx.fill();
  }
}

// ─── Update UI Dashboard ────────────────────────────────────
// SVG Icons
const ICON_CHECK = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
const ICON_WARN = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
const ICON_INFO = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;

function updateDashboardUI() {
  // Video Overlays
  repsCount.textContent = repCount;
  badRepsCount.textContent = badRepCount;
  
  // Sidebar Metrics
  angleText.textContent = `${Math.round(elbowAngle)}°`;
  stageText.textContent = stage;

  // Master Feedback Box
  let masterStatusText = "READY";
  if (feedbackType === "good") masterStatusText = "GOOD REP";
  else if (feedbackType === "bad") masterStatusText = "BAD FORM";
  else if (stage === Stage.UP) masterStatusText = "SQUEEZE";
  
  masterText.textContent = masterStatusText;
  masterBox.className = `feedback-box status-${feedbackType}`;

  // Cues Card
  cueTitleEl.textContent = cueTitle;
  cueDescEl.textContent = feedback;
  
  if (feedbackType === "good") {
    cueIcon.innerHTML = ICON_CHECK;
    cueIcon.className = "cue-icon good";
  } else if (feedbackType === "bad") {
    cueIcon.innerHTML = ICON_WARN;
    cueIcon.className = "cue-icon error";
  } else {
    cueIcon.innerHTML = ICON_INFO;
    // Blue info 
    cueIcon.className = "cue-icon";
    cueIcon.style.background = "rgba(37, 99, 235, 0.1)";
    cueIcon.style.color = "var(--accent-blue)";
  }
}

// ─── Control Logic ──────────────────────────────────────────
let pose = null;
async function init() {
  const params = new URLSearchParams(window.location.search);
  const type = params.get("type");
  titleTagEl.textContent = (type === "shoulder") ? "SHOULDER PRESS ACTIVE" : "BICEP ANALYSIS ACTIVE";

  pose = new Pose({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}` });
  pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  pose.onResults(onPoseResults);

  await startCameraFlow();
  setupActivityListeners();
  resetTimer();
}

function onPoseResults(results) {
  if (!canvasEl || !ctx) return;
  const width = canvasEl.width; const height = canvasEl.height;
  ctx.clearRect(0, 0, width, height);

  if (results.poseLandmarks) {
    analyzeForm(results.poseLandmarks);
    drawSkeleton(results.poseLandmarks, width, height);
    updateDashboardUI();
    if (isActive) resetTimer();
  }
}

let captureLoopRunning = false;
let lastVideoTime = -1;

async function captureFrame() {
  if (!isActive) { captureLoopRunning = false; return; }
  if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
    if (videoEl.currentTime !== lastVideoTime) {
      lastVideoTime = videoEl.currentTime;
      if (videoEl.videoWidth && videoEl.videoHeight) {
        canvasEl.width = videoEl.videoWidth; canvasEl.height = videoEl.videoHeight;
      }
      if (pose) await pose.send({ image: videoEl });
      if (!poseReady) {
        poseReady = true;
        if (loadingEl) loadingEl.style.display = "none";
        startTimer(); // start recording clock
      }
    }
  }
  requestAnimationFrame(captureFrame);
}

async function startCameraFlow() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } });
    videoEl.srcObject = stream;
    isActive = true;
    if (cameraToggleBtn) {
        cameraToggleBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
    }

    videoEl.addEventListener("loadedmetadata", () => {
      videoEl.play();
      if (!captureLoopRunning) { captureLoopRunning = true; requestAnimationFrame(captureFrame); }
    });
  } catch (err) {
    if (loadingEl) loadingEl.style.display = "none";
    document.getElementById("error-msg").style.display = "flex";
  }
}

function toggleCamera() {
  if (isActive) {
    if (stream) { stream.getTracks().forEach((track) => track.stop()); videoEl.srcObject = null; stream = null; }
    isActive = false;
    // Play button SVG
    if (cameraToggleBtn) cameraToggleBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    cueTitle = "System Paused"; feedback = "Camera is offline."; feedbackType = "neutral";
    updateDashboardUI();
  } else {
    startCameraFlow();
  }
}

function setupActivityListeners() {
  ["mousemove", "keydown", "click", "touchstart", "scroll"].forEach((e) => document.addEventListener(e, resetTimer));
  resumeBtn.addEventListener("click", () => {
    overlayEl.style.display = "none";
    if (!isActive) startCameraFlow();
    resetTimer();
  });
}

function resetTimer() {
  clearTimeout(inactivityTimer);
  if (isActive) inactivityTimer = setTimeout(handleInactivity, INACTIVITY_LIMIT_MS);
}

function handleInactivity() {
  if (stream) { stream.getTracks().forEach((track) => track.stop()); videoEl.srcObject = null; stream = null; }
  isActive = false;
  if (cameraToggleBtn) cameraToggleBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  overlayEl.style.display = "flex";
}

document.addEventListener("DOMContentLoaded", init);
