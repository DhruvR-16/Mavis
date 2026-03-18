// Mavis Frontend Logic — Bicep Curl & Shoulder Press Analyzer
// Uses MediaPipe Pose for real-time form detection

// ─── Configuration ───────────────────────────────────────────
const INACTIVITY_LIMIT_MS = 60000;

// Bicep Thresholds
const BICEP_UP = 50;
const BICEP_DOWN = 155;
const BICEP_DRIFT = 0.08;

// Shoulder Press Thresholds
const SHOULDER_UP = 150; // Nearly straight elbow
const SHOULDER_DOWN = 80; // Elbow below 90 deg

const MIN_REP_TIME_MS = 800;

// ─── State ───────────────────────────────────────────────────
let inactivityTimer;
let stream = null;
let isActive = true;
let poseReady = false;
let currentWorkoutType = "bicep"; // Default

// Form State
const Stage = { DOWN: "DOWN", UP: "UP" };
let stage = Stage.DOWN;
let repCount = 0;
let badRepCount = 0;
let feedback = "Start Workout";
let cueTitle = "Positioning";
let feedbackType = "info"; // "good", "bad", "info", "neutral"
let repStartTime = 0;

// Workout Specific Vars
let elbowAngle = 0; // Displayed angle
let anchorElbowX = 0;
let currentFormColor = "#E2FF00"; // Volt Yellow by default
let activeSide = "left";

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

// ─── Math Helpers ───────────────────────────────────────────
function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

const LM = { LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_ELBOW: 13, RIGHT_ELBOW: 14, LEFT_WRIST: 15, RIGHT_WRIST: 16, LEFT_HIP: 23, RIGHT_HIP: 24 };
function lm(landmarks, idx) { return landmarks[idx]; }
function isLandmarkVisible(l, threshold = 0.5) { return l && l.visibility > threshold; }

// ─── BICEP CURL LOGIC ───────────────────────────────────────
function detectActiveArm(landmarks) {
  const lS = lm(landmarks, LM.LEFT_SHOULDER); const lE = lm(landmarks, LM.LEFT_ELBOW); const lW = lm(landmarks, LM.LEFT_WRIST);
  const rS = lm(landmarks, LM.RIGHT_SHOULDER); const rE = lm(landmarks, LM.RIGHT_ELBOW); const rW = lm(landmarks, LM.RIGHT_WRIST);
  const leftVis = isLandmarkVisible(lS) && isLandmarkVisible(lE) && isLandmarkVisible(lW);
  const rightVis = isLandmarkVisible(rS) && isLandmarkVisible(rE) && isLandmarkVisible(rW);

  if (leftVis && rightVis) {
    const lA = calculateAngle(lS, lE, lW);
    const rA = calculateAngle(rS, rE, rW);
    // Arm with tighter angle is curling
    activeSide = (lA < rA) ? "left" : "right";
  } else if (leftVis) activeSide = "left";
  else if (rightVis) activeSide = "right";
  return activeSide;
}

function analyzeBicepForm(landmarks) {
  const side = detectActiveArm(landmarks);
  const shoulder = side === "left" ? lm(landmarks, LM.LEFT_SHOULDER) : lm(landmarks, LM.RIGHT_SHOULDER);
  const elbow = side === "left" ? lm(landmarks, LM.LEFT_ELBOW) : lm(landmarks, LM.RIGHT_ELBOW);
  const wrist = side === "left" ? lm(landmarks, LM.LEFT_WRIST) : lm(landmarks, LM.RIGHT_WRIST);
  const hip = side === "left" ? lm(landmarks, LM.LEFT_HIP) : lm(landmarks, LM.RIGHT_HIP);

  if (!isLandmarkVisible(shoulder) || !isLandmarkVisible(elbow) || !isLandmarkVisible(wrist) || !isLandmarkVisible(hip)) {
    cueTitle = "Positioning"; feedback = "Step into frame"; feedbackType = "neutral"; currentFormColor = "#94A3B8"; return;
  }

  elbowAngle = calculateAngle(shoulder, elbow, wrist);
  const shoulderAngle = calculateAngle(elbow, shoulder, hip);
  const now = Date.now();

  // Bicep State Machine
  if (elbowAngle > BICEP_DOWN) {
    if (stage === Stage.UP) { // Completing Rep
      if (now - repStartTime > MIN_REP_TIME_MS) {
        repCount++; cueTitle = "Correct Form"; feedback = "Excellent Rep!"; feedbackType = "good"; currentFormColor = "#00FF66";
      } else {
        badRepCount++; cueTitle = "Tempo Control"; feedback = "Too Fast! Slow down"; feedbackType = "bad"; currentFormColor = "#FF3366";
      }
      stage = Stage.DOWN;
    } else {
      currentFormColor = "#E2FF00";
      if (!feedback.includes("Fast") && !feedback.includes("Excellent")) { cueTitle = "Ready"; feedback = "Curl upwards!"; feedbackType = "info"; }
      anchorElbowX = elbow.x;
    }
  } else if (elbowAngle < BICEP_UP) {
    if (stage === Stage.DOWN) { // Contracting
      stage = Stage.UP; repStartTime = now; anchorElbowX = elbow.x;
      cueTitle = "Contraction"; feedback = "Squeeze the bicep!"; feedbackType = "good"; currentFormColor = "#00FF66";
    }
    // Drift check
    if (Math.abs(elbow.x - anchorElbowX) > BICEP_DRIFT) {
      cueTitle = "Form Warning"; feedback = "Fix Elbow! Keep it pinned."; feedbackType = "bad"; currentFormColor = "#FF3366";
    }
  } else {
    // Middle Phase
    currentFormColor = "#E2FF00";
    if (stage === Stage.UP) {
      if (shoulderAngle > 45) { cueTitle = "Torso Swing"; feedback = "Don't swing back!"; feedbackType = "bad"; currentFormColor = "#FF3366"; }
      else { cueTitle = "Eccentric"; feedback = "Lower slowly..."; feedbackType = "info"; }
    } else if (stage === Stage.DOWN) {
      cueTitle = "Concentric"; feedback = "Keep going up!"; feedbackType = "info";
    }
  }

  if (stage === Stage.DOWN && elbowAngle < BICEP_DOWN && elbowAngle > BICEP_UP && elbowAngle < 120) {
    cueTitle = "ROM Check"; feedback = "Extend arm fully!"; feedbackType = "info"; currentFormColor = "#FFB800";
  }
}

// ─── SHOULDER PRESS LOGIC ───────────────────────────────────
function analyzeShoulderForm(landmarks) {
  const lS = lm(landmarks, LM.LEFT_SHOULDER); const lE = lm(landmarks, LM.LEFT_ELBOW); const lW = lm(landmarks, LM.LEFT_WRIST);
  const rS = lm(landmarks, LM.RIGHT_SHOULDER); const rE = lm(landmarks, LM.RIGHT_ELBOW); const rW = lm(landmarks, LM.RIGHT_WRIST);

  const leftVis = isLandmarkVisible(lS) && isLandmarkVisible(lE) && isLandmarkVisible(lW);
  const rightVis = isLandmarkVisible(rS) && isLandmarkVisible(rE) && isLandmarkVisible(rW);

  if (!leftVis || !rightVis) {
    cueTitle = "Positioning"; feedback = "Both arms must be visible"; feedbackType = "neutral"; currentFormColor = "#94A3B8"; return;
  }

  const leftAngle = calculateAngle(lS, lE, lW);
  const rightAngle = calculateAngle(rS, rE, rW);
  elbowAngle = (leftAngle + rightAngle) / 2; // Average for UI display

  const now = Date.now();

  // Uneven pressing check
  if (Math.abs(leftAngle - rightAngle) > 25) {
    cueTitle = "Imbalance"; feedback = "Press evenly!"; feedbackType = "bad"; currentFormColor = "#FF3366";
  } else {
    // Shoulder Press State Machine
    if (elbowAngle < SHOULDER_DOWN) {
      // Bottom of the press
      if (stage === Stage.UP) {
        if (now - repStartTime > MIN_REP_TIME_MS) {
          repCount++; cueTitle = "Correct Form"; feedback = "Great Press!"; feedbackType = "good"; currentFormColor = "#00FF66";
        } else {
          badRepCount++; cueTitle = "Control Drop"; feedback = "Don't drop the weight abruptly"; feedbackType = "bad"; currentFormColor = "#FF3366";
        }
        stage = Stage.DOWN;
      } else {
        currentFormColor = "#E2FF00";
        if (!feedback.includes("Great") && !feedback.includes("drop")) {
          cueTitle = "Ready"; feedback = "Press overhead!"; feedbackType = "info";
        }
      }
    } else if (elbowAngle > SHOULDER_UP) {
      // Top of press
      if (stage === Stage.DOWN) {
        stage = Stage.UP; repStartTime = now;
        cueTitle = "Lockout"; feedback = "Hold at the top!"; feedbackType = "good"; currentFormColor = "#00FF66";
      }
    } else {
      // Mid Rep
      currentFormColor = "#E2FF00";
      if (stage === Stage.UP) { cueTitle = "Lowering"; feedback = "Control descent..."; feedbackType = "info"; }
      else if (stage === Stage.DOWN) { cueTitle = "Pressing"; feedback = "Push up!"; feedbackType = "info"; }
    }
  }
}

// ─── Drawing ────────────────────────────────────────────────
const POSE_CONNECTIONS_BODY = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]
];

function drawSkeleton(landmarks, width, height) {
  // Bicep highlights one arm, Shoulder highlights both
  const activeIndices = [];
  if (currentWorkoutType === "bicep") {
    if (activeSide === "left") activeIndices.push(LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST);
    else activeIndices.push(LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST);
  } else {
    // Both arms for shoulder press
    activeIndices.push(LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST);
  }

  for (const [i, j] of POSE_CONNECTIONS_BODY) {
    const a = landmarks[i]; const b = landmarks[j];
    if (!a || !b || a.visibility < 0.3 || b.visibility < 0.3) continue;
    
    const isActiveArm = activeIndices.includes(i) || activeIndices.includes(j);
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height); ctx.lineTo(b.x * width, b.y * height);
    // Huge visibility fix requested by user: full body drawn cleanly
    ctx.strokeStyle = isActiveArm ? currentFormColor : "rgba(255,255,255,0.7)";
    ctx.lineWidth = isActiveArm ? 6 : 3;
    ctx.stroke();
  }
  
  for (let i = 0; i <= 28; i++) {
    const l = landmarks[i];
    if (!l || l.visibility < 0.3) continue;
    const isActive = activeIndices.includes(i);
    ctx.beginPath();
    ctx.arc(l.x * width, l.y * height, isActive ? 8 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = isActive ? currentFormColor : "rgba(255,255,255,0.9)";
    ctx.fill();
  }
}

// ─── Update UI Dashboard ────────────────────────────────────
const ICON_CHECK = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
const ICON_WARN = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
const ICON_INFO = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;

function updateDashboardUI() {
  repsCount.textContent = repCount;
  badRepsCount.textContent = badRepCount;
  angleText.textContent = `${Math.round(elbowAngle)}°`;
  stageText.textContent = stage;

  // Formatting strings slightly to look cleaner
  let masterStatusText = "READY";
  if (feedbackType === "good") masterStatusText = "GOOD FORM";
  else if (feedbackType === "bad") masterStatusText = "BAD FORM";
  else if (stage === Stage.UP) masterStatusText = "HOLD";
  
  masterText.textContent = masterStatusText;
  masterBox.className = `feedback-box status-${feedbackType}`;

  // Styling text for high contrast based on theme
  cueTitleEl.textContent = cueTitle;
  cueDescEl.textContent = feedback;
  cueDescEl.style.color = "#E2E8F0"; // Slate-200 for extremely readable contrast
  
  if (feedbackType === "good") {
    cueIcon.innerHTML = ICON_CHECK;
    cueIcon.className = "cue-icon good";
  } else if (feedbackType === "bad") {
    cueIcon.innerHTML = ICON_WARN;
    cueIcon.className = "cue-icon error";
  } else {
    cueIcon.innerHTML = ICON_INFO;
    cueIcon.className = "cue-icon";
    cueIcon.style.background = "rgba(226, 255, 0, 0.15)";
    cueIcon.style.color = "var(--accent-main)";
  }
}

// ─── Control Logic ──────────────────────────────────────────
let pose = null;
async function init() {
  const params = new URLSearchParams(window.location.search);
  const type = params.get("type");
  currentWorkoutType = (type === "shoulder") ? "shoulder" : "bicep";
  titleTagEl.textContent = (currentWorkoutType === "shoulder") ? "SHOULDER PRESS ACTIVE" : "BICEP ANALYSIS ACTIVE";

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
    if (currentWorkoutType === "shoulder") {
      analyzeShoulderForm(results.poseLandmarks);
    } else {
      analyzeBicepForm(results.poseLandmarks);
    }
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
