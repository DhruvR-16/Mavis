// Mavis Frontend Logic — Bicep Curl Analyzer
// Uses MediaPipe Pose for real-time form detection

// ─── Configuration ───────────────────────────────────────────
const INACTIVITY_LIMIT_MS = 60000; // 60 seconds (longer for workouts)

// Bicep Curl Thresholds
const UP_THRESHOLD = 50;       // Elbow angle for peak contraction
const DOWN_THRESHOLD = 155;    // Elbow angle for full extension
const MIN_REP_TIME_MS = 800;   // Minimum rep duration (ms)
const ELBOW_DRIFT_TOLERANCE = 0.08; // Max lateral drift (fraction of frame width)
const WRIST_ABOVE_SHOULDER_TOLERANCE = 0.03; // Shoulder swing detection

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
let feedbackType = "neutral"; // "good", "bad", "neutral", "info"
let elbowAngle = 0;
let repStartTime = 0;
let anchorElbowX = 0;
let currentFormColor = "#22c55e"; // green

// Active side detection
let activeSide = "left";    // Will be auto-detected
let lastLeftAngle = 180;
let lastRightAngle = 180;

// ─── DOM Elements ────────────────────────────────────────────
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("pose-canvas");
const ctx = canvasEl.getContext("2d");
const overlayEl = document.getElementById("timeout-overlay");
const resumeBtn = document.getElementById("resume-btn");
const cameraToggleBtn = document.getElementById("camera-toggle-btn");
if (cameraToggleBtn) {
  cameraToggleBtn.addEventListener("click", toggleCamera);
}

const titleEl = document.getElementById("exercise-title");
const loadingEl = document.getElementById("loading-overlay");

// Stats elements
const stageText = document.getElementById("stage-text");
const repsCount = document.getElementById("reps-count");
const badRepsCount = document.getElementById("bad-reps-count");
const angleText = document.getElementById("angle-text");
const feedbackText = document.getElementById("feedback-text");

// ─── Angle Calculation ──────────────────────────────────────
function calculateAngle(a, b, c) {
  // a, b, c are {x, y} objects. b is the vertex.
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

// ─── Landmark Helpers ───────────────────────────────────────
// MediaPipe Pose landmark indices
const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
};

function lm(landmarks, idx) {
  return landmarks[idx];
}

function isLandmarkVisible(landmark, threshold = 0.5) {
  return landmark && landmark.visibility > threshold;
}

// ─── Auto-detect Active Arm ─────────────────────────────────
function detectActiveSide(landmarks) {
  const lShoulder = lm(landmarks, LM.LEFT_SHOULDER);
  const lElbow = lm(landmarks, LM.LEFT_ELBOW);
  const lWrist = lm(landmarks, LM.LEFT_WRIST);
  const rShoulder = lm(landmarks, LM.RIGHT_SHOULDER);
  const rElbow = lm(landmarks, LM.RIGHT_ELBOW);
  const rWrist = lm(landmarks, LM.RIGHT_WRIST);

  const leftVisible =
    isLandmarkVisible(lShoulder) &&
    isLandmarkVisible(lElbow) &&
    isLandmarkVisible(lWrist);
  const rightVisible =
    isLandmarkVisible(rShoulder) &&
    isLandmarkVisible(rElbow) &&
    isLandmarkVisible(rWrist);

  if (leftVisible && rightVisible) {
    // Both visible — use the arm with more elbow flexion (smaller angle)
    const leftAngle = calculateAngle(lShoulder, lElbow, lWrist);
    const rightAngle = calculateAngle(rShoulder, rElbow, rWrist);
    lastLeftAngle = leftAngle;
    lastRightAngle = rightAngle;
    activeSide = leftAngle < rightAngle ? "left" : "right";
  } else if (leftVisible) {
    activeSide = "left";
  } else if (rightVisible) {
    activeSide = "right";
  }
  return activeSide;
}

function getActiveKeypoints(landmarks) {
  const side = detectActiveSide(landmarks);
  if (side === "left") {
    return {
      shoulder: lm(landmarks, LM.LEFT_SHOULDER),
      elbow: lm(landmarks, LM.LEFT_ELBOW),
      wrist: lm(landmarks, LM.LEFT_WRIST),
      hip: lm(landmarks, LM.LEFT_HIP),
      side: "L",
    };
  } else {
    return {
      shoulder: lm(landmarks, LM.RIGHT_SHOULDER),
      elbow: lm(landmarks, LM.RIGHT_ELBOW),
      wrist: lm(landmarks, LM.RIGHT_WRIST),
      hip: lm(landmarks, LM.RIGHT_HIP),
      side: "R",
    };
  }
}

// ─── Form Analysis ──────────────────────────────────────────
function analyzeForm(landmarks) {
  const kp = getActiveKeypoints(landmarks);
  const { shoulder, elbow, wrist, hip } = kp;

  // Verify all keypoints visible
  if (
    !isLandmarkVisible(shoulder) ||
    !isLandmarkVisible(elbow) ||
    !isLandmarkVisible(wrist) ||
    !isLandmarkVisible(hip)
  ) {
    feedback = "Position yourself in frame";
    feedbackType = "neutral";
    return;
  }

  // Calculate elbow angle (shoulder → elbow → wrist)
  elbowAngle = calculateAngle(shoulder, elbow, wrist);

  // Calculate shoulder angle (elbow → shoulder → hip) to detect shoulder swing
  const shoulderAngle = calculateAngle(elbow, shoulder, hip);

  const now = Date.now();

  // ── State Machine ──
  if (elbowAngle > DOWN_THRESHOLD) {
    // ARM IS EXTENDED (DOWN position)
    if (stage === Stage.UP) {
      // Completing a rep (UP → DOWN transition)
      const duration = now - repStartTime;
      if (duration > MIN_REP_TIME_MS) {
        repCount++;
        feedback = "Good Rep! 💪";
        feedbackType = "good";
        currentFormColor = "#22c55e";
      } else {
        badRepCount++;
        feedback = "Too Fast! Slow down";
        feedbackType = "bad";
        currentFormColor = "#ef4444";
      }
      stage = Stage.DOWN;
    } else {
      // Already in DOWN — waiting for curl
      if (feedback === "Start Curls" || feedback === "Good Rep! 💪" || feedback === "Too Fast! Slow down") {
        feedback = "Curl up!";
        feedbackType = "info";
      }
      currentFormColor = "#22c55e";
      anchorElbowX = elbow.x; // Reset anchor point
    }
  } else if (elbowAngle < UP_THRESHOLD) {
    // ARM IS CURLED (UP position)
    if (stage === Stage.DOWN) {
      // Starting the up phase
      stage = Stage.UP;
      repStartTime = now;
      anchorElbowX = elbow.x;
      feedback = "Squeeze! 🔥";
      feedbackType = "good";
      currentFormColor = "#22c55e";
    }

    // Check elbow drift while curled
    const drift = Math.abs(elbow.x - anchorElbowX);
    if (drift > ELBOW_DRIFT_TOLERANCE) {
      feedback = "Fix Elbow! Keep it steady";
      feedbackType = "bad";
      currentFormColor = "#ef4444";
    }
  } else {
    // Mid-range movement
    if (stage === Stage.UP) {
      // Check for shoulder swing (bad form)
      if (shoulderAngle > 45) {
        feedback = "Don't swing! Use your bicep";
        feedbackType = "bad";
        currentFormColor = "#ef4444";
      } else {
        feedback = "Lower slowly...";
        feedbackType = "info";
        currentFormColor = "#22c55e";
      }
    } else if (stage === Stage.DOWN) {
      // Moving up from down
      if (elbowAngle < DOWN_THRESHOLD - 10) {
        feedback = "Keep going up!";
        feedbackType = "info";
        currentFormColor = "#22c55e";
      }
    }
  }

  // Check: incomplete extension when in DOWN stage
  if (stage === Stage.DOWN && elbowAngle < DOWN_THRESHOLD && elbowAngle > UP_THRESHOLD) {
    if (elbowAngle < 120) {
      feedback = "Extend fully!";
      feedbackType = "info";
    }
  }
}

// ─── Drawing ────────────────────────────────────────────────
const POSE_CONNECTIONS_BODY = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
];

function drawSkeleton(landmarks, width, height) {
  // Determine which landmarks belong to the active arm
  const activeIndices = activeSide === "left"
    ? [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST]
    : [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST];

  // Draw connections
  for (const [i, j] of POSE_CONNECTIONS_BODY) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (!a || !b || a.visibility < 0.4 || b.visibility < 0.4) continue;

    const isActiveArm =
      (activeIndices.includes(i) || activeIndices.includes(j));

    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.strokeStyle = isActiveArm ? currentFormColor : "rgba(255,255,255,0.35)";
    ctx.lineWidth = isActiveArm ? 4 : 2;
    ctx.stroke();
  }

  // Draw landmark dots
  for (let i = 0; i < landmarks.length; i++) {
    const l = landmarks[i];
    if (!l || l.visibility < 0.4) continue;
    // Only draw upper body (indices 0–28)
    if (i > 28) continue;

    const isActive = activeIndices.includes(i);
    const x = l.x * width;
    const y = l.y * height;

    ctx.beginPath();
    ctx.arc(x, y, isActive ? 6 : 3, 0, 2 * Math.PI);
    ctx.fillStyle = isActive ? currentFormColor : "rgba(255,255,255,0.5)";
    ctx.fill();

    if (isActive) {
      ctx.strokeStyle = "rgba(0,0,0,0.4)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

function drawAngleArc(landmarks, width, height) {
  const kp = getActiveKeypoints(landmarks);
  const { shoulder, elbow, wrist } = kp;
  if (
    !isLandmarkVisible(shoulder) ||
    !isLandmarkVisible(elbow) ||
    !isLandmarkVisible(wrist)
  ) return;

  const ex = elbow.x * width;
  const ey = elbow.y * height;
  const sx = shoulder.x * width;
  const sy = shoulder.y * height;
  const wx = wrist.x * width;
  const wy = wrist.y * height;

  // Draw angle arc
  const startAngle = Math.atan2(sy - ey, sx - ex);
  const endAngle = Math.atan2(wy - ey, wx - ex);
  const radius = 30;

  ctx.beginPath();
  ctx.arc(ex, ey, radius, startAngle, endAngle, elbowAngle > 180);
  ctx.strokeStyle = currentFormColor;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Draw angle text
  const textX = ex + Math.cos((startAngle + endAngle) / 2) * (radius + 20);
  const textY = ey + Math.sin((startAngle + endAngle) / 2) * (radius + 20);

  ctx.font = "bold 16px -apple-system, sans-serif";
  ctx.fillStyle = currentFormColor;
  ctx.textAlign = "center";
  ctx.fillText(`${Math.round(elbowAngle)}°`, textX, textY);
}

function drawFeedbackBanner(width) {
  if (!feedback) return;

  const bannerHeight = 44;
  const y = 12;

  // Background pill
  let bgColor;
  switch (feedbackType) {
    case "good": bgColor = "rgba(34, 197, 94, 0.85)"; break;
    case "bad": bgColor = "rgba(239, 68, 68, 0.85)"; break;
    case "info": bgColor = "rgba(59, 130, 246, 0.85)"; break;
    default: bgColor = "rgba(0, 0, 0, 0.6)";
  }

  ctx.font = "bold 18px -apple-system, sans-serif";
  const textWidth = ctx.measureText(feedback).width;
  const pillWidth = textWidth + 36;
  const pillX = (width - pillWidth) / 2;

  ctx.beginPath();
  ctx.roundRect(pillX, y, pillWidth, bannerHeight, 22);
  ctx.fillStyle = bgColor;
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(feedback, width / 2, y + bannerHeight / 2);
}

// ─── Update UI Stats ────────────────────────────────────────
function updateStats() {
  stageText.textContent = stage;
  stageText.style.color = stage === Stage.UP ? "var(--accent)" : "var(--text-main)";

  repsCount.textContent = repCount;
  badRepsCount.textContent = badRepCount;
  angleText.textContent = `${Math.round(elbowAngle)}°`;

  feedbackText.textContent = feedback.replace(/[💪🔥]/g, "").trim();
  switch (feedbackType) {
    case "good":
      feedbackText.style.color = "var(--success)";
      break;
    case "bad":
      feedbackText.style.color = "var(--error)";
      break;
    case "info":
      feedbackText.style.color = "var(--accent)";
      break;
    default:
      feedbackText.style.color = "var(--text-muted)";
  }
}

// ─── MediaPipe Pose Setup ───────────────────────────────────
function initPose() {
  const pose = new Pose({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
  });

  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  pose.onResults(onPoseResults);
  return pose;
}

function onPoseResults(results) {
  if (!canvasEl || !ctx) return;

  const width = canvasEl.width;
  const height = canvasEl.height;

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  if (results.poseLandmarks) {
    const landmarks = results.poseLandmarks;

    // Run analysis
    analyzeForm(landmarks);

    // Draw skeleton
    drawSkeleton(landmarks, width, height);

    // Draw angle arc near elbow
    drawAngleArc(landmarks, width, height);

    // Draw feedback banner on canvas
    drawFeedbackBanner(width);

    // Update DOM stats
    updateStats();

    // Reset inactivity on valid pose detection
    if (isActive) resetTimer();
  }
}

// ─── Camera & Canvas Sizing ─────────────────────────────────
function syncCanvasSize() {
  if (videoEl.videoWidth && videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }
}

let pose = null;

// ─── Initialize ─────────────────────────────────────────────
async function init() {
  // Set Title based on URL param
  const params = new URLSearchParams(window.location.search);
  const type = params.get("type");

  if (type === "bicep") {
    titleEl.textContent = "Bicep Curls Analysis";
  } else if (type === "shoulder") {
    titleEl.textContent = "Shoulder Press Analysis";
  } else {
    titleEl.textContent = "Workout Analysis";
  }

  // Initialize MediaPipe Pose
  pose = initPose();

  // Start Camera
  await startCameraFlow();

  // Setup Inactivity Listeners
  setupActivityListeners();
  resetTimer();
}

let captureLoopRunning = false;
let lastVideoTime = -1;

async function captureFrame() {
  if (!isActive) {
    captureLoopRunning = false;
    return;
  }
  
  if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
    if (videoEl.currentTime !== lastVideoTime) {
      lastVideoTime = videoEl.currentTime;
      syncCanvasSize();
      if (pose) await pose.send({ image: videoEl });
      
      // Hide loading overlay once the first frame is processed
      if (!poseReady) {
        poseReady = true;
        if (loadingEl) loadingEl.style.display = "none";
        console.log("MediaPipe Pose initialized.");
      }
    }
  }
  requestAnimationFrame(captureFrame);
}

async function startCameraFlow() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
    });
    videoEl.srcObject = stream;
    isActive = true;
    if (cameraToggleBtn) cameraToggleBtn.textContent = "Turn Off Camera";

    videoEl.addEventListener("loadedmetadata", () => {
      videoEl.play();
      syncCanvasSize();

      if (!captureLoopRunning) {
        captureLoopRunning = true;
        requestAnimationFrame(captureFrame);
      }
    });
  } catch (err) {
    console.error("Camera access denied:", err);
    if (loadingEl) loadingEl.style.display = "none";
    document.getElementById("error-msg").style.display = "flex";
  }
}

function toggleCamera() {
  if (isActive) {
    // Turn off
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      videoEl.srcObject = null;
      stream = null;
    }
    isActive = false;
    if (cameraToggleBtn) cameraToggleBtn.textContent = "Turn On Camera";
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    feedback = "Camera Off";
    feedbackType = "neutral";
    updateStats();
  } else {
    // Turn on
    if (cameraToggleBtn) cameraToggleBtn.textContent = "Starting...";
    startCameraFlow();
  }
}

// ─── Inactivity Logic ───────────────────────────────────────
function setupActivityListeners() {
  const events = ["mousemove", "keydown", "click", "touchstart", "scroll"];
  events.forEach((event) => {
    document.addEventListener(event, resetTimer);
  });

  resumeBtn.addEventListener("click", () => {
    overlayEl.style.display = "none";
    if (!isActive) {
      startCameraFlow();
    }
    resetTimer();
  });
}

function resetTimer() {
  clearTimeout(inactivityTimer);
  if (isActive) {
    inactivityTimer = setTimeout(handleInactivity, INACTIVITY_LIMIT_MS);
  }
}

function handleInactivity() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    videoEl.srcObject = null;
    stream = null;
    isActive = false;
    console.log("Camera stopped (Inactivity)");
  }
  overlayEl.style.display = "flex";
}

// Run
document.addEventListener("DOMContentLoaded", init);
