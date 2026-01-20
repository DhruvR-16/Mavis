// Mavis Frontend Logic

// Configuration
const INACTIVITY_LIMIT_MS = 20000; // 20 Seconds

// State
let inactivityTimer;
let stream = null;
let isActive = true;

// DOM Elements
const videoEl = document.getElementById("webcam");
const overlayEl = document.getElementById("timeout-overlay");
const resumeBtn = document.getElementById("resume-btn");
const titleEl = document.getElementById("exercise-title");

// 1. Initialize Page
function init() {
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

  // Start Camera
  startCamera();

  // Setup Timer Listeners
  setupActivityListeners();
  resetTimer(); // Start timer
}

// 2. Camera Logic
async function startCamera() {
  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoEl.srcObject = stream;
      isActive = true;
      console.log("Camera started");
    }
  } catch (err) {
    console.error("Camera access denied:", err);
    document.getElementById("error-msg").style.display = "flex";
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    videoEl.srcObject = null;
    stream = null;
    isActive = false;
    console.log("Camera stopped (Inactivity)");
  }
}

// 3. Inactivity Logic
function setupActivityListeners() {
  const events = ["mousemove", "keydown", "click", "touchstart", "scroll"];
  events.forEach((event) => {
    document.addEventListener(event, resetTimer);
  });

  resumeBtn.addEventListener("click", () => {
    overlayEl.style.display = "none";
    startCamera();
    resetTimer();
  });
}

function resetTimer() {
  // Clear existing timer
  clearTimeout(inactivityTimer);

  // If we were inactive/stopped, don't auto-restart just by moving mouse.
  // User must click 'Resume' button (handled by resumeBtn listener).
  // We only reset the countdown if we are currently ACTIVE.
  if (isActive) {
    inactivityTimer = setTimeout(handleInactivity, INACTIVITY_LIMIT_MS);
  }
}

function handleInactivity() {
  stopCamera();
  overlayEl.style.display = "flex"; // Show Paused Overlay
}

// Run
document.addEventListener("DOMContentLoaded", init);
