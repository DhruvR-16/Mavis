import { useRef, useCallback, useEffect } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface UsePoseDetectionOptions {
  onResults: (worldLandmarks: PoseLandmark[], normalizedLandmarks: PoseLandmark[]) => void;
}

// Pinned to the npm-installed @mediapipe/tasks-vision version (see package.json)
// so a jsDelivr publish can't change production behavior under us.
const TASKS_VISION_VERSION = '0.10.32';
const WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const MODEL_ASSET_PATH = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// Rendering is the caller's responsibility — this hook only detects.
export function usePoseDetection({ onResults }: UsePoseDetectionOptions) {
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(-1);
  const isActiveRef = useRef(false);
  const onResultsRef = useRef(onResults);
  useEffect(() => {
    onResultsRef.current = onResults;
  });

  // Initialize PoseLandmarker
  const init = useCallback(async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
    landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET_PATH,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.7,
      minPosePresenceConfidence: 0.7,
      minTrackingConfidence: 0.7,
      outputSegmentationMasks: false,
    });
  }, []);

  // Detection loop — defined as a ref to avoid dependency issues
  const detectLoopRef = useRef<(() => void) | undefined>(undefined);
  useEffect(() => {
    detectLoopRef.current = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;

      if (!video || !landmarker || !isActiveRef.current) return;

      if (video.readyState >= 2 && video.videoWidth > 0) {
        const now = performance.now();
        if (video.currentTime !== lastTimeRef.current) {
          lastTimeRef.current = video.currentTime;

          const result = landmarker.detectForVideo(video, now);

          if (result.landmarks && result.landmarks.length > 0 && result.worldLandmarks && result.worldLandmarks.length > 0) {
            const normalizedLandmarks = result.landmarks[0] as PoseLandmark[];
            const worldLandmarks = result.worldLandmarks[0] as PoseLandmark[];
            onResultsRef.current(worldLandmarks, normalizedLandmarks);
          }
        }
      }

      rafRef.current = requestAnimationFrame(() => detectLoopRef.current?.());
    };
  });

  // Start camera. Returns false (without throwing) on permission denial or
  // any other getUserMedia failure, so the caller can keep its own
  // "enable camera" UI visible instead of assuming success.
  const startCamera = useCallback(async (video: HTMLVideoElement) => {
    videoRef.current = video;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
      });
      streamRef.current = stream;
      video.srcObject = stream;
      isActiveRef.current = true;

      video.addEventListener('loadedmetadata', () => {
        video.play();
        detectLoopRef.current?.();
      });
      return true;
    } catch {
      return false;
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    isActiveRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  // Resume camera
  const resumeCamera = useCallback(async () => {
    if (videoRef.current) {
      return startCamera(videoRef.current);
    }
    return false;
  }, [startCamera]);

  // Toggle
  const toggleCamera = useCallback(async () => {
    if (isActiveRef.current) {
      stopCamera();
      return false;
    } else {
      return resumeCamera();
    }
  }, [stopCamera, resumeCamera]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
    };
  }, [stopCamera]);

  return {
    init,
    startCamera,
    stopCamera,
    resumeCamera,
    toggleCamera,
    isActive: () => isActiveRef.current,
    landmarkerRef,
  };
}
