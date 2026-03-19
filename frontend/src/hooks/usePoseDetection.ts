import { useRef, useCallback, useEffect } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface UsePoseDetectionOptions {
  onResults: (worldLandmarks: PoseLandmark[], normalizedLandmarks: PoseLandmark[]) => void;
}

export function usePoseDetection({ onResults }: UsePoseDetectionOptions) {
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(-1);
  const isActiveRef = useRef(false);
  const onResultsRef = useRef(onResults);
  // eslint-disable-next-line react-compiler/react-compiler
  onResultsRef.current = onResults;

  // Initialize PoseLandmarker
  const init = useCallback(async () => {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
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
  // eslint-disable-next-line react-compiler/react-compiler
  detectLoopRef.current = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !canvas || !landmarker || !isActiveRef.current) return;

    if (video.readyState >= 2 && video.videoWidth > 0) {
      const now = performance.now();
      if (video.currentTime !== lastTimeRef.current) {
        lastTimeRef.current = video.currentTime;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const result = landmarker.detectForVideo(video, now);

        if (result.landmarks && result.landmarks.length > 0 && result.worldLandmarks && result.worldLandmarks.length > 0) {
          const normalizedLandmarks = result.landmarks[0] as PoseLandmark[];
          const worldLandmarks = result.worldLandmarks[0] as PoseLandmark[];
          onResultsRef.current(worldLandmarks, normalizedLandmarks);

          // Draw skeleton on canvas
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const drawingUtils = new DrawingUtils(ctx);
            const landmarks = normalizedLandmarks.map(l => ({
              x: l.x, y: l.y, z: l.z, visibility: l.visibility ?? 0,
            }));
            drawingUtils.drawLandmarks(landmarks, {
              radius: (data) => {
                return (data.from?.visibility ?? 0) > 0.5 ? 4 : 2;
              },
              color: 'rgba(255,255,255,0.8)',
              fillColor: 'rgba(255,255,255,0.6)',
            });
            drawingUtils.drawConnectors(
              landmarks,
              PoseLandmarker.POSE_CONNECTIONS,
              { color: 'rgba(255,255,255,0.5)', lineWidth: 2 }
            );
          }
        }
      }
    }

    rafRef.current = requestAnimationFrame(() => detectLoopRef.current?.());
  };

  // Start camera
  const startCamera = useCallback(async (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
    videoRef.current = video;
    canvasRef.current = canvas;

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
    if (videoRef.current && canvasRef.current) {
      return startCamera(videoRef.current, canvasRef.current);
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
