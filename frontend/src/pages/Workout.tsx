import { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import Navbar from '../components/Navbar';
import VideoPanel from '../components/VideoPanel';
import Sidebar from '../components/Sidebar';
import { usePoseDetection } from '../hooks/usePoseDetection';
import { useWorkoutAnalyzer, type WorkoutState, type WorkoutType } from '../hooks/useWorkoutAnalyzer';
import { useSessionTimer } from '../hooks/useSessionTimer';
import type { PoseLandmark } from '../hooks/usePoseDetection';

const INACTIVITY_LIMIT_MS = 60000;

export default function Workout() {
  const [searchParams] = useSearchParams();
  const type = (searchParams.get('type') === 'shoulder' ? 'shoulder' : 'bicep') as WorkoutType;

  const [workoutState, setWorkoutState] = useState<WorkoutState>({
    repCount: 0, badRepCount: 0, elbowAngle: 0, stage: 'DOWN',
    feedback: 'Start Workout', cueTitle: 'Positioning', feedbackType: 'info',
    formColor: '#2563EB', activeSide: 'left',
  });

  const [showLoading, setShowLoading] = useState(true);
  const [showError, setShowError] = useState(false);
  const [showTimeout, setShowTimeout] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);

  const timer = useSessionTimer();
  const { analyze } = useWorkoutAnalyzer(type);
  const initializedRef = useRef(false);
  const inactivityRef = useRef<number | null>(null);

  // Use a ref for the poseDetection object so resetInactivity can access stopCamera
  const poseDetectionRef = useRef<ReturnType<typeof usePoseDetection> | null>(null);

  const resetInactivity = useCallback(() => {
    if (inactivityRef.current) clearTimeout(inactivityRef.current);
    inactivityRef.current = window.setTimeout(() => {
      poseDetectionRef.current?.stopCamera();
      setCameraActive(false);
      setShowTimeout(true);
    }, INACTIVITY_LIMIT_MS);
  }, []);

  const handlePoseResults = useCallback(
    (worldLandmarks: PoseLandmark[], normalizedLandmarks: PoseLandmark[]) => {
      const state = analyze(worldLandmarks, normalizedLandmarks);
      setWorkoutState({ ...state });

      if (!initializedRef.current) {
        initializedRef.current = true;
        setShowLoading(false);
        timer.start();
      }

      resetInactivity();
    },
    [analyze, timer, resetInactivity],
  );

  const poseDetection = usePoseDetection({ onResults: handlePoseResults });
  poseDetectionRef.current = poseDetection;

  // Initialize pose landmarker and start camera
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await poseDetection.init();
      if (cancelled) return;

      const video = document.getElementById('webcam') as HTMLVideoElement;
      const canvas = document.getElementById('pose-canvas') as HTMLCanvasElement;
      if (video && canvas) {
        const success = await poseDetection.startCamera(video, canvas);
        if (success) {
          setCameraActive(true);
        } else {
          setShowLoading(false);
          setShowError(true);
        }
      }
    })();

    return () => {
      cancelled = true;
      poseDetection.stopCamera();
      timer.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Activity listeners for inactivity reset
  useEffect(() => {
    const events = ['mousemove', 'keydown', 'click', 'touchstart', 'scroll'];
    events.forEach((e) => document.addEventListener(e, resetInactivity));
    return () => events.forEach((e) => document.removeEventListener(e, resetInactivity));
  }, [resetInactivity]);

  const handleToggleCamera = async () => {
    const result = await poseDetection.toggleCamera();
    setCameraActive(!!result);
  };

  const handleResume = async () => {
    setShowTimeout(false);
    const success = await poseDetection.resumeCamera();
    setCameraActive(!!success);
    resetInactivity();
  };

  const exerciseTitle = type === 'shoulder' ? 'SHOULDER PRESS ACTIVE' : 'BICEP ANALYSIS ACTIVE';
  const durationStr = `${Math.floor(timer.elapsed / 1000)}s`;

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <Navbar />
      <main className="flex flex-1 p-5 gap-5 overflow-hidden">
        <VideoPanel
          timerFormatted={timer.formatted}
          elapsed={timer.elapsed}
          exerciseTitle={exerciseTitle}
          repCount={workoutState.repCount}
          badRepCount={workoutState.badRepCount}
          duration={durationStr}
          isActive={cameraActive}
          onToggleCamera={handleToggleCamera}
          showLoading={showLoading}
          showError={showError}
          showTimeout={showTimeout}
          onResume={handleResume}
        />
        <Sidebar state={workoutState} />
      </main>
    </div>
  );
}
