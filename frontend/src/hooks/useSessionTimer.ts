import { useState, useRef, useCallback } from 'react';

export interface SessionTimerState {
  elapsed: number;
  formatted: string;
  isRunning: boolean;
}

export function useSessionTimer() {
  const [elapsed, setElapsed] = useState(0);
  const intervalRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);

  const formatTime = (ms: number): string => {
    const totalSecs = Math.floor(ms / 1000);
    const h = Math.floor(totalSecs / 3600);
    const m = Math.floor((totalSecs % 3600) / 60);
    const s = totalSecs % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const start = useCallback(() => {
    if (startTimeRef.current === null) startTimeRef.current = Date.now();
    if (intervalRef.current !== null) clearInterval(intervalRef.current);
    intervalRef.current = window.setInterval(() => {
      const now = Date.now();
      setElapsed(now - (startTimeRef.current ?? now));
    }, 500);
  }, []);

  const stop = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  return {
    elapsed,
    formatted: formatTime(elapsed),
    isRunning: intervalRef.current !== null,
    start,
    stop,
  };
}
