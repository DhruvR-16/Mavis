import { forwardRef } from 'react';

interface VideoPanelProps {
  timerFormatted: string;
  elapsed: number;
  exerciseTitle: string;
  repCount: number;
  badRepCount: number;
  duration: string;
  isActive: boolean;
  onToggleCamera: () => void;
  showLoading: boolean;
  showError: boolean;
  showTimeout: boolean;
  onResume: () => void;
}

const VideoPanel = forwardRef<HTMLDivElement, VideoPanelProps>(function VideoPanel(props, ref) {
  return (
    <div ref={ref} className="flex-1 relative bg-black rounded-xl overflow-hidden border border-border flex">
      <video id="webcam" autoPlay playsInline muted className="w-full h-full object-cover" />
      <canvas id="pose-canvas" className="absolute inset-0 w-full h-full z-5 pointer-events-none" />

      {/* Top Left Overlays */}
      <div className="absolute top-6 left-6 z-10 flex flex-col gap-2">
        <div className="flex items-center gap-2 bg-bg-main/80 backdrop-blur-sm px-3 py-1.5 rounded-md text-xs font-medium border border-white/8">
          <div
            className="w-1.5 h-1.5 bg-error rounded-full"
            style={{ opacity: Math.floor(props.elapsed / 500) % 2 === 0 ? 1 : 0.3 }}
          />
          REC: <span>{props.timerFormatted}</span>
        </div>
        <div className="flex items-center gap-2 bg-bg-main/80 backdrop-blur-sm px-3 py-1.5 rounded-md text-xs font-medium border border-white/8">
          <div className="w-1.5 h-1.5 bg-accent rounded-full" />
          <span>{props.exerciseTitle}</span>
        </div>
      </div>

      {/* Bottom Left Stats */}
      <div className="absolute bottom-6 left-6 z-10 flex gap-8 bg-bg-panel/85 backdrop-blur-lg border border-white/8 rounded-lg px-6 py-4">
        <div className="flex flex-col">
          <span className="text-[0.7rem] text-text-muted font-medium mb-0.5">Reps</span>
          <span className="text-2xl font-semibold leading-none">{props.repCount}</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[0.7rem] text-text-muted font-medium mb-0.5">Bad Reps</span>
          <span className="text-2xl font-semibold leading-none text-error">{props.badRepCount}</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[0.7rem] text-text-muted font-medium mb-0.5">Duration</span>
          <span className="text-2xl font-semibold leading-none">{props.duration}</span>
        </div>
      </div>

      {/* Bottom Right Controls */}
      <div className="absolute bottom-6 right-6 z-10 flex gap-3">
        <button
          onClick={props.onToggleCamera}
          className="w-12 h-12 rounded-full flex items-center justify-center cursor-pointer transition-colors backdrop-blur-sm border border-white/10 bg-accent/90 text-white hover:bg-accent"
          title="Pause/Resume"
        >
          {props.isActive ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
          )}
        </button>
        <a
          href="/"
          className="w-12 h-12 rounded-full flex items-center justify-center cursor-pointer transition-colors backdrop-blur-sm border border-white/10 bg-error/90 text-white hover:bg-error"
          title="Stop"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" /></svg>
        </a>
      </div>

      {/* Loading Overlay */}
      {props.showLoading && (
        <div className="absolute inset-0 bg-bg-main/90 backdrop-blur-sm flex flex-col items-center justify-center z-20">
          <h3 className="text-xl font-medium mb-2">Initializing Engine...</h3>
          <p className="text-text-muted text-sm">Loading MediaPipe Pose Landmarker</p>
          <div className="w-8 h-8 mt-4 border-3 border-white/10 border-t-accent rounded-full animate-spin" />
        </div>
      )}

      {/* Timeout Overlay */}
      {props.showTimeout && (
        <div className="absolute inset-0 bg-bg-main/90 backdrop-blur-sm flex flex-col items-center justify-center z-20">
          <h3 className="text-xl font-medium mb-2">Session Paused</h3>
          <p className="text-text-muted text-sm">Camera turned off due to inactivity.</p>
          <button
            onClick={props.onResume}
            className="mt-4 px-6 py-2 bg-accent text-white text-sm font-medium rounded-md hover:bg-accent-hover transition-colors cursor-pointer"
          >
            Resume Session
          </button>
        </div>
      )}

      {/* Error Overlay */}
      {props.showError && (
        <div className="absolute inset-0 bg-error/95 flex flex-col items-center justify-center z-20">
          <h3 className="text-xl font-medium mb-2">Camera Error</h3>
          <p className="text-sm">Please allow camera access to continue.</p>
        </div>
      )}
    </div>
  );
});

export default VideoPanel;
