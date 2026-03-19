import type { WorkoutState, FeedbackType } from '../hooks/useWorkoutAnalyzer';

const ICON_CHECK = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);
const ICON_WARN = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
const ICON_INFO = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" />
  </svg>
);

function getMasterText(state: WorkoutState): string {
  if (state.feedbackType === 'good') return 'GOOD FORM';
  if (state.feedbackType === 'bad') return 'BAD FORM';
  if (state.stage === 'UP') return 'HOLD';
  return 'READY';
}

function feedbackBorderColor(type: FeedbackType): string {
  switch (type) {
    case 'good': return 'border-success';
    case 'bad': return 'border-error';
    default: return 'border-border';
  }
}

function feedbackTextColor(type: FeedbackType): string {
  switch (type) {
    case 'good': return 'text-success';
    case 'bad': return 'text-error';
    default: return 'text-text-main';
  }
}

interface SidebarProps {
  state: WorkoutState;
}

export default function Sidebar({ state }: SidebarProps) {
  const cueIcon = state.feedbackType === 'good' ? ICON_CHECK : state.feedbackType === 'bad' ? ICON_WARN : ICON_INFO;
  const cueIconBg = state.feedbackType === 'good'
    ? 'bg-success/10 text-success'
    : state.feedbackType === 'bad'
      ? 'bg-error/10 text-error'
      : 'bg-accent/15 text-accent';

  return (
    <aside className="w-[340px] flex flex-col gap-5 pl-2 overflow-y-auto">
      {/* Live Feedback */}
      <div>
        <div className="text-xs text-text-muted font-medium mb-2">Live Feedback</div>
        <div className={`bg-bg-panel rounded-lg p-8 text-center flex flex-col items-center justify-center min-h-[120px] border transition-colors ${feedbackBorderColor(state.feedbackType)}`}>
          <span className={`text-2xl font-semibold tracking-tight ${feedbackTextColor(state.feedbackType)}`}>
            {getMasterText(state)}
          </span>
        </div>
      </div>

      {/* Real-Time Cues */}
      <div>
        <div className="text-xs text-text-muted font-medium mb-2">Real-Time Cue</div>
        <div className="bg-bg-panel rounded-lg p-4 flex gap-4 items-start border border-border">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${cueIconBg}`}>
            {cueIcon}
          </div>
          <div>
            <h4 className="text-sm font-medium text-text-main">{state.cueTitle}</h4>
            <p className="text-sm text-text-muted">{state.feedback}</p>
          </div>
        </div>
      </div>

      {/* Biometrics */}
      <div>
        <div className="text-xs text-text-muted font-medium mb-2">Current Biometrics</div>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-bg-panel border border-border rounded-lg p-4 text-center">
            <div className="text-text-muted text-base mb-1">📐</div>
            <div className="text-xl font-semibold">{Math.round(state.elbowAngle)}°</div>
            <div className="text-xs text-text-muted">Elbow Angle</div>
          </div>
          <div className="bg-bg-panel border border-border rounded-lg p-4 text-center">
            <div className="text-text-muted text-base mb-1">⏱️</div>
            <div className="text-xl font-semibold">{state.stage}</div>
            <div className="text-xs text-text-muted">Phase</div>
          </div>
        </div>
      </div>

      {/* Finish Button */}
      <a
        href="/"
        className="mt-auto block w-full py-3 text-center bg-bg-panel border border-border rounded-lg text-sm font-medium text-text-main hover:bg-bg-card transition-colors"
      >
        Finish Workout Session
      </a>

      {/* GitHub Link */}
      <div className="text-center pb-2">
        <a
          href="https://github.com/DhruvR-16/Mavis"
          target="_blank"
          rel="noopener noreferrer"
          className="text-text-muted text-xs inline-flex items-center gap-1.5 hover:text-accent transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
          </svg>
          View Source Code
        </a>
      </div>
    </aside>
  );
}
