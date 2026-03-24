import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

type ExerciseType = 'bicep' | 'shoulder';
type SessionMode = 'free' | 'program';

type LastSessionStats = {
  total_reps?: number;
  avg_quality?: number;
  good_reps?: number;
  most_common_fault?: string;
};

const DEFAULT_STATS = {
  reps: '—',
  quality: '—',
  good: '—',
  fault: '—',
};

export default function Home() {
  const navigate = useNavigate();

  const [selectedExercise, setSelectedExercise] = useState<ExerciseType | null>(null);
  const [sessionMode, setSessionMode] = useState<SessionMode>('free');
  const [voiceOn, setVoiceOn] = useState(true);

  const [sets, setSets] = useState(3);
  const [reps, setReps] = useState(10);
  const [rest, setRest] = useState(60);

  const [stats, setStats] = useState(DEFAULT_STATS);

  useEffect(() => {
    const raw = localStorage.getItem('mavis_last_session');
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw) as LastSessionStats;
      setStats({
        reps: String(parsed.total_reps ?? '—'),
        quality: String(parsed.avg_quality ?? '—'),
        good: String(parsed.good_reps ?? '—'),
        fault: parsed.most_common_fault || 'None',
      });
    } catch {
      setStats(DEFAULT_STATS);
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Enter') return;
      if (!selectedExercise) return;
      launchSession();
    };

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedExercise, sessionMode, sets, reps, rest, voiceOn]);

  const launchLabel = useMemo(() => {
    if (!selectedExercise) return 'Select an exercise to begin';
    if (selectedExercise === 'bicep') {
      return sessionMode === 'program' ? 'Start Bicep Curl Program →' : 'Start Bicep Curl Session →';
    }
    return sessionMode === 'program' ? 'Start Shoulder Press Program →' : 'Start Shoulder Press Session →';
  }, [selectedExercise, sessionMode]);

  const launchSession = () => {
    if (!selectedExercise) return;

    const params = new URLSearchParams({
      exercise: selectedExercise,
      mode: sessionMode,
      voice: voiceOn ? '1' : '0',
    });

    if (sessionMode === 'program') {
      params.set('sets', String(sets));
      params.set('reps', String(reps));
      params.set('rest', String(rest));
    }

    navigate(`/workout?${params.toString()}`);
  };

  return (
    <div className="home-page">
      <div className="home-container">
        <nav className="home-nav anim">
          <div className="logo">
            <div className="logo-mark">M</div>
            <div>
              <div className="logo-text">Mavis</div>
              <div className="logo-sub">AI Fitness Coach</div>
            </div>
          </div>
          <div className="nav-badge">v2.0 - BETA</div>
        </nav>

        <div className="hero anim anim-d1">
          <div className="hero-eyebrow">Real-time pose analysis</div>
          <h1>
            Train smarter.<br />
            <span>Move better.</span>
          </h1>
          <p className="hero-sub">
            Mavis watches your form in real-time using computer vision - catching drift, asymmetry, and bad reps before they become bad habits.
          </p>
        </div>

        <div className="section-label">Last session</div>
        <div className="stats-strip anim anim-d2">
          <div className="stat-card">
            <div className="stat-label">Total reps</div>
            <div className="stat-value">{stats.reps}</div>
            <div className="stat-sub">across all sets</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Avg quality</div>
            <div className="stat-value">{stats.quality}</div>
            <div className="stat-sub">out of 100</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Good reps</div>
            <div className="stat-value">{stats.good}</div>
            <div className="stat-sub">&gt;=60 quality score</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Top fault</div>
            <div className="stat-value stat-value-fault">{stats.fault}</div>
            <div className="stat-sub">most common issue</div>
          </div>
        </div>

        <div className="section-label anim anim-d3">Choose exercise</div>
        <div className="exercise-grid anim anim-d3">
          <button
            type="button"
            className={`exercise-card bicep ${selectedExercise === 'bicep' ? 'selected' : ''}`}
            onClick={() => setSelectedExercise('bicep')}
          >
            <div className="ex-check">✓</div>
            <span className="ex-icon">💪</span>
            <div className="ex-name">Bicep Curl</div>
            <p className="ex-desc">Full ROM tracking with elbow drift detection and tempo control.</p>
            <div className="ex-tags">
              <span className="tag">Elbow angle</span>
              <span className="tag">Drift check</span>
              <span className="tag">Tempo</span>
            </div>
          </button>

          <button
            type="button"
            className={`exercise-card shoulder ${selectedExercise === 'shoulder' ? 'selected' : ''}`}
            onClick={() => setSelectedExercise('shoulder')}
          >
            <div className="ex-check">✓</div>
            <span className="ex-icon">🏋️</span>
            <div className="ex-name">Shoulder Press</div>
            <p className="ex-desc">Bilateral symmetry tracking - catches when one arm works harder than the other.</p>
            <div className="ex-tags">
              <span className="tag">Bilateral</span>
              <span className="tag">Lockout</span>
              <span className="tag">Symmetry</span>
            </div>
          </button>
        </div>

        <div className="section-label anim anim-d3">Session mode</div>
        <div className="mode-row anim anim-d3">
          <button
            type="button"
            className={`mode-btn ${sessionMode === 'free' ? 'active' : ''} ${selectedExercise === 'shoulder' && sessionMode === 'free' ? 'shoulder-mode' : ''}`}
            onClick={() => setSessionMode('free')}
          >
            Free mode
          </button>
          <button
            type="button"
            className={`mode-btn ${sessionMode === 'program' ? 'active' : ''} ${selectedExercise === 'shoulder' && sessionMode === 'program' ? 'shoulder-mode' : ''}`}
            onClick={() => setSessionMode('program')}
          >
            Workout program
          </button>
        </div>

        {sessionMode === 'program' && (
          <div className="program-section anim anim-d4">
            <div className="program-card">
              <h2>Build your workout</h2>
              <p className="desc">
                Mavis will guide you through sets, enforce rest periods, and flag any set with too many bad reps so you repeat it.
              </p>
              <div className="program-grid">
                <div className="field-group">
                  <label htmlFor="p-sets">Sets</label>
                  <input
                    id="p-sets"
                    type="number"
                    value={sets}
                    min={1}
                    max={20}
                    onChange={(event) => setSets(Math.max(1, Math.min(20, Number(event.target.value) || 1)))}
                  />
                </div>
                <div className="field-group">
                  <label htmlFor="p-reps">Reps per set</label>
                  <input
                    id="p-reps"
                    type="number"
                    value={reps}
                    min={1}
                    max={50}
                    onChange={(event) => setReps(Math.max(1, Math.min(50, Number(event.target.value) || 1)))}
                  />
                </div>
                <div className="field-group">
                  <label htmlFor="p-rest">Rest (seconds)</label>
                  <input
                    id="p-rest"
                    type="number"
                    value={rest}
                    min={10}
                    max={300}
                    onChange={(event) => setRest(Math.max(10, Math.min(300, Number(event.target.value) || 10)))}
                  />
                </div>
              </div>

              <div className="wrong-set-box">
                <strong>WRONG-SET DETECTION</strong>
                <br />
                If 3 or more reps in a set are bad quality, that set will be marked invalid and you'll need to repeat it - just like a real trainer would make you do.
              </div>
            </div>
          </div>
        )}

        <div className="voice-row anim anim-d4">
          <div>
            <div className="voice-label"><span>🎙</span> Voice coaching</div>
            <div className="voice-sub">Mavis speaks live cues based on what it sees.</div>
          </div>
          <label className="toggle" htmlFor="voice-toggle">
            <input
              id="voice-toggle"
              type="checkbox"
              checked={voiceOn}
              onChange={(event) => setVoiceOn(event.target.checked)}
            />
            <div className="toggle-track"><div className="toggle-thumb" /></div>
          </label>
        </div>

        <button
          type="button"
          className={`cta-btn anim anim-d4 ${selectedExercise === 'shoulder' ? 'shoulder-btn' : ''}`}
          onClick={launchSession}
          disabled={!selectedExercise}
        >
          {launchLabel}
        </button>

        <footer>
          <p>Mavis v2.0 - MediaPipe + LSTM pose engine</p>
          <p>Built with love by Dhruv &amp; team</p>
        </footer>
      </div>
    </div>
  );
}
