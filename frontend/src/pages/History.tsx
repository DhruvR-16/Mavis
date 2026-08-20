import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Chart } from 'chart.js/auto';

import { clearHistory, loadHistory, removeSession } from '../storage/history';
import { dailyTrend, improvement, summarise } from '../storage/stats';
import type { SessionRecord } from '../storage/types';
import './History.css';

const EXERCISE_LABEL: Record<string, string> = {
  bicep: 'Bicep Curl',
  shoulder: 'Shoulder Press',
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function formatDuration(seconds: number): string {
  if (!seconds) return '—';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m ? `${m}m ${s}s` : `${s}s`;
}

export default function History() {
  const [sessions, setSessions] = useState<SessionRecord[] | null>(null);
  const chartRef = useRef<Chart | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    void loadHistory().then((rows) => {
      if (!cancelled) setSessions(rows);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const summary = useMemo(() => (sessions ? summarise(sessions) : null), [sessions]);
  const trend = useMemo(() => (sessions ? dailyTrend(sessions) : []), [sessions]);
  const delta = useMemo(() => (sessions ? improvement(sessions) : null), [sessions]);

  useEffect(() => {
    chartRef.current?.destroy();
    chartRef.current = null;
    if (!canvasRef.current || trend.length < 2) return;

    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        labels: trend.map((p) => p.day.slice(5)),
        datasets: [
          {
            data: trend.map((p) => p.avgQuality),
            borderColor: '#4ade80',
            backgroundColor: 'rgba(74,222,128,0.12)',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#6b6b7e' }, grid: { display: false } },
          y: {
            min: 0,
            max: 100,
            ticks: { color: '#6b6b7e' },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [trend]);

  const handleDelete = async (id: string) => {
    await removeSession(id);
    setSessions(await loadHistory());
  };

  const handleClearAll = async () => {
    if (!window.confirm('Delete all workout history? This cannot be undone.')) return;
    await clearHistory();
    setSessions(await loadHistory());
  };

  if (sessions === null) {
    return (
      <div className="history-page">
        <div className="history-container">
          <p className="history-empty-text">Loading history…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="history-page">
      <div className="history-container">
        <nav className="history-nav">
          <Link className="back-link" to="/">← Home</Link>
          <h1>Progress</h1>
          {sessions.length > 0 && (
            <button type="button" className="clear-btn" onClick={() => void handleClearAll()}>
              Clear history
            </button>
          )}
        </nav>

        {sessions.length === 0 ? (
          <div className="history-empty">
            <div className="history-empty-icon">📈</div>
            <h2>No sessions yet</h2>
            <p>
              Finish a workout and it'll show up here. Mavis keeps your history on this
              device so you can see whether your form is actually improving.
            </p>
            <Link className="history-cta" to="/">Start a workout</Link>
          </div>
        ) : (
          <>
            <div className="history-stats">
              <div className="hstat">
                <div className="hstat-val">{summary!.totalSessions}</div>
                <div className="hstat-lbl">Sessions</div>
              </div>
              <div className="hstat">
                <div className="hstat-val">{summary!.totalReps}</div>
                <div className="hstat-lbl">Total reps</div>
              </div>
              <div className="hstat">
                <div className="hstat-val">{summary!.avgQuality}</div>
                <div className="hstat-lbl">Avg quality</div>
              </div>
              <div className="hstat">
                <div className="hstat-val">
                  {summary!.currentStreakDays}
                  <span className="hstat-unit">d</span>
                </div>
                <div className="hstat-lbl">Streak</div>
              </div>
            </div>

            {delta !== null && (
              <div className={`trend-callout ${delta >= 0 ? 'up' : 'down'}`}>
                <strong>
                  {delta >= 0 ? '▲' : '▼'} {Math.abs(delta)} points
                </strong>
                <span>
                  {delta >= 0
                    ? 'average quality is up compared with your earlier sessions'
                    : 'average quality is down compared with your earlier sessions'}
                </span>
              </div>
            )}

            <div className="history-section">
              <div className="section-label">Quality over time</div>
              {trend.length < 2 ? (
                <p className="history-hint">
                  Train on another day to start seeing a trend line.
                </p>
              ) : (
                <div className="history-chart">
                  <canvas ref={canvasRef} />
                </div>
              )}
            </div>

            {summary!.topFault && (
              <div className="history-section">
                <div className="section-label">Most common fault</div>
                <div className="fault-callout">{summary!.topFault}</div>
              </div>
            )}

            <div className="history-section">
              <div className="section-label">Sessions</div>
              <div className="session-list">
                {sessions.map((s) => (
                  <div key={s.id} className="session-row">
                    <div className="session-main">
                      <div className="session-title">{EXERCISE_LABEL[s.exercise] ?? s.exercise}</div>
                      <div className="session-meta">
                        {formatDate(s.date)} · {s.mode === 'program' ? 'Program' : 'Free'} ·{' '}
                        {formatDuration(s.durationSec)}
                      </div>
                      {s.topFault && <div className="session-fault">{s.topFault}</div>}
                    </div>
                    <div className="session-numbers">
                      <div className="session-reps">
                        <span>{s.totalReps}</span> reps
                      </div>
                      <div
                        className={`session-quality ${
                          s.avgQuality >= 80 ? 'great' : s.avgQuality >= 60 ? 'ok' : 'bad'
                        }`}
                      >
                        {s.avgQuality}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="session-delete"
                      aria-label="Delete session"
                      onClick={() => void handleDelete(s.id)}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <p className="history-privacy">
              Stored on this device only — nothing is uploaded.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
