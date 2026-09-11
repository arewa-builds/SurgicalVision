import { useEffect, useMemo, useRef, useState } from "react";
import { getAnalysis, overlayUrl, startDemo, uploadVideo } from "./api";
import { PIPELINE_STEPS, type AnalysisResult, type TimelineEvent } from "./types";

type View = "home" | "running" | "report";

export default function App() {
  const [view, setView] = useState<View>("home");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!analysis || (analysis.status !== "queued" && analysis.status !== "processing")) return;
    const tick = window.setInterval(async () => {
      try {
        const next = await getAnalysis(analysis.id);
        setAnalysis(next);
        if (next.status === "complete") setView("report");
        if (next.status === "failed") {
          setError(next.error || "Analysis failed");
          setView("home");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Polling failed");
      }
    }, 400);
    return () => window.clearInterval(tick);
  }, [analysis]);

  async function runDemo(profile: "efficient" | "novice") {
    setBusy(true);
    setError(null);
    try {
      const created = await startDemo(profile);
      setAnalysis(created);
      setView("running");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start demo");
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(file: File) {
    setBusy(true);
    setError(null);
    try {
      const created = await uploadVideo(file);
      setAnalysis(created);
      setView("running");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={() => { setView("home"); setAnalysis(null); }}>
          <span className="mark" aria-hidden />
          SurgicalVision
        </button>
        <p className="tag">AI surgical technique coach</p>
        <span className="pill">Research metrics · not clinical scores</span>
      </header>

      {error && <div className="banner error">{error}</div>}

      {view === "home" && <Home busy={busy} onDemo={runDemo} onUpload={onUpload} />}
      {view === "running" && analysis && <Running analysis={analysis} />}
      {view === "report" && analysis && (
        <Report
          analysis={analysis}
          onReset={() => {
            setView("home");
            setAnalysis(null);
          }}
        />
      )}
    </div>
  );
}

function Home({
  busy,
  onDemo,
  onUpload,
}: {
  busy: boolean;
  onDemo: (profile: "efficient" | "novice") => void;
  onUpload: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  return (
    <main className="home">
      <section className="hero">
        <p className="eyebrow">Laparoscopic & robotic simulation</p>
        <h1>See the technique, not just the task.</h1>
        <p className="lede">
          Recorded surgery in. Instrument tracks, gesture sequence, motion economy, and a review
          timeline out. Scores are research-derived technique metrics until expert validation.
        </p>
        <ol className="pipe">
          {["Detect", "Track", "Gestures", "Metrics", "Score", "Explain"].map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </section>

      <section className="actions">
        <article className="case">
          <h2>Efficient simulation</h2>
          <p>Deliberate bimanual suturing: reach, position, grasp, suture, knot, release.</p>
          <button disabled={busy} onClick={() => onDemo("efficient")}>
            Run efficient case
          </button>
        </article>
        <article className="case novice">
          <h2>Novice simulation</h2>
          <p>Overshoot, tremor, drop-and-regrasp, extra repositioning — the coaching contrast.</p>
          <button disabled={busy} onClick={() => onDemo("novice")}>
            Run novice case
          </button>
        </article>
        <article
          className={`case upload ${drag ? "drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            const file = e.dataTransfer.files[0];
            if (file) onUpload(file);
          }}
        >
          <h2>Your recording</h2>
          <p>Upload laparoscopic or robotic simulation video. Prototype detection prefers high-contrast instruments.</p>
          <button disabled={busy} onClick={() => inputRef.current?.click()}>
            Upload video
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.mkv,.avi"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUpload(file);
            }}
          />
        </article>
      </section>
    </main>
  );
}

function Running({ analysis }: { analysis: AnalysisResult }) {
  const active = analysis.step || PIPELINE_STEPS[0];
  return (
    <main className="running">
      <h1>Analyzing technique</h1>
      <p className="lede">
        {analysis.source.replace("_", " · ")} · {analysis.progress}%
      </p>
      <div className="bar">
        <span style={{ width: `${analysis.progress}%` }} />
      </div>
      <ol className="steps">
        {PIPELINE_STEPS.map((step) => (
          <li key={step} className={step === active ? "on" : analysis.progress >= 100 ? "on" : ""}>
            {step}
          </li>
        ))}
      </ol>
    </main>
  );
}

function Report({ analysis, onReset }: { analysis: AnalysisResult; onReset: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);

  function seek(time: number) {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = time;
    void el.play();
  }

  return (
    <main className="report">
      <section className="stage">
        <div className="player">
          <video
            ref={videoRef}
            src={overlayUrl(analysis.id)}
            controls
            playsInline
            onTimeUpdate={(e) => setT(e.currentTarget.currentTime)}
          />
        </div>
        <Timeline
          duration={analysis.duration_s}
          current={t}
          events={analysis.timeline_events}
          onSeek={seek}
        />
      </section>

      <aside className="scores">
        <ScoreRing value={analysis.overall_score ?? 0} />
        <ul className="dims">
          {analysis.dimensions.map((d) => (
            <li key={d.key}>
              <div>
                <span>{d.label}</span>
                <strong>{d.score.toFixed(0)}</strong>
              </div>
              <div className="meter">
                <span style={{ width: `${d.score}%` }} />
              </div>
              <small>{d.detail}</small>
            </li>
          ))}
        </ul>
      </aside>

      <section className="gestures">
        <h2>Gesture sequence</h2>
        <Sequence expected={analysis.sequence_expected} observed={analysis.sequence_observed} />
        <ul className="glist">
          {analysis.gestures
            .filter((g) => g.label !== "idle")
            .slice(0, 12)
            .map((g, i) => (
              <li key={`${g.instrument}-${g.start_s}-${i}`}>
                <button onClick={() => seek(g.start_s)}>
                  {fmt(g.start_s)} · {g.instrument} · {g.label}
                </button>
              </li>
            ))}
        </ul>
      </section>

      <section className="metrics">
        <h2>Motion metrics</h2>
        <div className="cards">
          {analysis.instruments.map((ins) => (
            <article key={ins.label}>
              <h3>{ins.label} instrument</h3>
              <dl>
                <Stat k="Path length" v={`${ins.path_length_px.toFixed(0)} px`} />
                <Stat k="Path efficiency" v={ins.path_efficiency.toFixed(2)} />
                <Stat k="Mean velocity" v={`${ins.mean_velocity_px_s.toFixed(0)} px/s`} />
                <Stat k="Peak velocity" v={`${ins.max_velocity_px_s.toFixed(0)} px/s`} />
                <Stat k="Mean jerk" v={`${ins.mean_jerk_px_s3.toFixed(0)} px/s³`} />
                <Stat k="Idle" v={`${(ins.idle_fraction * 100).toFixed(0)}%`} />
                <Stat k="Corrective moves" v={String(ins.corrective_movements)} />
                <Stat k="Workspace use" v={`${(ins.workspace_utilization * 100).toFixed(1)}%`} />
              </dl>
            </article>
          ))}
          {analysis.bimanual && (
            <article>
              <h3>Bimanual coordination</h3>
              <dl>
                <Stat k="Time sync" v={analysis.bimanual.time_sync.toFixed(2)} />
                <Stat k="Velocity corr." v={analysis.bimanual.velocity_correlation.toFixed(2)} />
                <Stat k="Mean tip distance" v={`${analysis.bimanual.mean_tip_distance_px.toFixed(0)} px`} />
                <Stat k="Dual activity" v={`${(analysis.bimanual.dual_activity_fraction * 100).toFixed(0)}%`} />
              </dl>
            </article>
          )}
        </div>
      </section>

      <footer className="foot">
        <p>{analysis.disclaimer}</p>
        {analysis.notes.length > 0 && (
          <ul>
            {analysis.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
        <button className="ghost" onClick={onReset}>
          New analysis
        </button>
      </footer>
    </main>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}

function ScoreRing({ value }: { value: number }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - Math.min(value, 100) / 100);
  return (
    <div className="ring">
      <svg viewBox="0 0 140 140" aria-hidden>
        <circle cx="70" cy="70" r={r} className="track" />
        <circle
          cx="70"
          cy="70"
          r={r}
          className="val"
          strokeDasharray={c}
          strokeDashoffset={offset}
        />
      </svg>
      <div>
        <strong>{value.toFixed(0)}</strong>
        <span>Overall / 100</span>
      </div>
    </div>
  );
}

function Sequence({ expected, observed }: { expected: string[]; observed: string[] }) {
  const extra = useMemo(() => {
    const set = new Set(expected);
    return new Set(observed.filter((g) => !set.has(g) || observed.filter((x) => x === g).length > 1));
  }, [expected, observed]);

  return (
    <div className="seq">
      <p>
        <span>Expected</span>
        {expected.map((g) => (
          <em key={`e-${g}`}>{g}</em>
        ))}
      </p>
      <p>
        <span>Observed</span>
        {observed.length === 0 && <em className="muted">none</em>}
        {observed.map((g, i) => (
          <em key={`o-${g}-${i}`} className={extra.has(g) || g === "reposition" ? "warn" : ""}>
            {g}
          </em>
        ))}
      </p>
    </div>
  );
}

function Timeline({
  duration,
  current,
  events,
  onSeek,
}: {
  duration: number;
  current: number;
  events: TimelineEvent[];
  onSeek: (t: number) => void;
}) {
  const pct = duration > 0 ? Math.min(100, (current / duration) * 100) : 0;
  return (
    <div className="timeline">
      <button
        className="rail"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const u = (e.clientX - rect.left) / rect.width;
          onSeek(u * duration);
        }}
      >
        <span className="fill" style={{ width: `${pct}%` }} />
        {events.map((ev) => (
          <i
            key={`${ev.t}-${ev.title}`}
            className={ev.severity}
            style={{ left: `${duration ? (ev.t / duration) * 100 : 0}%` }}
            title={`${fmt(ev.t)} ${ev.title}`}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(ev.t);
            }}
          />
        ))}
      </button>
      <ul>
        {events.map((ev) => (
          <li key={`${ev.t}-${ev.title}`}>
            <button className={ev.severity} onClick={() => onSeek(ev.t)}>
              <strong>{fmt(ev.t)}</strong> {ev.title}
              <span>{ev.detail}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function fmt(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
