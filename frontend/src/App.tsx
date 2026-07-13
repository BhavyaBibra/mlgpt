import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  api, AskAnswer, Explanation, Incident, IncidentContext, Model, Performance,
} from "./api";

const C = {
  bg: "#0B0E14", panel: "#12161F", panel2: "#0F131B", border: "#1E2530",
  text: "#E6E9EF", muted: "#8A93A3", teal: "#2DD4BF",
  green: "#34D399", amber: "#FBBF24", red: "#F87171",
};

type Highlight = { type: "feature" | "event" | "performance"; value?: string } | null;

const statusColor = (s: string) => (s === "healthy" ? C.green : s === "drifting" ? C.amber : C.red);
const fmtTime = (t: number) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [incident, setIncident] = useState<Incident | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [ctx, setCtx] = useState<IncidentContext | null>(null);
  const [perf, setPerf] = useState<Performance | null>(null);
  const [events, setEvents] = useState<{ id: number; ts: string; kind: string; description: string }[]>([]);
  const [highlight, setHighlight] = useState<Highlight>(null);
  const [recent, setRecent] = useState<{ id: number; ts: string; prediction: number; model_version: string }[]>([]);
  const [rate, setRate] = useState(0);
  const rateRef = useRef<{ count: number; t: number } | null>(null);

  // one-time bootstrap: pick the first model + its latest incident
  useEffect(() => {
    api.models().then((m) => {
      setModels(m);
      const first = m[0]?.model_id ?? null;
      setModel(first);
    }).catch(() => {});
  }, []);

  // live loop — poll everything so the dashboard visibly moves
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [m, inc] = await Promise.all([api.models(), api.incidents()]);
        if (!alive) return;
        setModels(m);
        setIncidents(inc);
        const mid = model ?? m[0]?.model_id ?? null;
        if (mid && !model) setModel(mid);
        setIncident((cur) => cur ?? inc.find((i) => i.model_id === mid) ?? null);

        // throughput (predictions/sec) from the total count delta
        const total = m.reduce((s, x) => s + x.predictions, 0);
        const now = Date.now();
        if (rateRef.current) {
          const dt = (now - rateRef.current.t) / 1000;
          const dc = total - rateRef.current.count;
          if (dt > 0) setRate(Math.max(0, dc / dt));
        }
        rateRef.current = { count: total, t: now };

        if (mid) {
          const [r, p] = await Promise.all([api.recent(mid), api.performance(mid)]);
          if (!alive) return;
          setRecent(r);
          setPerf(p);
          if (!events.length) api.events(mid).then(setEvents).catch(() => {});
        }
      } catch { /* ignore transient errors */ }
    };
    tick();
    const id = setInterval(tick, 1600);
    return () => { alive = false; clearInterval(id); };
  }, [model]);

  // when an incident appears/changes, load its explanation + context
  useEffect(() => {
    if (incident) return;
    const match = incidents.find((i) => i.model_id === model) ?? null;
    if (match) setIncident(match);
  }, [incidents, model]);

  useEffect(() => {
    if (!incident) { setExplanation(null); setCtx(null); return; }
    api.ensureExplanation(incident.id).then(setExplanation).catch(() => {});
    api.context(incident.id).then(setCtx).catch(() => {});
    setHighlight(null);
  }, [incident?.id]);

  return (
    <div className="flex h-full" style={{ background: C.bg }}>
      <Sidebar models={models} model={model} onPick={(m) => {
        setModel(m);
        setIncident(incidents.find((i) => i.model_id === m) ?? null);
      }} />
      <main className="flex-1 overflow-y-auto p-6" style={{ minWidth: 0 }}>
        <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
          <Header incidents={incidents} models={models} />
          <LiveBar rate={rate} />
        </div>
        {model && perf && (
          <Detail
            model={model} perf={perf} events={events} incident={incident}
            explanation={explanation} ctx={ctx} recent={recent}
            highlight={highlight} setHighlight={setHighlight}
          />
        )}
      </main>
      <AskPanel model={model} onCite={() => {}} />
    </div>
  );
}

function Sidebar({ models, model, onPick }: { models: Model[]; model: string | null; onPick: (m: string) => void }) {
  return (
    <aside className="w-56 shrink-0 border-r p-4" style={{ borderColor: C.border, background: C.panel2 }}>
      <div className="flex items-center gap-2 mb-6">
        <span className="glow" style={{ width: 10, height: 10, borderRadius: 999, background: C.teal, display: "inline-block" }} />
        <span className="mono font-semibold tracking-wide">MLGPT</span>
      </div>
      <div className="text-xs uppercase tracking-wider mb-2" style={{ color: C.muted }}>Models</div>
      {models.map((m) => (
        <button key={m.model_id} onClick={() => onPick(m.model_id)}
          className="w-full text-left px-2 py-2 rounded-md mb-1 flex items-center gap-2"
          style={{ background: model === m.model_id ? C.panel : "transparent" }}>
          <span style={{ width: 8, height: 8, borderRadius: 999, background: statusColor(m.status) }} />
          <span className="mono text-sm truncate">{m.model_id}</span>
        </button>
      ))}
      {models.length === 0 && <div className="text-sm" style={{ color: C.muted }}>No models yet. Run the demo.</div>}
    </aside>
  );
}

function Header({ incidents, models }: { incidents: Incident[]; models: Model[] }) {
  const open = incidents.filter((i) => i.status === "open").length;
  const preds = models.reduce((s, m) => s + m.predictions, 0);
  return (
    <div className="flex items-center gap-8">
      <Stat label="Models watched" value={models.length} />
      <Stat label="Active incidents" value={open} color={open ? C.red : C.green} />
      <Stat label="Predictions logged" value={preds} format={(v) => v.toLocaleString()} />
    </div>
  );
}
function Stat({ label, value, color, format }:
  { label: string; value: number; color?: string; format?: (v: number) => string }) {
  const [flash, setFlash] = useState(false);
  const prev = useRef(value);
  useEffect(() => {
    if (prev.current !== value) { setFlash(true); const t = setTimeout(() => setFlash(false), 700); prev.current = value; return () => clearTimeout(t); }
  }, [value]);
  return (
    <div>
      <div className="text-2xl font-semibold transition-colors" style={{ color: flash ? C.teal : color ?? C.text }}>
        {format ? format(value) : value}
      </div>
      <div className="text-xs" style={{ color: C.muted }}>{label}</div>
    </div>
  );
}

function LiveBar({ rate }: { rate: number }) {
  const live = rate > 0.01;
  return (
    <div className="flex items-center gap-4 rounded-lg border px-3 py-2"
      style={{ borderColor: C.border, background: C.panel }}>
      <span className="flex items-center gap-2">
        <span className={live ? "glow" : ""}
          style={{ width: 9, height: 9, borderRadius: 999, background: live ? C.green : C.muted, display: "inline-block" }} />
        <span className="text-xs font-semibold tracking-wider" style={{ color: live ? C.green : C.muted }}>
          {live ? "LIVE" : "IDLE"}
        </span>
      </span>
      <span className="text-xs mono" style={{ color: C.muted }}>
        <span style={{ color: C.text }}>{rate.toFixed(1)}</span> pred/s
      </span>
    </div>
  );
}

function LiveFeed({ recent }: { recent: { id: number; ts: string; prediction: number; model_version: string }[] }) {
  if (!recent.length) return <div className="text-sm" style={{ color: C.muted }}>Waiting for traffic… run scripts/stream.py</div>;
  return (
    <div className="flex flex-col gap-1" style={{ maxHeight: 230, overflow: "hidden" }}>
      {recent.map((r) => {
        const risk = r.prediction;
        const col = risk >= 0.6 ? C.red : risk >= 0.35 ? C.amber : C.green;
        return (
          <div key={r.id} className="feed-row flex items-center gap-3 rounded px-2 py-1"
            style={{ background: C.panel2 }}>
            <span className="mono text-[11px]" style={{ color: C.muted, width: 44 }}>#{r.id}</span>
            <span className="mono text-[11px]" style={{ color: C.muted, width: 30 }}>v{r.model_version}</span>
            <div className="flex-1 h-1.5 rounded" style={{ background: "#161b25" }}>
              <div className="h-1.5 rounded" style={{ width: `${Math.round(risk * 100)}%`, background: col }} />
            </div>
            <span className="mono text-[11px]" style={{ color: col, width: 44, textAlign: "right" }}>
              {(risk * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Detail(props: {
  model: string; perf: Performance; events: any[]; incident: Incident | null;
  explanation: Explanation | null; ctx: IncidentContext | null;
  recent: { id: number; ts: string; prediction: number; model_version: string }[];
  highlight: Highlight; setHighlight: (h: Highlight) => void;
}) {
  const { perf, events, incident, explanation, ctx, recent, highlight, setHighlight } = props;
  const deploy = events.find((e) => e.kind === "deploy");
  const onset = ctx?.drift.onset;

  return (
    <div className="grid gap-6" style={{ gridTemplateColumns: "1fr" }}>
      <Panel title="Live predictions" subtitle="streaming in · risk score">
        <LiveFeed recent={recent} />
      </Panel>

      <Panel title="Accuracy over time" subtitle="deploy markers + incident region">
        <AccuracyChart perf={perf} deployTs={deploy?.ts} onset={onset} highlight={highlight} />
      </Panel>

      <Panel title="Incident" subtitle={incident ? `#${incident.id} · ${incident.severity}` : "none"}>
        {explanation && ctx
          ? <ExplanationCard e={explanation} ctx={ctx} setHighlight={setHighlight} highlight={highlight} />
          : <div className="text-sm" style={{ color: C.muted }}>No open incident. The model looks healthy. 🟢</div>}
      </Panel>

      <Panel title="Feature drift" subtitle="PSI vs reference">
        <Heatmap ctx={ctx} highlight={highlight} />
      </Panel>
    </div>
  );
}

function Panel({ title, subtitle, children }: any) {
  return (
    <section className="rounded-xl border p-4" style={{ borderColor: C.border, background: C.panel }}>
      <div className="flex items-baseline justify-between gap-3 flex-wrap mb-3">
        <h2 className="text-sm font-semibold whitespace-nowrap">{title}</h2>
        <span className="text-xs mono" style={{ color: C.muted }}>{subtitle}</span>
      </div>
      {children}
    </section>
  );
}

function AccuracyChart({ perf, deployTs, onset, highlight }:
  { perf: Performance; deployTs?: string; onset?: string; highlight: Highlight }) {
  const data = perf.timeline.map((p) => ({ t: new Date(p.window_start).getTime(), accuracy: p.accuracy, auc: p.auc }));
  if (!data.length) return <div className="text-sm" style={{ color: C.muted }}>No labelled traffic yet.</div>;
  const deployMs = deployTs ? new Date(deployTs).getTime() : undefined;
  const onsetMs = onset ? new Date(onset).getTime() : undefined;
  const endMs = data[data.length - 1].t;
  const eventHot = highlight?.type === "event";
  const perfHot = highlight?.type === "performance";
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={C.border} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={fmtTime}
          stroke={C.muted} fontSize={11} />
        <YAxis domain={[0.5, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke={C.muted} fontSize={11} />
        <Tooltip contentStyle={{ background: C.panel2, border: `1px solid ${C.border}`, borderRadius: 8 }}
          labelFormatter={(t) => fmtTime(t as number)} formatter={(v: any, n) => [typeof v === "number" ? v.toFixed(3) : v, n]} />
        {onsetMs && (
          <ReferenceArea x1={onsetMs} x2={endMs} fill={C.red} fillOpacity={perfHot ? 0.22 : 0.08}
            stroke={perfHot ? C.red : "none"} />
        )}
        {deployMs && (
          <ReferenceLine x={deployMs} stroke={eventHot ? C.teal : C.muted}
            strokeWidth={eventHot ? 2.5 : 1.5} strokeDasharray={eventHot ? undefined : "4 3"}
            label={{ value: "v2.3", fill: eventHot ? C.teal : C.muted, fontSize: 11, position: "top" }} />
        )}
        <Line type="monotone" dataKey="accuracy" stroke={C.teal} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="auc" stroke={C.amber} strokeWidth={1.5} dot={false} strokeDasharray="4 3" isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function Heatmap({ ctx, highlight }: { ctx: IncidentContext | null; highlight: Highlight }) {
  if (!ctx) return <div className="text-sm" style={{ color: C.muted }}>—</div>;
  const feats = ctx.drift.flagged_features.slice(0, 10);
  const nulls = ctx.drift.null_spikes ?? [];
  if (!feats.length && !nulls.length)
    return <div className="text-sm" style={{ color: C.muted }}>No feature drift flagged (performance-only incident).</div>;
  const max = Math.max(1, ...feats.map((f) => f.value));
  return (
    <div className="flex flex-col gap-1.5">
      {feats.map((f) => {
        const hot = highlight?.type === "feature" && highlight.value === f.feature;
        const w = Math.max(6, Math.min(100, (f.value / max) * 100));
        return (
          <div key={f.report_id} className="flex items-center gap-2 rounded px-1"
            style={{ outline: hot ? `1px solid ${C.teal}` : "none", background: hot ? "rgba(45,212,191,0.08)" : "transparent" }}>
            <span className="mono text-xs w-28 truncate" style={{ color: C.muted }}>{f.feature}</span>
            <div className="flex-1 h-4 rounded" style={{ background: C.panel2 }}>
              <div className="h-4 rounded" style={{ width: `${w}%`, background: f.value >= 1 ? C.red : C.amber }} />
            </div>
            <span className="mono text-xs w-12 text-right">{f.value.toFixed(2)}</span>
          </div>
        );
      })}
      {nulls.map((n) => (
        <div key={n.report_id} className="flex items-center gap-2 px-1">
          <span className="mono text-xs w-28 truncate" style={{ color: C.muted }}>{n.feature}</span>
          <span className="text-xs" style={{ color: C.red }}>null spike {(n.null_rate * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

function ExplanationCard({ e, ctx, setHighlight, highlight }:
  { e: Explanation; ctx: IncidentContext; setHighlight: (h: Highlight) => void; highlight: Highlight }) {
  const refFeature = useMemo(() => {
    const m: Record<number, string> = {};
    ctx.drift.flagged_features.forEach((f) => (m[f.report_id] = f.feature));
    (ctx.drift.null_spikes ?? []).forEach((n) => (m[n.report_id] = n.feature));
    return m;
  }, [ctx]);
  const confColor = e.confidence === "high" ? C.red : e.confidence === "medium" ? C.amber : C.muted;

  const chipFor = (ev: { type: string; ref_id: number | null; detail: string }, i: number) => {
    let label = ev.detail;
    let onClick = () => setHighlight(null);
    let active = false;
    if (ev.type === "drift" || ev.type === "data_quality") {
      const feat = (ev.ref_id != null && refFeature[ev.ref_id]) || ev.detail.split(" ")[0];
      label = `${feat} · ${ev.detail.split(" ").slice(1).join(" ")}`;
      onClick = () => setHighlight({ type: "feature", value: feat });
      active = highlight?.type === "feature" && highlight.value === feat;
    } else if (ev.type === "event") {
      label = ev.detail;
      onClick = () => setHighlight({ type: "event" });
      active = highlight?.type === "event";
    } else if (ev.type === "performance") {
      label = ev.detail;
      onClick = () => setHighlight({ type: "performance" });
      active = highlight?.type === "performance";
    }
    return (
      <button key={i} onClick={onClick}
        className="text-xs mono px-2 py-1 rounded-md border transition-colors"
        style={{
          borderColor: active ? C.teal : C.border,
          background: active ? "rgba(45,212,191,0.12)" : C.panel2,
          color: active ? C.teal : C.text,
        }}>
        {label}
      </button>
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="text-[15px] leading-snug">{e.root_cause}</div>
      <div className="flex items-center gap-2">
        <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "rgba(255,255,255,0.05)", color: confColor }}>
          confidence: {e.confidence}
        </span>
        <span className="text-xs" style={{ color: C.muted }}>signature: {ctx.signature.label}</span>
      </div>
      <div>
        <div className="text-xs uppercase tracking-wider mb-1.5" style={{ color: C.muted }}>
          Evidence — click to highlight
        </div>
        <div className="flex flex-wrap gap-1.5">{e.evidence.map(chipFor)}</div>
      </div>
      <div className="text-sm mt-1 pt-3 border-t" style={{ borderColor: C.border, color: C.muted }}>
        <span style={{ color: C.text }}>Suggested action:</span> {e.suggested_action}
      </div>
    </div>
  );
}

function AskPanel({ model }: { model: string | null; onCite: () => void }) {
  const [q, setQ] = useState("");
  const [msgs, setMsgs] = useState<{ q: string; a: AskAnswer }[]>([]);
  const [busy, setBusy] = useState(false);
  const suggested = ["Why did accuracy drop this week?", "What changed after the last deploy?", "Is the model healthy today?"];

  const send = async (text: string) => {
    if (!text.trim() || busy) return;
    setBusy(true); setQ("");
    try {
      const a = await api.ask(text, model ?? undefined);
      setMsgs((m) => [...m, { q: text, a }]);
    } catch { /* ignore */ } finally { setBusy(false); }
  };

  return (
    <aside className="w-80 shrink-0 border-l flex flex-col" style={{ borderColor: C.border, background: C.panel2 }}>
      <div className="p-4 border-b" style={{ borderColor: C.border }}>
        <div className="text-sm font-semibold">Ask MLGPT</div>
        <div className="text-xs mono" style={{ color: C.muted }}>{model ? `about: ${model}` : "no model"}</div>
      </div>
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {msgs.length === 0 && (
          <div className="flex flex-col gap-2">
            <div className="text-xs" style={{ color: C.muted }}>Try:</div>
            {suggested.map((s) => (
              <button key={s} onClick={() => send(s)} className="text-left text-xs px-2 py-1.5 rounded-md border"
                style={{ borderColor: C.border, color: C.text }}>{s}</button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <div className="text-xs self-end px-2 py-1 rounded-md" style={{ background: C.panel, color: C.text }}>{m.q}</div>
            <div className="text-sm leading-snug">{m.a.answer}</div>
            {m.a.citations?.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {m.a.citations.map((c, j) => (
                  <span key={j} className="text-[11px] mono px-1.5 py-0.5 rounded border"
                    style={{ borderColor: C.border, color: C.teal }}>{c.type}#{c.ref}</span>
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="text-xs" style={{ color: C.muted }}>thinking…</div>}
      </div>
      <div className="p-3 border-t flex gap-2" style={{ borderColor: C.border }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send(q)}
          placeholder="Ask why…" className="flex-1 text-sm px-2 py-1.5 rounded-md outline-none"
          style={{ background: C.panel, border: `1px solid ${C.border}`, color: C.text }} />
        <button onClick={() => send(q)} className="text-sm px-3 rounded-md"
          style={{ background: C.teal, color: "#04211d" }}>→</button>
      </div>
    </aside>
  );
}
