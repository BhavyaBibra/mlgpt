const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export type ModelStatus = "healthy" | "drifting" | "degraded";
export interface Model {
  model_id: string;
  predictions: number;
  last_prediction: string | null;
  status: ModelStatus;
}
export interface Incident {
  id: number;
  model_id: string;
  opened_ts: string;
  severity: string;
  status: string;
  trigger_summary: string;
}
export interface EvidenceItem {
  type: string;
  ref_id: number | null;
  detail: string;
}
export interface Explanation {
  id: number;
  incident_id: number;
  ts: string;
  summary: string;
  root_cause: string;
  evidence: EvidenceItem[];
  confidence: "high" | "medium" | "low" | string;
  suggested_action: string;
}
export interface PerfPoint {
  window_start: string;
  window_end: string;
  n: number;
  accuracy: number;
  auc: number | null;
}
export interface Performance {
  overall: { n: number; accuracy: number; auc: number | null } | null;
  timeline: PerfPoint[];
}
export interface FlaggedFeature {
  report_id: number;
  feature: string;
  metric: string;
  value: number;
  threshold: number;
}
export interface IncidentContext {
  incident: Incident;
  drift: {
    onset: string;
    window_end: string;
    flagged_features: FlaggedFeature[];
    null_spikes: { report_id: number; feature: string; null_rate: number }[];
    prediction_drift_psi: number | null;
  };
  events: {
    nearest_deploy: { event_id: number; description: string; ts: string; hours_from_onset: number } | null;
    within_48h: { event_id: number; kind: string; description: string; ts: string; hours_from_onset: number }[];
  };
  performance: { pre: any; post: any; auc_delta: number | null; accuracy_delta: number | null; report_id: number | null };
  signals: any;
  signature: { label: string; confidence: string; rationale: string };
}
export interface AskAnswer {
  answer: string;
  citations: { type: string; ref: number }[];
  model_id: string | null;
  scope: string | null;
}

export const api = {
  models: () => get<Model[]>("/models"),
  incidents: () => get<Incident[]>("/incidents"),
  explanation: (id: number) => get<Explanation>(`/incidents/${id}/explanation`),
  ensureExplanation: (id: number) => post<Explanation>(`/incidents/${id}/explain`, {}),
  context: (id: number) => get<IncidentContext>(`/incidents/${id}/context`),
  recent: (m: string, limit = 14) =>
    get<{ id: number; ts: string; prediction: number; model_version: string }[]>(
      `/models/${m}/recent?limit=${limit}`
    ),
  performance: (m: string) => get<Performance>(`/models/${m}/performance?buckets=12`),
  events: (m: string) => get<{ id: number; ts: string; kind: string; description: string }[]>(`/models/${m}/events`),
  ask: (question: string, model_id?: string) => post<AskAnswer>("/ask", { question, model_id }),
};
