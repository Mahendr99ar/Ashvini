export interface CatalogueObject {
  n: number;
  name: string;
  t: string;
  l1: string;
  l2: string;
}

export interface ObjectBrief {
  norad_id: number;
  name: string;
  type: string;
  intl?: string;
  perigee_km?: number;
  apogee_km?: number;
  inclination_deg?: number;
  epoch?: string;
}

export type Level = "manoeuvre" | "monitor" | "dismiss";

export interface Decision {
  decision: Level;
  note: string;
  reviewer: string;
  at: string;
  recommendation_at_time?: Level;
}

export interface Conjunction {
  id: string;
  status: "open" | "passed";
  tca: string;
  primary: ObjectBrief;
  secondary: ObjectBrief;
  miss_km: number;
  rel_speed_kms: number;
  pc: number;
  log_pc: number;
  predicted: { log_risk: number; low: number; high: number; source: string };
  recommendation: Level;
  reasons: string[];
  thresholds: { action_pc: number; monitor_pc: number };
  data_age_days: number;
  origin: string;
  decision: Decision | null;
  history?: { seen_at: string; tca: string; miss_km: number; log_pc: number; predicted_log_risk: number; origin: string }[];
  reviews?: Decision[];
  miss_ric_km?: number[];
}

export interface KpPoint {
  time: string;
  kp: number;
  kind: string;
}

export interface SpaceWeather {
  kp_now: number | null;
  level_now: string | null;
  peak: { kp: number; time: string; level: string } | null;
  series: KpPoint[];
  source: string;
}

export interface Manoeuvre {
  norad_id: number;
  name: string | null;
  before: string;
  after: string;
  delta_a_km: number;
  score: number;
}

export interface Health {
  status: string;
  source: string;
  objects: number;
  open_events: number;
  data_age_hours: { newest_elements: number | null; newest_kp: number | null; last_run: number | null };
  risk_model: string | null;
}

export interface ModelCard {
  name: string;
  task: string;
  baseline: string;
  beats_baseline: boolean;
  data: string;
  limitations: string[];
}
