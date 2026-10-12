import * as THREE from "three";
import { eciToGeodetic, gstime, propagate, type SatRec } from "satellite.js";
import "./styles.css";
import { api, hasApiKey, loadSample, setApiKey } from "./api";
import { CATEGORIES, type Category, EARTH_RADIUS_KM, MU, categorize, countdown, escapeHtml, missText, pcText, stormLevel, utcStamp } from "./format";
import { Scene, satrecFor, sunDirection } from "./scene";
import type { CatalogueObject, Conjunction, SpaceWeather } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const root = document.documentElement;

// ---------- state ----------

interface Tracked { obj: CatalogueObject; cat: Category; satrec: SatRec | null }

const state = {
  objects: [] as Tracked[],
  byNorad: new Map<number, number>(),
  conjunctions: [] as Conjunction[],
  weather: null as SpaceWeather | null,
  visible: Object.fromEntries(CATEGORIES.map((c) => [c.id, true])) as Record<Category, boolean>,
  selected: -1,
  focused: null as Conjunction | null,
  simTime: Date.now(),
  speed: 1,
  playing: true,
  mode: "earth" as "earth" | "solar",
  live: false,
};

// ---------- theme ----------

let colors: Record<string, string> = {};

function readColors(): void {
  const s = getComputedStyle(root);
  for (const k of ["space", "ocean", "land", "coast", "grid", "glow", "text", "muted", "accent", "crit", "warn", "ok", ...CATEGORIES.map((c) => c.color)]) {
    colors[k] = s.getPropertyValue(`--${k}`).trim();
  }
}

function isDark(): boolean {
  const t = root.getAttribute("data-theme");
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

function setTheme(t: string, save: boolean): void {
  if (t === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", t);
  document.querySelectorAll<HTMLButtonElement>("[data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === t)));
  if (save) {
    try { localStorage.setItem("ashvini-theme", t); } catch { /* storage may be blocked */ }
  }
  applyColors();
}

function categoryColors(): THREE.Color[] {
  return state.objects.map((o) => new THREE.Color(colors[CATEGORIES.find((c) => c.id === o.cat)!.color]));
}

function applyColors(): void {
  readColors();
  scene.applyColors(colors, isDark());
  if (state.objects.length) scene.recolor(categoryColors());
  renderWeather();
}

// ---------- scene, camera, worker ----------

const canvas = $<HTMLCanvasElement>("scene");
const scene = new Scene(canvas);
const cam = { az: 0.9, el: 0.3, dist: 60 };
const HOME = { earth: { az: 0.9, el: 0.28, dist: 10.5 }, solar: { az: 0.5, el: 0.6, dist: 66 } };
{
  const sun = sunDirection(new Date());
  HOME.earth.az = Math.atan2(sun.x, sun.z) + 0.55; // open on the daylit side, with the terminator in view
}
let tween: { from: typeof cam; to: typeof cam; daz: number; t0: number; ms: number } | null = null;
let follow = -1;

function applyCamera(): void {
  scene.camera.position.set(Math.cos(cam.el) * Math.sin(cam.az), Math.sin(cam.el), Math.cos(cam.el) * Math.cos(cam.az)).multiplyScalar(cam.dist);
  scene.camera.lookAt(0, 0, 0);
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function flyTo(to: typeof cam, ms: number): void {
  const daz = ((to.az - cam.az + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
  if (reduced || !ms) { cam.az += daz; cam.el = to.el; cam.dist = to.dist; tween = null; return; }
  tween = { from: { ...cam }, to, daz, t0: performance.now(), ms };
}

function stepTween(now: number): void {
  if (!tween) return;
  const k = Math.min(1, (now - tween.t0) / tween.ms), e = ease(k), f = tween.from;
  cam.az = f.az + tween.daz * e;
  cam.el = f.el + (tween.to.el - f.el) * e;
  cam.dist = Math.exp(Math.log(f.dist) + (Math.log(tween.to.dist) - Math.log(f.dist)) * e);
  if (k >= 1) tween = null;
}

function lookAt(p: THREE.Vector3, dist: number): typeof cam {
  return { az: Math.atan2(p.x, p.z), el: Math.asin(Math.max(-1, Math.min(1, p.y / p.length()))), dist };
}

const worker = new Worker(new URL("./propagator.worker.ts", import.meta.url), { type: "module" });
let workerBusy = false;
let requestId = 0;
let lastStates: { time: number; pos: Float32Array; vel: Float32Array } | null = null;

worker.onmessage = (e: MessageEvent) => {
  if (e.data.type === "states") {
    lastStates = { time: e.data.time, pos: e.data.pos, vel: e.data.vel };
    workerBusy = false;
  }
};

function requestStates(): void {
  if (workerBusy || !state.objects.length) return;
  const stale = !lastStates || Math.abs(state.simTime - lastStates.time) > Math.max(2000, state.speed * 400);
  if (!stale) return;
  workerBusy = true;
  worker.postMessage({ type: "propagate", time: state.simTime + state.speed * 150, id: ++requestId });
}

// ---------- data ----------

function setCatalogue(list: CatalogueObject[]): void {
  state.objects = list.map((obj) => ({ obj, cat: categorize(obj.name, obj.t, obj.l2), satrec: null }));
  state.byNorad = new Map(state.objects.map((o, i) => [o.obj.n, i]));
  scene.setObjects(categoryColors());
  updateVisibility();
  lastStates = null;
  worker.postMessage({ type: "init", tles: list.map((o) => [o.l1, o.l2]) });
  renderFilters();
}

function satrecOf(i: number): SatRec {
  const o = state.objects[i];
  o.satrec ??= satrecFor(o.obj.l1, o.obj.l2);
  return o.satrec;
}

function updateVisibility(): void {
  scene.setVisible(state.objects.map((o, i) => state.visible[o.cat] || i === state.selected || isFocusObject(i)));
}

function isFocusObject(i: number): boolean {
  const f = state.focused;
  if (!f) return false;
  const n = state.objects[i]?.obj.n;
  return n === f.primary.norad_id || n === f.secondary.norad_id;
}

function setSourcePill(text: string, cls: string): void {
  const pill = $("sourcePill");
  pill.textContent = text;
  pill.className = `pill ${cls}`;
}

async function loadLive(): Promise<boolean> {
  try {
    const health = await api.health();
    const [cat, conj, weather] = await Promise.all([api.catalogue(), api.conjunctions(), api.spaceweather()]);
    setCatalogue(cat.objects);
    state.conjunctions = conj;
    state.weather = weather;
    state.live = true;
    setSourcePill(health.source === "celestrak" ? "Live · CelesTrak" : "Live · synthetic catalogue", "live");
    renderEvents();
    renderWeather();
    listen();
    return true;
  } catch {
    return false;
  }
}

async function loadFallback(): Promise<void> {
  try {
    const sample = await loadSample();
    setCatalogue(sample.catalogue);
    state.conjunctions = sample.conjunctions;
    state.weather = sample.weather;
    setSourcePill(`Saved sample from ${sample.savedAt.slice(0, 10)} · API waking up`, "sample");
    renderEvents();
    renderWeather();
  } catch {
    setSourcePill("Can't reach the API", "sample");
  }
}

async function boot(): Promise<void> {
  const quick = await Promise.race([loadLive(), new Promise<boolean>((r) => setTimeout(() => r(false), 9000))]);
  if (quick || state.live) return;
  await loadFallback();
  const retry = setInterval(async () => {
    if (await loadLive()) clearInterval(retry);
  }, 30000);
}

function listen(): void {
  try {
    const es = new EventSource(api.streamUrl());
    const refresh = async () => { state.conjunctions = await api.conjunctions(); renderEvents(); };
    for (const kind of ["conjunction.created", "conjunction.changed", "conjunction.reviewed", "cdm.received", "run.finished"]) {
      es.addEventListener(kind, () => void refresh());
    }
  } catch { /* live updates are a bonus */ }
}

// ---------- filters ----------

function renderFilters(): void {
  const counts = new Map<Category, number>();
  for (const o of state.objects) counts.set(o.cat, (counts.get(o.cat) ?? 0) + 1);
  $("fTotal").textContent = state.objects.length.toLocaleString("en-US");
  $("fList").innerHTML = CATEGORIES.map((c) => `<button class="chipbtn" data-cat="${c.id}" aria-pressed="${state.visible[c.id]}">
    <span class="dot" style="background:var(--${c.color})"></span><span>${c.label}</span><span class="n">${(counts.get(c.id) ?? 0).toLocaleString("en-US")}</span></button>`).join("");
  $("fList").querySelectorAll<HTMLButtonElement>("[data-cat]").forEach((b) => b.addEventListener("click", () => {
    const id = b.dataset.cat as Category;
    state.visible[id] = !state.visible[id];
    b.setAttribute("aria-pressed", String(state.visible[id]));
    updateVisibility();
  }));
}

// ---------- events ----------

function cardHtml(c: Conjunction): string {
  const tca = Date.parse(c.tca);
  const decided = c.decision ? ` · decided: ${c.decision.decision}` : "";
  return `<span class="row"><span class="tag t-${c.recommendation}">${c.recommendation}</span>
    <span class="ttl">${escapeHtml(c.primary.name)} × ${escapeHtml(c.secondary.name)}</span>
    <span class="when">TCA ${countdown(tca - Date.now())}</span></span>
    <p>${escapeHtml(c.reasons[0] ?? "")}${escapeHtml(decided)}</p>
    <dl class="facts"><div><dt>Miss</dt><dd>${missText(c.miss_km)}</dd></div><div><dt>Pc</dt><dd>${pcText(c.pc)}</dd></div>
    <div><dt>Rel. speed</dt><dd>${c.rel_speed_kms.toFixed(1)} km/s</dd></div></dl>`;
}

const seenCards = new Set<string>();

function renderEvents(): void {
  const list = $("evList");
  const open = state.conjunctions.filter((c) => c.status === "open");
  $("evCount").textContent = `${open.length} open`;
  if (!open.length) {
    list.innerHTML = `<li class="empty">No close approaches inside the screening distance right now. New element sets arrive every few hours.</li>`;
    return;
  }
  list.innerHTML = "";
  for (const c of open) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "card" + (seenCards.has(c.id) || reduced ? "" : " enter") + (state.focused?.id === c.id ? " sel" : "");
    btn.innerHTML = cardHtml(c);
    btn.addEventListener("click", () => void focusEvent(c));
    li.appendChild(btn);
    list.appendChild(li);
    if (!seenCards.has(c.id)) requestAnimationFrame(() => requestAnimationFrame(() => btn.classList.remove("enter")));
    seenCards.add(c.id);
  }
}

async function focusEvent(c: Conjunction): Promise<void> {
  if (state.mode !== "earth") setMode("earth");
  state.focused = c;
  updateVisibility();
  const date = new Date(state.simTime);
  const a = state.byNorad.get(c.primary.norad_id), b = state.byNorad.get(c.secondary.norad_id);
  scene.setOrbit("a", a !== undefined ? satrecOf(a) : null, date);
  scene.setOrbit("b", b !== undefined ? satrecOf(b) : null, date);
  if (a !== undefined) {
    select(a);
    follow = a;
    flyTo(lookAt(scene.pointOf(a).lengthSq() ? scene.pointOf(a) : new THREE.Vector3(0, 0, 1), 2.4), 1200);
  }
  renderEvents();
  openDetail(c);
  if (state.live) {
    try { openDetail(await api.conjunction(c.id)); } catch { /* the summary is enough */ }
  }
}

function historyChart(c: Conjunction): string {
  const h = c.history ?? [];
  if (h.length < 2) return `<p class="note">History appears here as new element sets update this event.</p>`;
  const w = 600, ht = 120, xs = h.map((_, i) => 20 + (i / (h.length - 1)) * (w - 40));
  const lo = Math.min(...h.map((p) => p.log_pc), -8), hi = Math.max(...h.map((p) => p.log_pc), -3);
  const y = (v: number) => 10 + (1 - (v - lo) / (hi - lo)) * (ht - 30);
  const actionY = y(Math.log10(c.thresholds.action_pc));
  const path = h.map((p, i) => `${i ? "L" : "M"}${xs[i].toFixed(1)},${y(p.log_pc).toFixed(1)}`).join(" ");
  const dots = h.map((p, i) => `<circle cx="${xs[i]}" cy="${y(p.log_pc)}" r="3.5" fill="${p.origin === "cdm" ? colors.warn : colors.accent}"/>`).join("");
  return `<svg class="chart" viewBox="0 0 ${w} ${ht}" role="img" aria-label="Pc over successive updates">
    <line x1="10" x2="${w - 10}" y1="${actionY}" y2="${actionY}" stroke="${colors.crit}" stroke-dasharray="4 4"/>
    <text x="${w - 10}" y="${actionY - 4}" text-anchor="end" font-size="10" fill="${colors.muted}" font-family="JetBrains Mono, monospace">action line</text>
    <path d="${path}" fill="none" stroke="${colors.accent}" stroke-width="2"/>${dots}
    <text x="10" y="${ht - 4}" font-size="10" fill="${colors.muted}" font-family="JetBrains Mono, monospace">first seen</text>
    <text x="${w - 10}" y="${ht - 4}" text-anchor="end" font-size="10" fill="${colors.muted}" font-family="JetBrains Mono, monospace">latest</text></svg>`;
}

function openDetail(c: Conjunction): void {
  $("dTitle").textContent = `${c.primary.name} × ${c.secondary.name}`;
  const p = c.predicted;
  const range = p.source.startsWith("model")
    ? `<li>Predicted final risk 10<sup>${p.log_risk.toFixed(1)}</sup>, 90% range 10<sup>${p.low.toFixed(1)}</sup> to 10<sup>${p.high.toFixed(1)}</sup> (${escapeHtml(p.source)}).</li>`
    : "";
  const reviews = (c.reviews ?? []).map((r) => `<li><b>${r.decision}</b> by ${escapeHtml(r.reviewer)}, ${utcStamp(r.at)}${r.note ? `: ${escapeHtml(r.note)}` : ""}</li>`).join("");
  $("dBody").innerHTML = `
    <div class="row"><span class="tag t-${c.recommendation}">${c.recommendation}</span><span class="sub">TCA ${utcStamp(c.tca)} (${countdown(Date.parse(c.tca) - Date.now())})</span></div>
    <dl class="facts">
      <div><dt>Miss distance</dt><dd>${missText(c.miss_km)}</dd></div><div><dt>Pc</dt><dd>${pcText(c.pc)}</dd></div>
      <div><dt>Relative speed</dt><dd>${c.rel_speed_kms.toFixed(2)} km/s</dd></div>
      <div><dt>Primary</dt><dd>${escapeHtml(c.primary.name)} (${c.primary.norad_id})</dd></div>
      <div><dt>Secondary</dt><dd>${escapeHtml(c.secondary.name)} (${c.secondary.norad_id})</dd></div>
      <div><dt>Element age</dt><dd>${c.data_age_days.toFixed(1)} days</dd></div>
    </dl>
    <h3>Why this recommendation</h3>
    <ul>${c.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}${range}</ul>
    <h3>Pc over updates</h3>${historyChart(c)}
    ${reviews ? `<h3>Decisions</h3><ul>${reviews}</ul>` : ""}
    <div class="row"><a class="btn" href="${api.cdmUrl(c.id)}" target="_blank" rel="noopener">Download CDM</a>
    <span class="note">Screened on public data. Not an official conjunction assessment.</span></div>
    <form class="decide" id="decideForm">
      <span class="k">Record a decision</span>
      <div class="seg" role="radiogroup" aria-label="Decision">
        ${["manoeuvre", "monitor", "dismiss"].map((d) => `<button type="button" class="btn" data-decision="${d}" aria-pressed="${d === c.recommendation}">${d}</button>`).join("")}
      </div>
      <textarea id="decideNote" placeholder="Why. Required for a manoeuvre and whenever you differ from the recommendation."></textarea>
      ${hasApiKey() ? "" : `<input id="decideKey" type="password" placeholder="Analyst API key" autocomplete="off" />`}
      <div class="row"><button class="btn primary" type="submit">Save decision</button><span class="err" id="decideErr"></span></div>
    </form>`;
  $("detail").hidden = false;
  let choice = c.recommendation as string;
  $("dBody").querySelectorAll<HTMLButtonElement>("[data-decision]").forEach((b) => b.addEventListener("click", () => {
    choice = b.dataset.decision!;
    $("dBody").querySelectorAll("[data-decision]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  $("decideForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const keyInput = document.getElementById("decideKey") as HTMLInputElement | null;
    if (keyInput?.value) setApiKey(keyInput.value);
    try {
      const updated = await api.review(c.id, choice, ($("decideNote") as HTMLTextAreaElement).value);
      state.conjunctions = state.conjunctions.map((x) => (x.id === updated.id ? updated : x));
      renderEvents();
      openDetail(updated);
    } catch (err) {
      $("decideErr").textContent = (err as Error).message;
    }
  });
}

// ---------- space weather ----------

function renderWeather(): void {
  const w = state.weather;
  const box = $("weather");
  if (!w || !w.series.length) { box.innerHTML = ""; $("sKp").textContent = "-"; return; }
  const now = Date.now();
  const series = w.series.slice(-16);
  const width = 300, h = 64, bw = width / series.length;
  const bars = series.map((p, i) => {
    const bh = (p.kp / 9) * h, col = p.kp >= 7 ? colors.crit : p.kp >= 5 ? colors.warn : colors.accent;
    const past = Date.parse(p.time) <= now;
    return `<rect x="${i * bw + 1.5}" y="${h - bh}" width="${bw - 3}" height="${bh}" rx="1.5" fill="${col}" opacity="${past ? 0.45 : 0.95}"/>`;
  }).join("");
  const storm = h - (5 / 9) * h;
  const peak = w.peak ? `Forecast peak Kp ${w.peak.kp.toFixed(1)} (${w.peak.level}) at ${utcStamp(w.peak.time)}.` : "No forecast loaded.";
  box.innerHTML = `<div class="row"><span class="k">Geomagnetic activity</span><span class="tag ${w.peak && w.peak.kp >= 5 ? "t-monitor" : "t-acc"}">${w.peak && w.peak.kp >= 5 ? "Storm watch" : "Quiet"}</span></div>
    <svg viewBox="0 0 ${width} ${h + 4}" role="img" aria-label="Kp by three-hour block, past and forecast">
      <line x1="0" x2="${width}" y1="${storm}" y2="${storm}" stroke="${colors.warn}" stroke-dasharray="3 3"/>${bars}</svg>
    <small>${peak} Storms raise drag on low orbits, so Pc values move after the next element sets.</small>`;
  $("sKp").textContent = w.kp_now === null ? "-" : w.kp_now.toFixed(1);
  $("sKp").className = w.kp_now !== null && w.kp_now >= 5 ? "warnc" : "";
  if (w.kp_now !== null) $("sKp").title = stormLevel(w.kp_now);
}

// ---------- selection ----------

function select(i: number): void {
  state.selected = i;
  $("insp").hidden = i < 0;
  if (i < 0) { scene.setOrbit("sel", null, new Date()); scene.showSelection(null); updateVisibility(); return; }
  scene.setOrbit("sel", satrecOf(i), new Date(state.simTime));
  updateVisibility();
  renderInspect();
}

function renderInspect(): void {
  const i = state.selected;
  if (i < 0) return;
  const o = state.objects[i], cat = CATEGORIES.find((c) => c.id === o.cat)!;
  const sr = satrecOf(i), date = new Date(state.simTime);
  const st = propagate(sr, date);
  $("iName").textContent = o.obj.name;
  $("iTag").textContent = cat.label;
  $("iTag").style.color = `var(--${cat.color})`;
  $("iSub").textContent = o.obj.t === "DEBRIS" ? "Tracked fragment" : o.obj.t === "ROCKET BODY" ? "Spent upper stage" : "Payload";
  $("iNorad").textContent = String(o.obj.n);
  $("iInc").textContent = `${((sr.inclo * 180) / Math.PI).toFixed(1)}°`;
  $("iPer").textContent = `${((2 * Math.PI) / sr.no).toFixed(1)} min`;
  const p = st.position;
  if (p && typeof p !== "boolean") {
    const r = Math.hypot(p.x, p.y, p.z);
    const geo = eciToGeodetic(p, gstime(date));
    const lat = (geo.latitude * 180) / Math.PI, lon = (geo.longitude * 180) / Math.PI;
    $("iAlt").textContent = `${Math.round(r - EARTH_RADIUS_KM).toLocaleString("en-US")} km`;
    $("iSpd").textContent = `${Math.sqrt(MU / r).toFixed(2)} km/s`;
    $("iOver").textContent = `${Math.abs(lat).toFixed(0)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(0)}°${lon >= 0 ? "E" : "W"}`;
  }
}

$("iClose").addEventListener("click", () => select(-1));

function pick(x: number, y: number): void {
  if (state.mode !== "earth") return;
  let best = -1, bd = 12;
  const v = new THREE.Vector3();
  for (let i = 0; i < scene.pointCount(); i++) {
    if (!scene.isShown(i)) continue;
    const p = scene.pointOf(i);
    v.copy(p).project(scene.camera);
    if (v.z > 1) continue;
    const sx = ((v.x + 1) / 2) * innerWidth, sy = ((1 - v.y) / 2) * innerHeight;
    const d = Math.hypot(sx - x, sy - y);
    if (d < bd && !scene.occluded(p)) { bd = d; best = i; }
  }
  follow = -1;
  select(best);
}

// ---------- input ----------

let drag: { x: number; y: number; moved: number } | null = null;
const pointers = new Map<number, PointerEvent>();
let pinch = 0;

function zoomBy(f: number): void {
  tween = null;
  const [lo, hi] = state.mode === "earth" ? [1.35, 40] : [8, 160];
  cam.dist = Math.max(lo, Math.min(hi, cam.dist * f));
}

canvas.addEventListener("pointerdown", (e) => {
  pointers.set(e.pointerId, e);
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, moved: 0 };
  if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); }
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (pointers.has(e.pointerId)) pointers.set(e.pointerId, e);
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (pinch) zoomBy(pinch / d);
    pinch = d;
    drag.moved += 10;
    return;
  }
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
  if (drag.moved > 4) {
    canvas.classList.add("drag");
    tween = null;
    follow = -1;
    const k = 2.6 / Math.max(innerHeight, 400);
    cam.az -= dx * k;
    cam.el = Math.max(-1.45, Math.min(1.45, cam.el + dy * k));
  }
});
function endPointer(e: PointerEvent): void {
  pointers.delete(e.pointerId);
  canvas.classList.remove("drag");
  if (drag && drag.moved <= 4 && pointers.size === 0) pick(e.clientX, e.clientY);
  if (pointers.size === 0) drag = null;
  pinch = 0;
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomBy(Math.exp(e.deltaY * 0.0012)); }, { passive: false });

// ---------- controls ----------

document.querySelectorAll<HTMLButtonElement>("[data-theme-set]").forEach((b) => b.addEventListener("click", () => setTheme(b.dataset.themeSet!, true)));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!root.getAttribute("data-theme")) applyColors(); });

$("play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶";
  $("play").setAttribute("aria-label", state.playing ? "Pause" : "Play");
});
document.querySelectorAll<HTMLButtonElement>("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  state.speed = Number(b.dataset.speed);
  if (state.speed === 1) state.simTime = Date.now();
  document.querySelectorAll("[data-speed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));

function setMode(m: "earth" | "solar"): void {
  state.mode = m;
  scene.mode = m;
  document.querySelectorAll("[data-mode]").forEach((x) => x.setAttribute("aria-pressed", String((x as HTMLElement).dataset.mode === m)));
  if (m === "solar") { select(-1); follow = -1; }
  cam.dist = HOME[m].dist * 1.6;
  flyTo(HOME[m], 1200);
}
document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode as "earth" | "solar")));
$("home").addEventListener("click", () => { follow = -1; flyTo(HOME[state.mode], 1000); });

function openSheet(id: string): void { $(id).hidden = false; }
function closeSheet(id: string): void { $(id).hidden = true; }
$("refBtn").addEventListener("click", () => openSheet("refs"));
$("modelsBtn").addEventListener("click", async () => {
  openSheet("models");
  try {
    const m = await api.models();
    $("mBody").innerHTML = `<p class="lede">Serving: <b>${escapeHtml(m.serving)}</b>. Every model is compared with a plain baseline on held-out data, and the card says whether it won.</p>` +
      (m.cards.length ? m.cards.map((c) => `<div class="modelcard"><div class="row"><b>${escapeHtml(c.name)}</b>
        <span class="tag ${c.beats_baseline ? "t-dismiss" : "t-monitor"}">${c.beats_baseline ? "beats baseline" : "does not beat baseline"}</span></div>
        <span class="sub">${escapeHtml(c.task)}</span><span class="note">Data: ${escapeHtml(c.data)}. Baseline: ${escapeHtml(c.baseline)}.</span></div>`).join("")
        : `<p class="sub">No model cards published to this API yet. Train them with ml/notebooks/kaggle_train_all.ipynb.</p>`);
  } catch {
    $("mBody").innerHTML = `<p class="sub">The API is not reachable right now, so the model cards cannot be shown.</p>`;
  }
});
document.querySelectorAll<HTMLButtonElement>("[data-close]").forEach((b) => b.addEventListener("click", () => closeSheet(b.dataset.close!)));
$("dClose").addEventListener("click", () => closeSheet("detail"));
for (const id of ["refs", "models", "detail"]) $(id).addEventListener("click", (e) => { if (e.target === $(id)) closeSheet(id); });
window.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key !== "Escape") return;
  const open = ["detail", "models", "refs"].find((id) => !$(id).hidden);
  if (open) closeSheet(open); else select(-1);
});

$("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = ($("askInput") as HTMLInputElement).value.trim();
  if (q.length < 2) return;
  const out = $("askAnswer");
  out.hidden = false;
  out.textContent = "Looking it up…";
  try {
    const a = await api.ask(q);
    out.textContent = a.answer;
  } catch (err) {
    out.textContent = `The copilot is unavailable: ${(err as Error).message}`;
  }
});

// ---------- labels ----------

const labelsEl = $("labels");
const pool: HTMLDivElement[] = [];

function label(i: number, text: string, x: number, y: number, cls = ""): HTMLDivElement {
  let el = pool[i];
  if (!el) { el = document.createElement("div"); labelsEl.appendChild(el); pool[i] = el; }
  el.className = `lbl ${cls}`;
  el.textContent = text;
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
  el.hidden = false;
  el.onclick = null;
  return el;
}

function screenOf(p: THREE.Vector3): [number, number, number] {
  const v = p.clone().project(scene.camera);
  return [((v.x + 1) / 2) * innerWidth, ((1 - v.y) / 2) * innerHeight, v.z];
}

// ---------- loop ----------

function resize(): void { scene.resize(innerWidth, innerHeight); }
addEventListener("resize", resize);

let last = performance.now(), uiTimer = 0;
const J2000 = Date.UTC(2000, 0, 1, 12);
let solarOffsetDays = 0;

function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (state.playing) state.simTime += dt * 1000 * state.speed;
  stepTween(now);
  const date = new Date(state.simTime);
  let li = 0;

  if (state.mode === "earth") {
    scene.setTime(date);
    requestStates();
    if (lastStates) scene.updatePositions(lastStates.pos, lastStates.vel, (state.simTime - lastStates.time) / 1000);
    if (follow >= 0 && !tween) {
      const t = lookAt(scene.pointOf(follow), cam.dist);
      cam.az += (((t.az - cam.az + Math.PI * 3) % (Math.PI * 2)) - Math.PI) * 0.08;
      cam.el += (t.el - cam.el) * 0.08;
    }
    applyCamera();
    scene.showSelection(state.selected >= 0 ? scene.pointOf(state.selected) : null);
    const f = state.focused;
    const a = f ? state.byNorad.get(f.primary.norad_id) : undefined, b = f ? state.byNorad.get(f.secondary.norad_id) : undefined;
    if (f && a !== undefined && b !== undefined && f.status === "open") {
      scene.showLink(scene.pointOf(a), scene.pointOf(b), reduced ? 0.5 : ((now / 1000) * 0.8) % 1);
    } else scene.hideConjunction();
    scene.render();
    const tagged = new Set<number>();
    if (state.selected >= 0) tagged.add(state.selected);
    if (a !== undefined) tagged.add(a);
    if (b !== undefined) tagged.add(b);
    for (const i of tagged) {
      const p = scene.pointOf(i);
      const [x, y, z] = screenOf(p);
      if (z < 1 && !scene.occluded(p) && p.lengthSq() > 0) label(li++, state.objects[i].obj.name, x, y);
    }
  } else {
    if (state.playing) solarOffsetDays += dt * Math.max(2, state.speed / 30);  // planets need days per second to visibly move
    const planets = scene.placePlanets((state.simTime - J2000) / 864e5 + solarOffsetDays);
    applyCamera();
    scene.render();
    for (const p of planets) {
      const [x, y, z] = screenOf(p.pos);
      if (z < 1) {
        const el = label(li++, p.name, x, y, `planet${p.name === "Earth" ? " earth" : ""}`);
        if (p.name === "Earth") el.onclick = () => setMode("earth");
      }
    }
  }
  for (let i = li; i < pool.length; i++) pool[i].hidden = true;

  if (now - uiTimer > 250) {
    uiTimer = now;
    const iso = date.toISOString();
    $("clock").innerHTML = `<b>${iso.slice(11, 19)}</b> UTC · ${iso.slice(0, 10)}`;
    $("sObj").textContent = state.objects.filter((o) => state.visible[o.cat]).length.toLocaleString("en-US");
    const open = state.conjunctions.filter((c) => c.status === "open");
    $("sCa").textContent = String(open.length);
    $("sCa").className = open.some((c) => c.recommendation === "manoeuvre") ? "hot" : "";
    const next = open.map((c) => Date.parse(c.tca)).filter((t) => t > Date.now()).sort((x, y) => x - y)[0];
    $("sTca").textContent = next ? countdown(next - Date.now()).replace("in ", "") : "--";
    renderInspect();
  }
  requestAnimationFrame(frame);
}

// ---------- start ----------

const savedTheme = (() => { try { return localStorage.getItem("ashvini-theme") ?? "auto"; } catch { return "auto"; } })();
setTheme(savedTheme, false);
resize();
applyCamera();
flyTo(HOME.earth, reduced ? 0 : 3200);
requestAnimationFrame(frame);
void boot();
