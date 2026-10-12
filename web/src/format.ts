/** Small pure helpers. Everything here is unit-tested. */

export const EARTH_RADIUS_KM = 6378.135;
export const MU = 398600.8;

export type Category = "broad" | "eo" | "nav" | "geo" | "crew" | "sci" | "deb" | "rb";

export const CATEGORIES: { id: Category; label: string; color: string }[] = [
  { id: "broad", label: "Broadband constellations", color: "c-broad" },
  { id: "eo", label: "Earth observation", color: "c-eo" },
  { id: "nav", label: "Navigation", color: "c-nav" },
  { id: "geo", label: "Geostationary", color: "c-geo" },
  { id: "crew", label: "Crewed stations", color: "c-crew" },
  { id: "sci", label: "Other payloads", color: "c-sci" },
  { id: "deb", label: "Debris", color: "c-deb" },
  { id: "rb", label: "Rocket bodies", color: "c-rb" },
];

const BROADBAND = /^(STARLINK|ONEWEB|LUMEN|ARC|KUIPER|QIANFAN|GUOWANG|IRIDIUM|GLOBALSTAR|ORBCOMM)/;
const NAVIGATION = /(NAVSTAR|GPS|GLONASS|COSMOS 2[45]\d\d|GALILEO|GSAT0|BEIDOU|IRNSS|NVS|QZS|NAV-)/;
const CREWED = /^(ISS|CSS|TIANHE|TIANGONG)/;
const IMAGING = /(SENTINEL|LANDSAT|CARTOSAT|RESOURCESAT|RISAT|EOS-|WORLDVIEW|PLEIADES|SPOT |FLOCK|LEMUR|DOVE|SKYSAT|TERRA|AQUA|NOAA|METOP|FENGYUN|GAOFEN|ICEYE|CAPELLA|OCEANSAT|AURORA|TERRA-OBS)/;

/** Mean motion, rev/day, from line 2 of a TLE. */
export function meanMotion(line2: string): number {
  return parseFloat(line2.slice(52, 63));
}

export function inclination(line2: string): number {
  return parseFloat(line2.slice(8, 16));
}

export function semiMajorAxisKm(revPerDay: number): number {
  const n = (revPerDay * 2 * Math.PI) / 86400;
  return Math.cbrt(MU / (n * n));
}

export function categorize(name: string, type: string, line2: string): Category {
  const upper = name.toUpperCase();
  if (type === "DEBRIS") return "deb";
  if (type === "ROCKET BODY") return "rb";
  if (CREWED.test(upper)) return "crew";
  const n = meanMotion(line2);
  if (Math.abs(n - 1.0027) < 0.03) return "geo";
  if (NAVIGATION.test(upper) || (n > 1.7 && n < 2.6)) return "nav";
  if (BROADBAND.test(upper)) return "broad";
  const inc = inclination(line2);
  if (IMAGING.test(upper) || (inc > 96 && inc < 99.5 && n > 13.5)) return "eo";
  return "sci";
}

export function missText(km: number): string {
  return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(2)} km`;
}

const SUPERSCRIPT: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

export function pcText(p: number): string {
  if (!(p > 0)) return "below 10⁻³⁰";
  const e = Math.floor(Math.log10(p));
  const m = p / 10 ** e;
  return `${m.toFixed(1)}×10${String(e).replace(/[-0-9]/g, (c) => SUPERSCRIPT[c])}`;
}

export function countdown(ms: number): string {
  const past = ms < 0;
  let s = Math.abs(Math.round(ms / 1000));
  const d = Math.floor(s / 86400);
  s %= 86400;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const text = d > 0 ? `${d}d ${String(h).padStart(2, "0")}h` : `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  return past ? `${text} ago` : `in ${text}`;
}

export function utcStamp(iso: string): string {
  return `${iso.slice(0, 16).replace("T", " ")} UTC`;
}

/** NOAA G-scale for a Kp value. */
export function stormLevel(kp: number): string {
  if (kp >= 5) return `G${Math.min(5, Math.floor(kp) - 4)}`;
  return kp >= 4 ? "active" : "quiet";
}

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);
}
