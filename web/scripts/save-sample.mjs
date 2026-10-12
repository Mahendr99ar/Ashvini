// Save what the live API serves into public/sample.json, which the site shows while the free API wakes up.
// Usage: VITE_API_URL=https://ashvini-api.onrender.com node scripts/save-sample.mjs
import { writeFileSync } from "node:fs";

const api = (process.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const get = async (path) => {
  const res = await fetch(api + path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
};

const [catalogue, conjunctions, weather] = await Promise.all([get("/api/catalogue"), get("/api/conjunctions"), get("/api/spaceweather")]);
writeFileSync(new URL("../public/sample.json", import.meta.url),
  JSON.stringify({ savedAt: new Date().toISOString(), catalogue: catalogue.objects, conjunctions, weather }));
console.log(`saved ${catalogue.objects.length} objects and ${conjunctions.length} events`);
