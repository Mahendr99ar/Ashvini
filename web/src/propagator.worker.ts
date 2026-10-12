/// <reference lib="webworker" />
/**
 * SGP4 for every object, off the main thread. The page asks for states at a time;
 * the worker answers with positions and velocities (TEME, km and km/s) in two flat
 * arrays, and the page extrapolates between answers so motion stays smooth.
 */
import { propagate, twoline2satrec, type SatRec } from "satellite.js";

let sats: SatRec[] = [];

type Message = { type: "init"; tles: [string, string][] } | { type: "propagate"; time: number; id: number };

self.onmessage = (event: MessageEvent<Message>) => {
  const msg = event.data;
  if (msg.type === "init") {
    sats = msg.tles.map(([l1, l2]) => twoline2satrec(l1, l2));
    (self as DedicatedWorkerGlobalScope).postMessage({ type: "ready", count: sats.length });
    return;
  }
  const when = new Date(msg.time);
  const pos = new Float32Array(sats.length * 3);
  const vel = new Float32Array(sats.length * 3);
  for (let i = 0; i < sats.length; i++) {
    const state = propagate(sats[i], when);
    const p = state.position;
    const v = state.velocity;
    if (p && typeof p !== "boolean" && v && typeof v !== "boolean") {
      pos[i * 3] = p.x; pos[i * 3 + 1] = p.y; pos[i * 3 + 2] = p.z;
      vel[i * 3] = v.x; vel[i * 3 + 1] = v.y; vel[i * 3 + 2] = v.z;
    } else {
      pos[i * 3] = pos[i * 3 + 1] = pos[i * 3 + 2] = NaN;
    }
  }
  (self as DedicatedWorkerGlobalScope).postMessage({ type: "states", id: msg.id, time: msg.time, pos, vel }, [pos.buffer, vel.buffer]);
};
