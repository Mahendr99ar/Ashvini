/**
 * The 3D view: Earth with a real day/night line, every tracked object as a point,
 * orbits and the link line for a selected close approach, and a solar-system view.
 * Positions arrive in TEME (km); three.js axes are x = X, y = Z, z = -Y, in Earth radii.
 */
import * as THREE from "three";
import { gstime, propagate, twoline2satrec, type SatRec } from "satellite.js";
import land from "./data/land.json";
import { EARTH_RADIUS_KM, MU } from "./format";

export type Colors = Record<string, string>;

const D = Math.PI / 180;

export function toThree(x: number, y: number, z: number, out: THREE.Vector3): THREE.Vector3 {
  return out.set(x / EARTH_RADIUS_KM, z / EARTH_RADIUS_KM, -y / EARTH_RADIUS_KM);
}

export function sunDirection(date: Date): THREE.Vector3 {
  const n = (date.getTime() - Date.UTC(2000, 0, 1, 12)) / 864e5;
  const L = (280.46 + 0.9856474 * n) * D;
  const g = (357.528 + 0.9856003 * n) * D;
  const lam = L + (1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * D;
  const eps = 23.439 * D;
  return new THREE.Vector3(Math.cos(lam), Math.sin(eps) * Math.sin(lam), -Math.cos(eps) * Math.sin(lam)).normalize();
}

function dotTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d")!;
  g.beginPath();
  g.arc(16, 16, 13, 0, 7);
  g.fillStyle = "#fff";
  g.fill();
  return new THREE.CanvasTexture(c);
}

function ringTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  g.beginPath();
  g.arc(32, 32, 26, 0, 7);
  g.lineWidth = 4;
  g.strokeStyle = "#fff";
  g.stroke();
  return new THREE.CanvasTexture(c);
}

function orbitLine(): THREE.Line {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(361 * 3), 3));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ transparent: true, opacity: 0.9 }));
  line.visible = false;
  line.frustumCulled = false;
  return line;
}

const PLANETS: [string, number, number, number, number, string][] = [
  ["Mercury", 0.387, 252.25, 149472.67, 0.16, "#9C978F"], ["Venus", 0.723, 181.98, 58517.82, 0.26, "#D8B77A"],
  ["Earth", 1.0, 100.46, 35999.37, 0.27, "#4C8DFF"], ["Mars", 1.524, -4.55, 19140.3, 0.2, "#C8613F"],
  ["Jupiter", 5.203, 34.4, 3034.75, 0.62, "#C9A27A"], ["Saturn", 9.537, 49.95, 1222.49, 0.52, "#D9C08B"],
  ["Uranus", 19.19, 313.24, 428.48, 0.38, "#8FC7CF"], ["Neptune", 30.07, -55.12, 218.46, 0.37, "#5574D9"],
];
const planetRadius = (a: number) => 5 * Math.sqrt(a) + 1.2;

export class Scene {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera = new THREE.PerspectiveCamera(38, 1, 0.01, 2000);
  private earthScene = new THREE.Scene();
  private solarScene = new THREE.Scene();
  private earthCanvas = document.createElement("canvas");
  private earthTex: THREE.CanvasTexture;
  private earthMat: THREE.ShaderMaterial;
  private earth: THREE.Mesh;
  private atmoMat: THREE.ShaderMaterial;
  private stars: THREE.Points[] = [];
  private points!: THREE.Points;
  private positions = new Float32Array(0);
  private colors = new Float32Array(0);
  private visible: boolean[] = [];
  private selRing: THREE.Sprite;
  private ping: THREE.Sprite;
  private orbits = { sel: orbitLine(), a: orbitLine(), b: orbitLine() };
  private link: THREE.Line;
  private planets: { name: string; a: number; mesh: THREE.Mesh }[] = [];
  mode: "earth" | "solar" = "earth";
  dark = true;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
    this.earthCanvas.width = 2048;
    this.earthCanvas.height = 1024;
    this.earthTex = new THREE.CanvasTexture(this.earthCanvas);
    this.earthTex.anisotropy = 4;
    this.earthMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: this.earthTex }, sun: { value: new THREE.Vector3(1, 0, 0) }, night: { value: 0.4 } },
      vertexShader: `varying vec2 vUv; varying vec3 vN; void main(){ vUv = uv; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D map; uniform vec3 sun; uniform float night; varying vec2 vUv; varying vec3 vN;
        void main(){ vec3 c = texture2D(map, vUv).rgb; float k = smoothstep(-0.12, 0.22, dot(normalize(vN), sun));
        gl_FragColor = vec4(mix(c * night, c, k), 1.0); }`,
    });
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), this.earthMat);
    this.earthScene.add(this.earth);
    this.atmoMat = new THREE.ShaderMaterial({
      uniforms: { glow: { value: new THREE.Color(0x4c8dff) }, power: { value: 1 } },
      vertexShader: `varying vec3 vN; void main(){ vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform vec3 glow; uniform float power; varying vec3 vN; void main(){ float i = pow(0.72 - dot(vN, vec3(0.0, 0.0, 1.0)), 3.0) * power; gl_FragColor = vec4(glow, clamp(i, 0.0, 1.0)); }`,
      side: THREE.BackSide, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
    });
    this.earthScene.add(new THREE.Mesh(new THREE.SphereGeometry(1.09, 64, 48), this.atmoMat));
    this.addStars();
    const ring = ringTexture();
    this.selRing = new THREE.Sprite(new THREE.SpriteMaterial({ map: ring, depthTest: false, transparent: true }));
    this.ping = new THREE.Sprite(new THREE.SpriteMaterial({ map: ring, depthTest: false, transparent: true }));
    this.selRing.visible = this.ping.visible = false;
    const linkGeo = new THREE.BufferGeometry();
    linkGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    this.link = new THREE.Line(linkGeo, new THREE.LineDashedMaterial({ dashSize: 0.01, gapSize: 0.008, depthTest: false, transparent: true }));
    this.link.visible = false;
    this.link.frustumCulled = false;
    this.earthScene.add(this.selRing, this.ping, this.orbits.sel, this.orbits.a, this.orbits.b, this.link);
    this.buildSolar();
  }

  private addStars(): void {
    const pts: number[] = [];
    for (let i = 0; i < 2600; i++) {
      const u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      pts.push(400 * s * Math.cos(t), 400 * u, 400 * s * Math.sin(t));
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 1.1, sizeAttenuation: false, transparent: true, opacity: 0.55, depthWrite: false });
    for (const scene of [this.earthScene, this.solarScene]) {
      const s = new THREE.Points(geo, mat);
      scene.add(s);
      this.stars.push(s);
    }
  }

  private buildSolar(): void {
    this.solarScene.add(new THREE.Mesh(new THREE.SphereGeometry(0.9, 48, 32), new THREE.MeshBasicMaterial({ color: 0xffc75a })));
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d")!;
    const gr = g.createRadialGradient(64, 64, 8, 64, 64, 64);
    gr.addColorStop(0, "rgba(255,210,120,.9)");
    gr.addColorStop(1, "rgba(255,170,60,0)");
    g.fillStyle = gr;
    g.fillRect(0, 0, 128, 128);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    glow.scale.set(4, 4, 1);
    this.solarScene.add(glow, new THREE.AmbientLight(0xffffff, 0.35), new THREE.PointLight(0xffffff, 1.4, 0, 0));
    for (const [name, a, , , size, color] of PLANETS) {
      const mat = name === "Earth" ? new THREE.MeshLambertMaterial({ map: this.earthTex }) : new THREE.MeshLambertMaterial({ color });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 32, 24), mat);
      const pts: number[] = [];
      for (let k = 0; k <= 180; k++) {
        const t = (k / 180) * Math.PI * 2;
        pts.push(planetRadius(a) * Math.cos(t), 0, -planetRadius(a) * Math.sin(t));
      }
      const og = new THREE.BufferGeometry();
      og.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      this.solarScene.add(mesh, new THREE.Line(og, new THREE.LineBasicMaterial({ color: 0x8792a8, transparent: true, opacity: 0.35 })));
      if (name === "Saturn") {
        const ring = new THREE.Mesh(new THREE.RingGeometry(size * 1.4, size * 2.3, 64), new THREE.MeshBasicMaterial({ color: 0xd9c08b, side: THREE.DoubleSide, transparent: true, opacity: 0.55 }));
        ring.rotation.x = -Math.PI / 2 + 0.47;
        mesh.add(ring);
      }
      this.planets.push({ name, a, mesh });
    }
  }

  applyColors(c: Colors, dark: boolean): void {
    this.dark = dark;
    this.renderer.setClearColor(new THREE.Color(c.space), 1);
    this.paintEarth(c);
    this.earthMat.uniforms.night.value = dark ? 0.42 : 0.72;
    this.atmoMat.uniforms.glow.value.set(c.glow);
    this.atmoMat.uniforms.power.value = dark ? 1 : 0.55;
    this.atmoMat.blending = dark ? THREE.AdditiveBlending : THREE.NormalBlending;
    this.atmoMat.needsUpdate = true;
    for (const s of this.stars) s.visible = dark;
    (this.selRing.material as THREE.SpriteMaterial).color.set(c.accent);
    (this.ping.material as THREE.SpriteMaterial).color.set(c.crit);
    (this.orbits.sel.material as THREE.LineBasicMaterial).color.set(c.accent);
    (this.orbits.a.material as THREE.LineBasicMaterial).color.set(c.accent);
    (this.orbits.b.material as THREE.LineBasicMaterial).color.set(c.crit);
    (this.link.material as THREE.LineDashedMaterial).color.set(c.crit);
  }

  private paintEarth(c: Colors): void {
    const g = this.earthCanvas.getContext("2d")!;
    const W = 2048, H = 1024;
    g.fillStyle = c.ocean;
    g.fillRect(0, 0, W, H);
    g.strokeStyle = c.grid;
    g.lineWidth = 1;
    g.beginPath();
    for (let lon = -180; lon <= 180; lon += 15) { const x = ((lon + 180) / 360) * W; g.moveTo(x, 0); g.lineTo(x, H); }
    for (let lat = -75; lat <= 75; lat += 15) { const y = ((90 - lat) / 180) * H; g.moveTo(0, y); g.lineTo(W, y); }
    g.stroke();
    g.beginPath();
    for (const ring of land as number[][][]) {
      ring.forEach(([lo, la], i) => {
        const x = ((lo + 180) / 360) * W, y = ((90 - la) / 180) * H;
        if (i) g.lineTo(x, y); else g.moveTo(x, y);
      });
      g.closePath();
    }
    g.fillStyle = c.land;
    g.fill();
    g.strokeStyle = c.coast;
    g.lineWidth = 1.4;
    g.stroke();
    this.earthTex.needsUpdate = true;
  }

  /** Allocate one point per object, coloured by category. */
  setObjects(colors: THREE.Color[]): void {
    if (this.points) this.earthScene.remove(this.points);
    const n = colors.length;
    this.positions = new Float32Array(n * 3);
    this.colors = new Float32Array(n * 3);
    this.visible = new Array(n).fill(true);
    colors.forEach((col, i) => { this.colors[i * 3] = col.r; this.colors[i * 3 + 1] = col.g; this.colors[i * 3 + 2] = col.b; });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    const mat = new THREE.PointsMaterial({ size: 2.3 * Math.min(2, devicePixelRatio || 1), sizeAttenuation: false, vertexColors: true,
      map: dotTexture(), alphaTest: 0.4, transparent: true });
    this.points = new THREE.Points(geo, mat);
    this.points.frustumCulled = false;
    this.earthScene.add(this.points);
  }

  recolor(colors: THREE.Color[]): void {
    colors.forEach((col, i) => { this.colors[i * 3] = col.r; this.colors[i * 3 + 1] = col.g; this.colors[i * 3 + 2] = col.b; });
    this.points.geometry.attributes.color.needsUpdate = true;
  }

  setVisible(mask: boolean[]): void {
    this.visible = mask;
  }

  /** Second-order extrapolation from the last worker answer: r + v dt + a dt^2 / 2, with two-body gravity. */
  updatePositions(pos: Float32Array, vel: Float32Array, dtSeconds: number): void {
    const out = this.positions;
    for (let i = 0; i < this.visible.length; i++) {
      const k = i * 3;
      const x = pos[k], y = pos[k + 1], z = pos[k + 2];
      if (!this.visible[i] || Number.isNaN(x)) { out[k] = out[k + 1] = out[k + 2] = 0; continue; }
      const r = Math.hypot(x, y, z);
      const g = (-MU / (r * r * r)) * 0.5 * dtSeconds * dtSeconds;
      const X = x + vel[k] * dtSeconds + g * x, Y = y + vel[k + 1] * dtSeconds + g * y, Z = z + vel[k + 2] * dtSeconds + g * z;
      out[k] = X / EARTH_RADIUS_KM; out[k + 1] = Z / EARTH_RADIUS_KM; out[k + 2] = -Y / EARTH_RADIUS_KM;
    }
    this.points.geometry.attributes.position.needsUpdate = true;
  }

  pointOf(i: number): THREE.Vector3 {
    return new THREE.Vector3(this.positions[i * 3], this.positions[i * 3 + 1], this.positions[i * 3 + 2]);
  }

  pointCount(): number {
    return this.visible.length;
  }

  isShown(i: number): boolean {
    return this.visible[i] && this.positions[i * 3] !== 0;
  }

  setTime(date: Date): void {
    this.earth.rotation.y = gstime(date);
    this.earthMat.uniforms.sun.value.copy(sunDirection(date));
  }

  setOrbit(which: "sel" | "a" | "b", satrec: SatRec | null, date: Date): void {
    const line = this.orbits[which];
    if (!satrec) { line.visible = false; return; }
    const arr = line.geometry.attributes.position.array as Float32Array;
    const periodMin = (2 * Math.PI) / satrec.no;
    const v = new THREE.Vector3();
    for (let k = 0; k <= 360; k++) {
      const st = propagate(satrec, new Date(date.getTime() + (k / 360) * periodMin * 60000));
      const p = st.position;
      if (!p || typeof p === "boolean") continue;
      toThree(p.x, p.y, p.z, v);
      arr[k * 3] = v.x; arr[k * 3 + 1] = v.y; arr[k * 3 + 2] = v.z;
    }
    line.geometry.attributes.position.needsUpdate = true;
    line.visible = true;
  }

  showSelection(p: THREE.Vector3 | null): void {
    this.selRing.visible = !!p;
    if (p) {
      this.selRing.position.copy(p);
      const s = this.camera.position.distanceTo(p) * 0.035;
      this.selRing.scale.set(s, s, 1);
    }
  }

  showLink(a: THREE.Vector3 | null, b: THREE.Vector3 | null, phase: number): void {
    if (!a || !b) { this.link.visible = this.ping.visible = false; return; }
    const arr = this.link.geometry.attributes.position.array as Float32Array;
    arr.set([a.x, a.y, a.z, b.x, b.y, b.z]);
    this.link.geometry.attributes.position.needsUpdate = true;
    this.link.computeLineDistances();
    this.link.visible = true;
    this.ping.position.copy(a);
    const s = this.camera.position.distanceTo(a) * (0.03 + phase * 0.09);
    this.ping.scale.set(s, s, 1);
    (this.ping.material as THREE.SpriteMaterial).opacity = 1 - phase;
    this.ping.visible = true;
  }

  hideConjunction(): void {
    this.orbits.a.visible = this.orbits.b.visible = false;
    this.showLink(null, null, 0);
  }

  placePlanets(daysSinceJ2000: number): { name: string; pos: THREE.Vector3 }[] {
    const T = daysSinceJ2000 / 36525;
    return this.planets.map((p, i) => {
      const L = (PLANETS[i][2] + PLANETS[i][3] * T) * D, R = planetRadius(p.a);
      p.mesh.position.set(R * Math.cos(L), 0, -R * Math.sin(L));
      p.mesh.rotation.y += 0.004;
      return { name: p.name, pos: p.mesh.position };
    });
  }

  render(): void {
    this.renderer.render(this.mode === "earth" ? this.earthScene : this.solarScene, this.camera);
  }

  resize(w: number, h: number): void {
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.setViewOffset(w, h, w > 900 ? -40 : 0, w > 900 ? 0 : h * 0.12, w, h);
    this.camera.updateProjectionMatrix();
  }

  /** Is the straight line from the camera to p blocked by the Earth? */
  occluded(p: THREE.Vector3): boolean {
    const c = this.camera.position, d = p.clone().sub(c);
    const a = d.dot(d), b = 2 * c.dot(d), cc = c.dot(c) - 1, disc = b * b - 4 * a * cc;
    if (disc < 0) return false;
    const t = (-b - Math.sqrt(disc)) / (2 * a);
    return t > 0 && t < 1;
  }
}

export function satrecFor(l1: string, l2: string): SatRec {
  return twoline2satrec(l1, l2);
}
