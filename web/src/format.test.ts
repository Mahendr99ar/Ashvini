import { describe, expect, it } from "vitest";
import { categorize, countdown, escapeHtml, meanMotion, missText, pcText, semiMajorAxisKm, stormLevel } from "./format";

const ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
const GEO_L2 = "2 41866   0.0400 271.4500 0001000 100.0000 200.0000  1.00271000 10000";
const SSO_L2 = "2 40697  98.5600  10.0000 0001000  90.0000 270.0000 14.30000000 50000";

describe("format", () => {
  it("reads mean motion from line 2", () => {
    expect(meanMotion(ISS_L2)).toBeCloseTo(15.72125391, 6);
  });

  it("puts geostationary semi-major axis near 42,164 km", () => {
    expect(semiMajorAxisKm(1.00273791)).toBeGreaterThan(42160);
    expect(semiMajorAxisKm(1.00273791)).toBeLessThan(42168);
  });

  it("sorts objects into the catalogue groups", () => {
    expect(categorize("ISS (ZARYA)", "PAYLOAD", ISS_L2)).toBe("crew");
    expect(categorize("SOME COMSAT", "PAYLOAD", GEO_L2)).toBe("geo");
    expect(categorize("STARLINK-1007", "PAYLOAD", ISS_L2)).toBe("broad");
    expect(categorize("UNNAMED IMAGER", "PAYLOAD", SSO_L2)).toBe("eo");
    expect(categorize("FENGYUN 1C DEB", "DEBRIS", SSO_L2)).toBe("deb");
  });

  it("writes miss distances and probabilities the way operators read them", () => {
    expect(missText(0.212)).toBe("212 m");
    expect(missText(1.4)).toBe("1.40 km");
    expect(pcText(3.1e-4)).toBe("3.1×10⁻⁴");
    expect(pcText(0)).toBe("below 10⁻³⁰");
  });

  it("counts down and up", () => {
    expect(countdown(3 * 3600e3 + 5 * 60e3)).toBe("in 03:05");
    expect(countdown(-26 * 3600e3)).toBe("1d 02h ago");
  });

  it("maps Kp to the G-scale", () => {
    expect(stormLevel(7)).toBe("G3");
    expect(stormLevel(4.3)).toBe("active");
    expect(stormLevel(2)).toBe("quiet");
  });

  it("escapes text before it goes into HTML", () => {
    expect(escapeHtml('<b a="x">&')).toBe("&lt;b a=&quot;x&quot;&gt;&amp;");
  });
});
