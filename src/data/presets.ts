import { gaussian, seeded, type Rng } from "../model/random";

export type Point = { x: number; y: number; label: 0 | 1 };
export type Preset = "moons" | "circles" | "spiral" | "xor";

export const PRESETS: Preset[] = ["moons", "circles", "spiral", "xor"];

const PER_CLASS = 80;

const makers: Record<
  Preset,
  (rng: Rng, label: 0 | 1, i: number) => [number, number]
> = {
  moons: (rng, label) => {
    const t = Math.PI * rng();
    const x = label ? 1 - Math.cos(t) : Math.cos(t);
    const y = label ? 0.5 - Math.sin(t) : Math.sin(t);
    return [
      (x - 0.5) / 1.6 + gaussian(rng) * 0.06,
      (y - 0.25) / 1.6 + gaussian(rng) * 0.06,
    ];
  },
  circles: (rng, label) => {
    const a = 2 * Math.PI * rng();
    const r = (label ? 0.35 : 0.85) + gaussian(rng) * 0.05;
    return [r * Math.cos(a), r * Math.sin(a)];
  },
  spiral: (rng, label, i) => {
    const t = (i + 4) / (PER_CLASS + 4);
    const a = 3.5 * Math.PI * t + label * Math.PI;
    return [
      0.95 * t * Math.cos(a) + gaussian(rng) * 0.02,
      0.95 * t * Math.sin(a) + gaussian(rng) * 0.02,
    ];
  },
  xor: (rng, label) => {
    const side = () => (0.1 + 0.85 * rng()) * (rng() < 0.5 ? -1 : 1);
    const x = side();
    const y = Math.abs(side()) * Math.sign(x) * (label ? 1 : -1);
    return [x, y];
  },
};

export function makePreset(preset: Preset, seed = 7): Point[] {
  const rng = seeded(seed);
  return ([0, 1] as const).flatMap((label) =>
    Array.from({ length: PER_CLASS }, (_, i) => {
      const [x, y] = makers[preset](rng, label, i);
      return { x, y, label };
    }),
  );
}
