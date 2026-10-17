import { mul, scale, softplus, softplusOf, sum } from "../autograd/ops";
import type { Value } from "../autograd/value";
import type { Point } from "../data/presets";
import { forward, predict, type Mlp } from "./mlp";

const signed = (p: Point) => (p.label === 1 ? -1 : 1);

export function graphLoss(mlp: Mlp, points: Point[], l2: number): Value {
  const data = points.map((p) =>
    softplus(scale(forward(mlp, [p.x, p.y]), signed(p))),
  );
  const fit = scale(sum(data), 1 / Math.max(points.length, 1));
  if (l2 === 0) return fit;
  return sum([fit, scale(sum(mlp.weights.map((w) => mul(w, w))), l2)]);
}

export function numericLoss(mlp: Mlp, points: Point[], l2: number): number {
  const fit = points.reduce(
    (acc, p) => acc + softplusOf(signed(p) * predict(mlp, [p.x, p.y])),
    0,
  );
  const penalty = mlp.weights.reduce((acc, w) => acc + w.data * w.data, 0);
  return fit / Math.max(points.length, 1) + l2 * penalty;
}

export function accuracy(mlp: Mlp, points: Point[]): number {
  if (!points.length) return 0;
  const right = points.filter(
    (p) => (predict(mlp, [p.x, p.y]) > 0 ? 1 : 0) === p.label,
  ).length;
  return right / points.length;
}
