import { backward } from "../autograd/value";
import type { Point } from "../data/presets";
import { graphLoss, numericLoss } from "./loss";
import type { Mlp } from "./mlp";
import type { Step } from "./optimizer";

const H = 1e-5;
const FLOOR = 1e-7;

function gradients(mlp: Mlp, points: Point[], l2: number): number {
  for (const p of mlp.params) p.grad = 0;
  const loss = graphLoss(mlp, points, l2);
  backward(loss);
  return loss.data;
}

export function trainStep(
  mlp: Mlp,
  step: Step,
  points: Point[],
  l2: number,
  lr: number,
): number {
  const loss = gradients(mlp, points, l2);
  step(lr);
  return loss;
}

export function gradCheck(mlp: Mlp, points: Point[], l2: number) {
  gradients(mlp, points, l2);
  let maxRelError = 0;
  for (const p of mlp.params) {
    const original = p.data;
    p.data = original + H;
    const up = numericLoss(mlp, points, l2);
    p.data = original - H;
    const down = numericLoss(mlp, points, l2);
    p.data = original;
    const numeric = (up - down) / (2 * H);
    const error =
      Math.abs(p.grad - numeric) /
      Math.max(Math.abs(p.grad), Math.abs(numeric), FLOOR);
    maxRelError = Math.max(maxRelError, error);
  }
  return { maxRelError, checked: mlp.params.length };
}
