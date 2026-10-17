import type { Value } from "../autograd/value";

export type OptimizerKind = "sgd" | "adam";
export type Step = (lr: number) => void;

const MOMENTUM = 0.9;
const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;

export function createOptimizer(kind: OptimizerKind, params: Value[]): Step {
  const m = new Float64Array(params.length);
  const v = new Float64Array(params.length);
  let t = 0;

  if (kind === "sgd")
    return (lr) => {
      params.forEach((p, i) => {
        m[i] = MOMENTUM * m[i] - lr * p.grad;
        p.data += m[i];
      });
    };

  return (lr) => {
    t += 1;
    const fix1 = 1 - BETA1 ** t;
    const fix2 = 1 - BETA2 ** t;
    params.forEach((p, i) => {
      m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad;
      v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad * p.grad;
      p.data -= (lr * (m[i] / fix1)) / (Math.sqrt(v[i] / fix2) + EPS);
    });
  };
}
