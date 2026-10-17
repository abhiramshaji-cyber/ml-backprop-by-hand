import { mul, relu, scale, sum, tanh } from "../autograd/ops";
import { value, type Value } from "../autograd/value";
import { gaussian, seeded } from "./random";

export type Activation = "tanh" | "relu";
export type Layer = { w: Value[][]; b: Value[] };
export type Mlp = {
  sizes: number[];
  activation: Activation;
  layers: Layer[];
  params: Value[];
  weights: Value[];
};

export function createMlp(
  sizes: number[],
  activation: Activation,
  seed: number,
): Mlp {
  const rng = seeded(seed);
  const layers = sizes.slice(1).map((fanOut, i) => {
    const fanIn = sizes[i];
    const gain = activation === "relu" && i < sizes.length - 2 ? 2 : 1;
    const spread = Math.sqrt(gain / fanIn);
    return {
      w: Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, () => value(gaussian(rng) * spread)),
      ),
      b: Array.from({ length: fanOut }, () =>
        value(activation === "relu" ? 0.01 : 0),
      ),
    };
  });
  const weights = layers.flatMap((l) => l.w.flat());
  return {
    sizes,
    activation,
    layers,
    weights,
    params: [...weights, ...layers.flatMap((l) => l.b)],
  };
}

export function forward(mlp: Mlp, point: readonly number[]): Value {
  const squash = mlp.activation === "tanh" ? tanh : relu;
  const last = mlp.layers.length - 1;
  let h: Value[] = [];
  mlp.layers.forEach((layer, li) => {
    h = layer.w.map((row, j) => {
      const terms = row.map((w, k) =>
        li === 0 ? scale(w, point[k]) : mul(w, h[k]),
      );
      const z = sum([...terms, layer.b[j]]);
      return li === last ? z : squash(z);
    });
  });
  return h[0];
}

export function predict(mlp: Mlp, point: readonly number[]): number {
  const last = mlp.layers.length - 1;
  let h: number[] = [...point];
  mlp.layers.forEach((layer, li) => {
    h = layer.w.map((row, j) => {
      let z = layer.b[j].data;
      for (let k = 0; k < row.length; k++) z += row[k].data * h[k];
      if (li === last) return z;
      return mlp.activation === "tanh" ? Math.tanh(z) : Math.max(0, z);
    });
  });
  return h[0];
}
