import { value, type Value } from "./value";

export function mul(a: Value, b: Value): Value {
  const out = value(a.data * b.data, [a, b]);
  out.back = () => {
    a.grad += b.data * out.grad;
    b.grad += a.data * out.grad;
  };
  return out;
}

export function scale(a: Value, k: number): Value {
  const out = value(a.data * k, [a]);
  out.back = () => {
    a.grad += k * out.grad;
  };
  return out;
}

export function sum(terms: Value[]): Value {
  const out = value(
    terms.reduce((acc, t) => acc + t.data, 0),
    terms,
  );
  out.back = () => {
    for (const t of terms) t.grad += out.grad;
  };
  return out;
}

export function tanh(a: Value): Value {
  const t = Math.tanh(a.data);
  const out = value(t, [a]);
  out.back = () => {
    a.grad += (1 - t * t) * out.grad;
  };
  return out;
}

export function relu(a: Value): Value {
  const out = value(Math.max(0, a.data), [a]);
  out.back = () => {
    if (a.data > 0) a.grad += out.grad;
  };
  return out;
}

export const softplusOf = (x: number) =>
  Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
export const sigmoidOf = (x: number) =>
  x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));

export function softplus(a: Value): Value {
  const out = value(softplusOf(a.data), [a]);
  out.back = () => {
    a.grad += sigmoidOf(a.data) * out.grad;
  };
  return out;
}
