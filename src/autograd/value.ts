export type Value = {
  data: number;
  grad: number;
  prev: Value[];
  back: () => void;
};

const idle = () => {};

export function value(data: number, prev: Value[] = []): Value {
  return { data, grad: 0, prev, back: idle };
}

export function backward(root: Value) {
  // order
  const order: Value[] = [];
  const seen = new Set<Value>();
  const stack: [Value, boolean][] = [[root, false]];
  while (stack.length) {
    const [node, expanded] = stack.pop()!;
    if (expanded) {
      order.push(node);
      continue;
    }
    if (seen.has(node)) continue;
    seen.add(node);
    stack.push([node, true]);
    for (const child of node.prev)
      if (!seen.has(child)) stack.push([child, false]);
  }

  // sweep
  root.grad = 1;
  for (let i = order.length - 1; i >= 0; i--) order[i].back();
}
