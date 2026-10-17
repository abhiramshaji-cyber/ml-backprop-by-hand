import { memo } from "react";
import type { Mlp } from "../../model/mlp";

const W = 320;
const H = 200;
const PAD = 16;

type Props = { mlp: Mlp; showGrad: boolean; tick: number };

function Network({ mlp, showGrad }: Props) {
  const column = (i: number) =>
    PAD + (i * (W - 2 * PAD)) / (mlp.sizes.length - 1);
  const row = (j: number, n: number) =>
    n === 1 ? H / 2 : PAD + (j * (H - 2 * PAD)) / (n - 1);
  const peak = (pick: (w: { data: number; grad: number }) => number) =>
    Math.max(
      1e-12,
      ...mlp.weights.map((w) => Math.abs(pick(w))).filter(Number.isFinite),
    );
  const top = peak((w) => (showGrad ? w.grad : w.data));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="network"
      role="img"
      aria-label={`Network ${mlp.sizes.join(" by ")}`}
    >
      {mlp.layers.map((layer, li) =>
        layer.w.map((weights, j) =>
          weights.map((w, k) => {
            const strength =
              Math.min(1, Math.abs(showGrad ? w.grad : w.data) / top) || 0;
            const tone = showGrad
              ? "var(--accent-3)"
              : w.data >= 0
                ? "var(--accent-1)"
                : "var(--accent-2)";
            return (
              <line
                key={`${li}-${j}-${k}`}
                x1={column(li)}
                y1={row(k, mlp.sizes[li])}
                x2={column(li + 1)}
                y2={row(j, mlp.sizes[li + 1])}
                stroke={tone}
                strokeWidth={0.4 + 3 * strength}
                strokeOpacity={0.15 + 0.8 * strength}
              />
            );
          }),
        ),
      )}
      {mlp.sizes.map((n, i) =>
        Array.from({ length: n }, (_, j) => (
          <circle
            key={`${i}-${j}`}
            cx={column(i)}
            cy={row(j, n)}
            r={4}
            fill="var(--card)"
            stroke="var(--foreground)"
            strokeOpacity={0.6}
          />
        )),
      )}
    </svg>
  );
}

export default memo(
  Network,
  (a, b) => a.tick === b.tick && a.showGrad === b.showGrad && a.mlp === b.mlp,
);
