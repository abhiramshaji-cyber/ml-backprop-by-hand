const W = 320;
const H = 90;

export default function LossCurve({ history }: { history: number[] }) {
  const finite = history.filter(Number.isFinite);
  const top = Math.max(1e-9, ...finite);
  const line = finite
    .map(
      (v, i) =>
        `${((i / Math.max(finite.length - 1, 1)) * W).toFixed(1)},${(H - 4 - (v / top) * (H - 8)).toFixed(1)}`,
    )
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="curve"
      role="img"
      aria-label="Loss curve"
    >
      <line x1={0} y1={H - 4} x2={W} y2={H - 4} stroke="var(--border)" />
      {finite.length > 1 && (
        <polyline
          points={line}
          fill="none"
          stroke="var(--accent-3)"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}
