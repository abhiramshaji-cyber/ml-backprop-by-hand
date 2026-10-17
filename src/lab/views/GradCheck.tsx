import { button, panel } from "../Controls";

const PASS = 1e-4;

export type CheckResult = { maxRelError: number; checked: number };

export default function GradCheck({
  result,
  disabled,
  onCheck,
}: {
  result: CheckResult | null;
  disabled: boolean;
  onCheck: () => void;
}) {
  return (
    <div className={`${panel} stack-tight`}>
      <button
        type="button"
        className={button}
        onClick={onCheck}
        disabled={disabled}
      >
        gradient check
      </button>
      <p className="note" aria-live="polite">
        {result
          ? `autograd vs finite differences · ${result.checked} params · max rel error ${result.maxRelError.toExponential(2)} · ${result.maxRelError < PASS ? "pass" : "check relu kinks"}`
          : "compares every backprop gradient with a central difference"}
      </p>
    </div>
  );
}
