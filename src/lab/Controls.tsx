
import type { Config } from "./useTrainer";

const LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
const PENALTIES = [0, 0.0001, 0.001, 0.01];

export const panel = "panel";
export const caption = "caption";
export const button = "btn";

type Option<T> = { value: T; label: string };

export function Choice<T extends string | number>({
  label,
  value,
  options,
  onPick,
}: {
  label: string;
  value: T;
  options: Option<T>[];
  onPick: (v: T) => void;
}) {
  return (
    <div className="field">
      <div className={caption}>{label}</div>
      <div
        role="radiogroup"
        aria-label={label}
        className="choices"
      >
        {options.map((o) => (
          <button
            key={o.label}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            onClick={() => onPick(o.value)}
            className="choice"
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

const options = <T extends string | number>(
  values: T[],
  show: (v: T) => string = String,
) => values.map((value) => ({ value, label: show(value) }));

export default function Controls({
  config,
  onChange,
}: {
  config: Config;
  onChange: (patch: Partial<Config>) => void;
}) {
  return (
    <div className="controls">
      <Choice
        label="Hidden layers"
        value={config.layers}
        options={options([1, 2, 3, 4])}
        onPick={(layers) => onChange({ layers })}
      />
      <Choice
        label="Width"
        value={config.width}
        options={options([2, 4, 8, 16])}
        onPick={(width) => onChange({ width })}
      />
      <Choice
        label="Activation"
        value={config.activation}
        options={options(["tanh", "relu"] as const)}
        onPick={(activation) => onChange({ activation })}
      />
      <Choice
        label="Optimizer"
        value={config.optimizer}
        options={options(["adam", "sgd"] as const, (v) =>
          v === "sgd" ? "sgd · momentum" : v,
        )}
        onPick={(optimizer) => onChange({ optimizer })}
      />
      <Choice
        label="Learning rate"
        value={config.lr}
        options={options(LEARNING_RATES)}
        onPick={(lr) => onChange({ lr })}
      />
      <Choice
        label="L2"
        value={config.l2}
        options={options(PENALTIES, (v) => (v ? v.toExponential(0) : "off"))}
        onPick={(l2) => onChange({ l2 })}
      />
    </div>
  );
}
