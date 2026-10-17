import { caption, panel } from "../Controls";
import type { Run } from "../useTrainer";
import LossCurve from "./LossCurve";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="stat">
        {value}
      </div>
      <div className={caption}>{label}</div>
    </div>
  );
}

export default function Scoreboard({ run }: { run: Run }) {
  return (
    <div className={`${panel} scoreboard`}>
      <Stat label="epoch" value={String(run.epoch)} />
      <Stat
        label="loss"
        value={Number.isFinite(run.loss) ? run.loss.toFixed(3) : "·"}
      />
      <Stat label="accuracy" value={`${Math.round(run.acc * 100)}%`} />
      <div className="span-all">
        <LossCurve history={run.history} />
      </div>
    </div>
  );
}
