
import { useEffect, useState } from "react";
import { makePreset, PRESETS, type Point, type Preset } from "../data/presets";
import Controls, { button, caption, Choice, panel } from "./Controls";
import { useTrainer, type Config } from "./useTrainer";
import Board from "./views/Board";
import GradCheck, { type CheckResult } from "./views/GradCheck";
import Network from "./views/Network";
import Scoreboard from "./views/Scoreboard";

const MAX_POINTS = 400;
const NETWORK_EVERY = 6;
const START: Config = {
  layers: 2,
  width: 8,
  activation: "tanh",
  optimizer: "adam",
  lr: 0.03,
  l2: 0,
};
const DATA = [...PRESETS, "custom" as const].map((value) => ({
  value,
  label: value,
}));
const BRUSHES = [
  { value: 0 as const, label: "● class a" },
  { value: 1 as const, label: "■ class b" },
];

export default function Lab() {
  const [preset, setPreset] = useState<Preset | "custom">("moons");
  const [points, setPoints] = useState<Point[]>(() => makePreset("moons"));
  const [brush, setBrush] = useState<0 | 1>(0);
  const [config, setConfig] = useState(START);
  const [showGrad, setShowGrad] = useState(false);
  const [calm, setCalm] = useState(false);
  const [checked, setChecked] = useState<CheckResult | null>(null);
  const { run, frame, playing, setPlaying, stepOnce, reset, check } =
    useTrainer(points, config, calm);
  const idle = run.diverged || !points.length;

  useEffect(() => {
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    setCalm(reduce);
    setPlaying(!reduce);
  }, [setPlaying]);

  const place = (x: number, y: number, flip: boolean) => {
    const label: 0 | 1 = (flip ? 1 - brush : brush) === 1 ? 1 : 0;
    setPreset("custom");
    setPoints((prev) => [...prev, { x, y, label }].slice(-MAX_POINTS));
  };

  const restart = () => {
    setChecked(null);
    reset();
  };

  const load = (next: Preset | "clear") => {
    setPreset(next === "clear" ? "custom" : next);
    setPoints(next === "clear" ? [] : makePreset(next));
    restart();
  };

  const gradientCheck = () => {
    setChecked(check());
  };

  const status = run.diverged
    ? "diverged · reset"
    : !points.length
      ? "place some points"
      : playing
        ? "training"
        : "paused";

  return (
    <div className="lab">
      <div className="stack">
        <Board mlp={run.mlp} points={points} frame={frame} onPlace={place} />
        <div className="toolbar">
          <button
            type="button"
            className={button}
            disabled={idle}
            onClick={() => setPlaying(!playing)}
          >
            {playing ? "pause" : "play"}
          </button>
          <button
            type="button"
            className={button}
            disabled={playing || idle}
            onClick={() => stepOnce()}
          >
            step
          </button>
          <button
            type="button"
            className={button}
            onClick={() => restart()}
          >
            reset
          </button>
          <span className={`${caption} status`} aria-live="polite">
            {status}
          </span>
        </div>
        <div className="row">
          <Choice
            label="Data"
            value={preset}
            options={DATA}
            onPick={(v) => v !== "custom" && load(v)}
          />
          <div className="field">
            <div className={caption}>Edit</div>
            <button
              type="button"
              className={button}
              onClick={() => load("clear")}
            >
              clear
            </button>
          </div>
          <Choice
            label="Tap places"
            value={brush}
            options={BRUSHES}
            onPick={setBrush}
          />
        </div>
      </div>

      <div className="stack">
        <Scoreboard run={run} />
        <div className={panel}>
          <div className="panel-head">
            <span className={caption}>
              {showGrad
                ? "|gradient| per weight"
                : "thickness |w| · colour sign"}
            </span>
            <button
              type="button"
              className={button}
              aria-pressed={showGrad}
              onClick={() => setShowGrad(!showGrad)}
            >
              {showGrad ? "show weights" : "show gradients"}
            </button>
          </div>
          <Network
            mlp={run.mlp}
            showGrad={showGrad}
            tick={
              playing
                ? Math.floor(frame / NETWORK_EVERY) * NETWORK_EVERY
                : frame
            }
          />
        </div>
        <GradCheck
          result={checked}
          disabled={!points.length}
          onCheck={gradientCheck}
        />
        <div className={panel}>
          <Controls
            config={config}
            onChange={(patch) => {
              setChecked(null);
              setConfig((c) => ({ ...c, ...patch }));
            }}
          />
        </div>
      </div>
    </div>
  );
}
