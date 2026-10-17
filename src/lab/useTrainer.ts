
import { useCallback, useEffect, useRef, useState } from "react";
import type { Point } from "../data/presets";
import { accuracy } from "../model/loss";
import { createMlp, type Activation, type Mlp } from "../model/mlp";
import {
  createOptimizer,
  type OptimizerKind,
  type Step,
} from "../model/optimizer";
import { gradCheck, trainStep } from "../model/train";

export type Config = {
  layers: number;
  width: number;
  activation: Activation;
  optimizer: OptimizerKind;
  lr: number;
  l2: number;
};

export type Run = {
  mlp: Mlp;
  step: Step;
  epoch: number;
  loss: number;
  acc: number;
  history: number[];
  diverged: boolean;
};

const BUDGET_MS = 10;
const HISTORY_CAP = 600;
const CALM_FRAME_MS = 400;

function fresh(config: Config, seed: number): Run {
  const mlp = createMlp(
    [2, ...Array<number>(config.layers).fill(config.width), 1],
    config.activation,
    seed,
  );
  return {
    mlp,
    step: createOptimizer(config.optimizer, mlp.params),
    epoch: 0,
    loss: NaN,
    acc: 0,
    history: [],
    diverged: false,
  };
}

function record(run: Run, loss: number) {
  run.epoch += 1;
  run.loss = loss;
  if (!Number.isFinite(loss)) run.diverged = true;
  run.history.push(loss);
  if (run.history.length > HISTORY_CAP)
    run.history = run.history.filter((_, i) => i % 2 === 1);
}

export function useTrainer(points: Point[], config: Config, calm: boolean) {
  const [seed, setSeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const [initial] = useState(() => fresh(config, 1));
  const run = useRef(initial);
  const live = useRef({ points, config });

  useEffect(() => {
    live.current = { points, config };
  });

  // rebuild
  useEffect(() => {
    run.current = fresh(live.current.config, seed);
    run.current.acc = accuracy(run.current.mlp, live.current.points);
    setFrame((f) => f + 1);
  }, [config.layers, config.width, config.activation, seed]);

  // optimizer
  useEffect(() => {
    run.current.step = createOptimizer(
      config.optimizer,
      run.current.mlp.params,
    );
  }, [config.optimizer]);

  const advance = useCallback((budget: number) => {
    const { points, config } = live.current;
    const r = run.current;
    if (!points.length || r.diverged) return;
    const start = performance.now();
    do record(r, trainStep(r.mlp, r.step, points, config.l2, config.lr));
    while (!r.diverged && performance.now() - start < budget);
    r.acc = accuracy(r.mlp, points);
  }, []);

  // loop
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    let painted = 0;
    const tick = (now: number) => {
      advance(BUDGET_MS);
      if (run.current.diverged || !live.current.points.length)
        setPlaying(false);
      if (!calm || now - painted > CALM_FRAME_MS) {
        painted = now;
        setFrame((f) => f + 1);
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, calm, advance]);

  // points
  useEffect(() => {
    run.current.acc = accuracy(run.current.mlp, points);
    setFrame((f) => f + 1);
  }, [points]);

  const stepOnce = useCallback(() => {
    advance(0);
    setFrame((f) => f + 1);
  }, [advance]);

  const reset = useCallback(() => setSeed((s) => s + 1), []);

  const check = useCallback(
    () =>
      gradCheck(run.current.mlp, live.current.points, live.current.config.l2),
    [],
  );

  return {
    run: run.current,
    frame,
    playing,
    setPlaying,
    stepOnce,
    reset,
    check,
  };
}
