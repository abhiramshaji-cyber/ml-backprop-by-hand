
import { useEffect, useRef, type MouseEvent } from "react";
import { mix, parseHex } from "../../theme/color";
import { usePaletteColors } from "../../theme/usePaletteColors";
import { sigmoidOf } from "../../autograd/ops";
import type { Point } from "../../data/presets";
import { predict, type Mlp } from "../../model/mlp";

export const SPAN = 1.15;
const GRID = 56;
const SIZE = 480;
const RADIUS = 5;

const toPx = (v: number) => ((v + SPAN) / (2 * SPAN)) * SIZE;
const toWorld = (px: number) => (px / SIZE) * 2 * SPAN - SPAN;

type Props = {
  mlp: Mlp;
  points: Point[];
  frame: number;
  onPlace: (x: number, y: number, flip: boolean) => void;
};

export default function Board({ mlp, points, frame, onPlace }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const field = useRef<HTMLCanvasElement | null>(null);
  const [bg, ink, a, b] = usePaletteColors([
    "--background",
    "--foreground",
    "--accent-1",
    "--accent-2",
  ]);

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.current!.width !== SIZE * dpr)
      canvas.current!.width = canvas.current!.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // field
    field.current ??= document.createElement("canvas");
    const off = field.current;
    off.width = off.height = GRID;
    const fctx = off.getContext("2d")!;
    const img = fctx.createImageData(GRID, GRID);
    const [base, zero, one] = [parseHex(bg), parseHex(a), parseHex(b)];
    for (let row = 0; row < GRID; row++)
      for (let col = 0; col < GRID; col++) {
        const x = ((col + 0.5) / GRID) * 2 * SPAN - SPAN;
        const y = SPAN - ((row + 0.5) / GRID) * 2 * SPAN;
        const p = sigmoidOf(predict(mlp, [x, y]));
        const sure = Number.isFinite(p) ? Math.abs(2 * p - 1) : 0;
        const rgb = mix(base, p > 0.5 ? one : zero, 0.12 + 0.38 * sure);
        img.data.set([rgb[0], rgb[1], rgb[2], 255], (row * GRID + col) * 4);
      }
    fctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, SIZE, SIZE);

    // axes
    ctx.strokeStyle = ink;
    ctx.globalAlpha = 0.12;
    ctx.beginPath();
    ctx.moveTo(SIZE / 2, 0);
    ctx.lineTo(SIZE / 2, SIZE);
    ctx.moveTo(0, SIZE / 2);
    ctx.lineTo(SIZE, SIZE / 2);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // points
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = bg;
    for (const p of points) {
      const px = toPx(p.x);
      const py = SIZE - toPx(p.y);
      ctx.fillStyle = p.label ? b : a;
      ctx.beginPath();
      if (p.label) ctx.rect(px - RADIUS, py - RADIUS, RADIUS * 2, RADIUS * 2);
      else ctx.arc(px, py, RADIUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [mlp, points, frame, bg, ink, a, b]);

  const place = (e: MouseEvent<HTMLCanvasElement>, flip: boolean) => {
    e.preventDefault();
    const box = e.currentTarget.getBoundingClientRect();
    const x = toWorld(((e.clientX - box.left) / box.width) * SIZE);
    const y = -toWorld(((e.clientY - box.top) / box.height) * SIZE);
    onPlace(x, y, flip || e.shiftKey);
  };

  return (
    <canvas
      ref={canvas}
      onClick={(e) => place(e, false)}
      onContextMenu={(e) => place(e, true)}
      aria-label="Training canvas. Tap to place a point, right click or shift click for the other class."
      className="board"
    />
  );
}
