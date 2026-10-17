import { useEffect, useState } from "react";

const FALLBACK: Record<string, string> = {
  "--background": "#1a130e",
  "--foreground": "#eee2ce",
  "--accent-1": "#d9a340",
  "--accent-2": "#a9835b",
  "--accent-3": "#c7642e",
};

export function usePaletteColors(tokens: readonly string[]): string[] {
  const key = tokens.join(",");
  const [values, setValues] = useState(() => tokens.map((t) => FALLBACK[t]));

  useEffect(() => {
    const names = key.split(",");
    const read = () => {
      const style = getComputedStyle(document.documentElement);
      setValues(names.map((n) => style.getPropertyValue(n).trim() || FALLBACK[n]));
    };
    read();
    const scheme = matchMedia("(prefers-color-scheme: light)");
    scheme.addEventListener("change", read);
    return () => scheme.removeEventListener("change", read);
  }, [key]);

  return values;
}
