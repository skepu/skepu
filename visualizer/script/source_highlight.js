// source_highlight.js — background of the selected source lines: the accent over the code theme's background, as strong as the text contrast allows.
// Pure: colours come in as CSS strings or { r, g, b }.

/** { r, g, b } (0–255) of `rgb(...)`, `rgba(...)` or `#rrggbb`; a colour with alpha is put over `over`. */
export function parseColor(css, over = { r: 255, g: 255, b: 255 }) {
  const s = String(css).trim();
  let m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(s);
  if (m) {
    const h = m[1].length === 3 ? m[1].replace(/./g, c => c + c) : m[1];
    return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
  }
  m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:[\s,/]+([\d.]+%?))?\s*\)$/i.exec(s);
  if (!m) return null;
  const a = m[4] === undefined ? 1 : m[4].endsWith('%') ? parseFloat(m[4]) / 100 : +m[4];
  const at = (c, o) => +c * a + o * (1 - a);
  return { r: at(m[1], over.r), g: at(m[2], over.g), b: at(m[3], over.b) };
}

/** `a` at `fraction` over `b`. */
export function mix(a, b, fraction) {
  return { r: a.r * fraction + b.r * (1 - fraction), g: a.g * fraction + b.g * (1 - fraction), b: a.b * fraction + b.b * (1 - fraction) };
}

const channel = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
const luminance = c => 0.2126 * channel(c.r) + 0.7152 * channel(c.g) + 0.0722 * channel(c.b);

/** WCAG contrast ratio, 1–21. */
export function contrast(a, b) {
  const x = luminance(a), y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

/**
 /**
  * How much text contrast the highlight may cost against the plain background: the base text (first colour) keeps BASE_KEEP of it, at most BASE_ENOUGH;
  * syntax tokens keep TOKEN_KEEP, at most TOKEN_ENOUGH. Tokens may dim more, as a comment is dim already.
  */
const BASE_KEEP = 0.9, BASE_ENOUGH = 4.5, TOKEN_KEEP = 0.55, TOKEN_ENOUGH = 3;
/** Strongest and weakest accent mix; the weakest applies when nothing stronger keeps the text contrast. */
const MAX_FRACTION = 0.6, MIN_FRACTION = 0.06, STEP = 0.02;

/**
 /** The background of a selected line: `accent` over `background`, as strong as the text contrast allows. `texts[0]` is the base text colour. Returns { color, fraction }. */
export function selectionBackground(background, texts, accent) {
  const floors = texts.map((t, i) => {
    const own = contrast(t, background);
    return i === 0 ? Math.min(own, BASE_ENOUGH) * BASE_KEEP : Math.min(own, TOKEN_ENOUGH) * TOKEN_KEEP;
  });
  const holds = f => { const bg = mix(accent, background, f); return texts.every((t, i) => contrast(t, bg) >= floors[i]); };
  let fraction = MIN_FRACTION;
  for (let f = MAX_FRACTION; f >= MIN_FRACTION; f -= STEP) if (holds(f)) { fraction = f; break; }
  return { color: mix(accent, background, fraction), fraction };
}

export function cssColor(c) {
  return 'rgb(' + [c.r, c.g, c.b].map(v => Math.round(Math.max(0, Math.min(255, v)))).join(', ') + ')';
}
