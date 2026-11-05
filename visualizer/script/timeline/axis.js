// timeline/axis.js — the time axis: the format of a duration and the ticks of a stretch of time. Pure.
'use strict';

/** "2.00s", "1.5ms", "3.0µs". */
export function formatDuration(us) {
    if (us >= 1e6) return (us / 1e6).toFixed(2) + 's';
    if (us >= 1e3) return (us / 1e3).toFixed(1) + 'ms';
    return us.toFixed(1) + 'µs';
}

/** The least of 1, 2, 5 × 10^k that is at least `min`. */
export function niceStep(min) {
    if (!(min > 0)) return 1;
    const exp = Math.floor(Math.log10(min));
    const base = Math.pow(10, exp);
    for (const m of [1, 2, 5, 10]) if (m * base >= min * (1 - 1e-9)) return m * base;
    return 10 * base;
}

const UNITS = [[1e6, 's'], [1e3, 'ms'], [1, 'µs']];

/** "+1.2ms"; digits enough to tell it from the tick `step` later. */
export function formatTick(us, step) {
    const ref = Math.max(Math.abs(us), step);
    const [size, unit] = UNITS.find(([s]) => ref >= s) ?? UNITS[2];
    const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(step / size) - 1e-9)));
    return '+' + (us / size).toFixed(decimals) + unit;
}

/** Ticks between `from` and `to` (µs after the start), at least `minPx` apart at `pxPerUs`: [{ t, label }] at multiples of a round step. */
export function ticks(from, to, pxPerUs, minPx) {
    if (!(pxPerUs > 0) || to < from) return [];
    const step = niceStep(minPx / pxPerUs);
    const out = [];
    for (let k = Math.max(0, Math.ceil(from / step)); k * step <= to; k++) {
        const t = k * step;
        out.push({ t, label: formatTick(t, step) });
    }
    return out;
}
