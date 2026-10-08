// format.js — how SkeVU writes numbers of the trace for people.

/** A time in microseconds as '12 μs', '3.4 ms' or '1.2 s'; null if there is none. */
export function formatDuration(us) {
  if (us == null || us < 0) return null;
  if (us >= 1e6)  return (us / 1e6).toFixed(1)  + ' s';
  if (us >= 1000) return (us / 1000).toFixed(1) + ' ms';
  return us.toFixed(0) + ' μs';
}
