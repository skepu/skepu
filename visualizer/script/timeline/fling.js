// timeline/fling.js — momentum of a released drag. Pure.
'use strict';

export const FLING = {
    tauMs:     200,    // time for the speed to fall to 1/e
    minSpeed:  0.03,   // px/ms; below this it stops
    windowMs:  100,    // the release speed is measured over this last stretch
    staleMs:   60,     // no momentum after standing still this long
};

/** Speed at the release, px/ms: { x, y }. `samples`: [{ t (ms), x, y }] of the scroll position. */
export function releaseVelocity(samples, now, { windowMs, staleMs } = FLING) {
    const recent = samples.filter(s => s.t >= now - windowMs);
    if (recent.length < 2) return { x: 0, y: 0 };
    const first = recent[0], last = recent[recent.length - 1], dt = last.t - first.t;
    if (dt <= 0 || now - last.t > staleMs) return { x: 0, y: 0 };
    return { x: (last.x - first.x) / dt, y: (last.y - first.y) / dt };
}

/** The speed `dt` ms later. */
export function decayVelocity(v, dt, { tauMs } = FLING) {
    const k = Math.exp(-dt / tauMs);
    return { x: v.x * k, y: v.y * k };
}

export const isMoving = (v, { minSpeed } = FLING) => Math.hypot(v.x, v.y) >= minSpeed;
