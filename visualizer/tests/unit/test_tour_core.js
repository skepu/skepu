import { describe, it, expect } from 'vitest';
import { visibleSteps, stepIndex, spotRect, placePopover, placeInCorner } from '../../script/tour_core.js';

const viewport = { width: 1000, height: 700 };
const size = { width: 300, height: 140 };

describe('the steps', () => {
  it('leave out those that do not apply', () => {
    const steps = [{ id: 'a' }, { id: 'b', when: ctx => !ctx.vscode }, { id: 'c', when: () => true }];
    expect(visibleSteps(steps, { vscode: false }).map(s => s.id)).toEqual(['a', 'b', 'c']);
    expect(visibleSteps(steps, { vscode: true }).map(s => s.id)).toEqual(['a', 'c']);
  });
  it('have a next and a previous one, until the ends', () => {
    expect(stepIndex(0, 1, 3)).toBe(1);
    expect(stepIndex(2, 1, 3)).toBeNull();
    expect(stepIndex(0, -1, 3)).toBeNull();
    expect(stepIndex(2, -1, 3)).toBe(1);
  });
});

describe('the spotlight', () => {
  it('is the target with some room, inside the viewport', () => {
    expect(spotRect({ left: 100, top: 50, width: 200, height: 40 }, 6, viewport)).toEqual({ left: 94, top: 44, width: 212, height: 52 });
    expect(spotRect({ left: 0, top: 0, width: 2000, height: 40 }, 6, viewport)).toEqual({ left: 0, top: 0, width: 1000, height: 46 });
  });
});

describe('the popover', () => {
  it('is below a target that has room below', () => {
    const p = placePopover({ left: 100, top: 100, width: 200, height: 40 }, size, viewport);
    expect(p.side).toBe('bottom');
    expect(p.top).toBe(154);
    expect(p.left).toBe(50);
  });
  it('is above a target at the bottom of the viewport', () => {
    const p = placePopover({ left: 100, top: 650, width: 200, height: 40 }, size, viewport);
    expect(p.side).toBe('top');
    expect(p.top).toBe(650 - 140 - 14);
  });
  it('is shifted along the side, not moved to another side, at the edge of the viewport', () => {
    const p = placePopover({ left: 0, top: 100, width: 40, height: 40 }, size, viewport);
    expect(p.side).toBe('bottom');
    expect(p.left).toBe(12);
  });
  it('tries the side that is preferred first', () => {
    const p = placePopover({ left: 100, top: 300, width: 100, height: 40 }, size, viewport, 'right');
    expect(p.side).toBe('right');
    expect(p.left).toBe(214);
  });
  it('is inside a target that fills the viewport', () => {
    const p = placePopover({ left: 0, top: 0, width: 1000, height: 700 }, size, viewport);
    expect(p.side).toBe('inside');
    expect(p.top).toBe(700 - 140 - 12);
  });
  it('is in the middle without a target', () => {
    expect(placePopover(null, size, viewport)).toEqual({ left: 350, top: 280, side: 'center' });
  });
  it('is always inside the viewport', () => {
    for (const rect of [{ left: 990, top: 690, width: 5, height: 5 }, { left: -20, top: -20, width: 10, height: 10 }, { left: 500, top: 350, width: 1, height: 1 }]) {
      const p = placePopover(rect, size, viewport);
      expect(p.left).toBeGreaterThanOrEqual(12); expect(p.top).toBeGreaterThanOrEqual(12);
      expect(p.left + size.width).toBeLessThanOrEqual(988); expect(p.top + size.height).toBeLessThanOrEqual(688);
    }
  });
});

describe('placeInCorner', () => {
  const viewport = { width: 1000, height: 700 }, size = { width: 300, height: 200 };
  it('puts the popover in the bottom-left corner of the rectangle, inside it', () => {
    expect(placeInCorner({ left: 0, top: 22, width: 640, height: 480 }, size, viewport)).toEqual({ left: 12, top: 290, side: 'corner' });
  });
  it('keeps it inside the viewport', () => {
    expect(placeInCorner({ left: 900, top: 22, width: 640, height: 2000 }, size, viewport)).toEqual({ left: 688, top: 488, side: 'corner' });
  });
});
