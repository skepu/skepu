// test_menu_overflow.js — which titles of the menu bar collapse into the "…" menu when the bar is too narrow.
import { describe, it, expect } from 'vitest';
import { collapseCount } from '../../script/menu.js';

describe('collapseCount', () => {
    const widths = [50, 50, 60, 70, 80];    // five titles; 310 together
    it('collapses nothing when all fit', () => {
        expect(collapseCount(widths, 100, 30, 410)).toBe(0);
        expect(collapseCount(widths, 100, 30, 1000)).toBe(0);
    });
    it('collapses from the right, as few as needed, and counts the width of the "…" title', () => {
        expect(collapseCount(widths, 100, 30, 409)).toBe(1);   // 100 + 30 + 230 = 360
        expect(collapseCount(widths, 100, 30, 360)).toBe(1);
        expect(collapseCount(widths, 100, 30, 359)).toBe(2);   // 100 + 30 + 160 = 290 (without the 60)
        expect(collapseCount(widths, 100, 30, 289)).toBe(3);
    });
    it('collapses all of them if even one does not fit with the "…" title, and has nothing to collapse without titles', () => {
        expect(collapseCount(widths, 100, 30, 120)).toBe(5);
        expect(collapseCount([], 100, 30, 50)).toBe(0);
    });
});
