// timeline/cursors.js — the hover line, the line a click locks, and the distance between them. Positions are x of the content; `origin` is where the shown window starts.
'use strict';

import { METRICS as M } from './style.js';
import { formatDuration } from './axis.js';

const NS = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, parent = null) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
}

export class Cursors {
    /** `plot` and `axis` are the window svgs; `limits()` gives { min, max }, the x range where a time can be placed. */
    constructor(plot, axis, limits) {
        this._limits = limits;
        this.origin = 0;
        this.hoverX = null;
        this.lockX = null;
        this._pal = null;

        const lines = el('g', { class: 'pointer-events-none' }, plot);
        this._hoverLine = el('line', { y1: 0 }, lines);
        this._lockLine = el('line', { y1: 0, 'stroke-width': 1.5 }, lines);
        const chips = el('g', { class: 'pointer-events-none' }, axis);
        this._hover = this._chip(chips, 0.5, 0.92);
        this._lock = this._chip(chips, 0.8, 0.95);
        this._delta = el('g', {}, chips);
        this._shaft = el('line', {}, this._delta);
        this._arrowL = el('polygon', {}, this._delta);
        this._arrowR = el('polygon', {}, this._delta);
        this._deltaChip = this._chip(this._delta, 0.6, 0.93);
        this._deltaChip.text.setAttribute('dominant-baseline', 'middle');
        this.hideAll();
    }

    _chip(parent, strokeWidth, opacity) {
        const g = el('g', {}, parent);
        const rect = el('rect', { y: M.chipY, height: M.chipH, rx: 2, 'stroke-width': strokeWidth, opacity }, g);
        const text = el('text', { y: M.chipY + M.chipH - 3, 'text-anchor': 'middle' }, g);
        return { g, rect, text };
    }

    setPalette(pal) {
        this._pal = pal;
        this._hoverLine.setAttribute('stroke', pal.hoverStroke);
        this._lockLine.setAttribute('stroke', pal.lockStroke);
        this._hover.rect.setAttribute('fill', pal.hoverChip); this._hover.rect.setAttribute('stroke', pal.hoverStroke); this._hover.text.setAttribute('fill', pal.hoverText);
        this._lock.rect.setAttribute('fill', pal.lockChip); this._lock.rect.setAttribute('stroke', pal.lockChipStroke); this._lock.text.setAttribute('fill', pal.lockText);
        this._shaft.setAttribute('stroke', pal.delta);
        this._arrowL.setAttribute('fill', pal.delta); this._arrowR.setAttribute('fill', pal.delta);
        this._deltaChip.rect.setAttribute('fill', pal.deltaChip); this._deltaChip.rect.setAttribute('stroke', pal.deltaChipStroke);
        this._deltaChip.text.setAttribute('fill', pal.deltaText); this._deltaChip.text.setAttribute('font-size', 10);
    }

    /** Sets the line height (the rows) and redraws; call after the window moves. */
    update(rowsHeight, hoverText, lockText) {
        this._hoverLine.setAttribute('y2', rowsHeight);
        this._lockLine.setAttribute('y2', rowsHeight);
        this.hoverText = hoverText ?? this.hoverText;
        this.lockText = lockText ?? this.lockText;
        this._draw();
    }

    show(kind, x, text) {
        if (kind === 'hover') { this.hoverX = x; this.hoverText = text; } else { this.lockX = x; this.lockText = text; }
        this._draw();
    }

    /** Whether `x` is on the locked line, or on its time chip when `inAxis`. */
    hitsLock(x, inAxis) {
        if (this.lockX == null) return false;
        if (Math.abs(x - this.lockX) <= M.lockHitPx) return true;
        if (!inAxis) return false;
        const left = +this._lock.rect.getAttribute('x') + this.origin;
        return x >= left && x <= left + +this._lock.rect.getAttribute('width');
    }

    hideHover() { this.hoverX = null; this._draw(); }
    unlock() { this.lockX = null; this._draw(); }
    hideAll() { this.hoverX = null; this.lockX = null; this._draw(); }

    _place(chip, line, x, text) {
        const visible = x != null;
        chip.g.style.display = visible ? '' : 'none';
        if (line) line.style.display = visible ? '' : 'none';
        if (!visible) return;
        chip.text.textContent = text;
        let width;
        try { width = chip.text.getBBox().width; } catch (e) { width = 0; }
        if (!width) width = text.length * 6.5;
        const left = Math.max(this._limits().min, Math.min(this._limits().max - width - 2 * M.chipPadX, x - width / 2 - M.chipPadX));
        chip.rect.setAttribute('x', left - this.origin);
        chip.rect.setAttribute('width', width + 2 * M.chipPadX);
        chip.text.setAttribute('x', left - this.origin + M.chipPadX + width / 2);
        if (line) { line.setAttribute('x1', x - this.origin); line.setAttribute('x2', x - this.origin); }
    }

    _draw() {
        this._place(this._hover, this._hoverLine, this.hoverX, this.hoverText ?? '');
        this._place(this._lock, this._lockLine, this.lockX, this.lockText ?? '');
        this._drawDelta();
    }

    /** Arrow and time between the lock and the pointer. `perUs`: px per µs. */
    _drawDelta() {
        const show = this.hoverX != null && this.lockX != null && this.perUs > 0;
        this._delta.style.display = show ? '' : 'none';
        if (!show) return;
        const lx = this.lockX, hx = this.hoverX, o = this.origin, y = M.deltaY, a = M.arrowHead;
        const dt = Math.abs(hx - lx) / this.perUs;
        const text = (hx >= lx ? '+' : '-') + formatDuration(dt);
        const mid = (lx + hx) / 2 - o;
        const lo = Math.min(lx, hx) - o, hi = Math.max(lx, hx) - o;
        const t = this._deltaChip.text;
        t.textContent = text;
        t.setAttribute('x', mid); t.setAttribute('y', y);
        let width;
        try { width = t.getBBox().width; } catch (e) { width = 0; }
        if (!width) width = text.length * 6;
        const r = this._deltaChip.rect;
        r.setAttribute('x', mid - width / 2 - M.chipPadX); r.setAttribute('y', y - M.chipH / 2); r.setAttribute('width', width + 2 * M.chipPadX);
        if (hi - lo > a * 3) {
            this._shaft.setAttribute('x1', lo + a); this._shaft.setAttribute('x2', hi - a); this._shaft.setAttribute('y1', y); this._shaft.setAttribute('y2', y);
            this._arrowL.setAttribute('points', `${lo},${y} ${lo + a},${y - a / 2} ${lo + a},${y + a / 2}`);
            this._arrowR.setAttribute('points', `${hi},${y} ${hi - a},${y - a / 2} ${hi - a},${y + a / 2}`);
        } else {
            this._shaft.setAttribute('x1', lo); this._shaft.setAttribute('x2', hi); this._shaft.setAttribute('y1', y); this._shaft.setAttribute('y2', y);
            this._arrowL.setAttribute('points', ''); this._arrowR.setAttribute('points', '');
        }
    }
}
