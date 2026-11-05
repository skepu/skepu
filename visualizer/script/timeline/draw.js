// timeline/draw.js — the model of the timeline (model.js) as SVG markup, for a window of the x axis. Pure: strings in, strings out.
//
// A window is drawn in its own coordinates (x = 0 at its left edge), so that the numbers stay small at any zoom; the content is far wider than the window.
'use strict';

import { forEachItemIn, tooltip } from './model.js';
import { METRICS as M, LIGHT_OPACITY, MARKERS } from './style.js';
import { ticks, niceStep } from './axis.js';
import { parseColor, contrast } from '../source_highlight.js';

export function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const num = v => +v.toFixed(2);

/**
 * How the time maps to x, for content of the width `contentW` (px). The bar area is what is left of it after the label column and the end padding.
 * x(t) is the x of the time `t` in the content; timeAt(x) is its inverse, in µs after the start of the trace.
 */
export function makeView(model, contentW) {
    const barW = Math.max(1, contentW - M.labelW - M.endPad);
    const pxPerUs = barW / model.range.span;
    return {
        contentW, barW, pxPerUs, minT: model.range.minT,
        endX: M.labelW + barW,
        x: t => M.labelW + (t - model.range.minT) * pxPerUs,
        timeAt: x => (x - M.labelW) / pxPerUs,
    };
}

/** The zoom (1: the trace fits the pane) at which `dt` µs of the trace fill the plot of a pane that is `viewportW` px wide. */
export function scaleToFit(model, viewportW, dt) {
    return ((viewportW - M.labelW) * model.range.span / dt + M.labelW + M.endPad) / viewportW;
}

/** The ticks in the window, and those just outside whose text reaches into it. */
function windowTicks(view, win) {
    const margin = M.tickPx / 2;
    return ticks(view.timeAt(win.x0 - margin), Math.min(view.timeAt(win.x1 + margin), view.timeAt(view.endX)), view.pxPerUs, M.tickPx);
}

function rowLabel(row, pal) {
    const y = row.y + row.h / 2 + M.textDy;
    if (row.kind === 'rank') return `<text x="6" y="${y}" fill="${pal.label}" font-weight="bold">${esc(row.label)}</text>`;
    const arrow = row.toggle ? (row.toggle === 'open' ? '▾ ' : '▸ ') : '';
    const attrs = row.toggle ? ' data-comm-toggle="1" class="tl-toggle"' : '';
    const title = row.toggle ? '<title>Show what every rank did in the collectives</title>' : '';
    return `<text x="${M.labelW - 6}" y="${y}" text-anchor="end" fill="${pal.label}"${attrs}>${title}${arrow}${esc(row.label)}</text>`;
}

/** The label column; opaque, so that it hides what scrolls under it. */
export function drawLabels(model, pal) {
    const out = [];
    for (const row of model.rows) {
        out.push(`<rect x="0" y="${row.y}" width="${M.labelW}" height="${row.h}" fill="${pal[row.shade]}"/>`);
        out.push(rowLabel(row, pal));
    }
    out.push(`<line x1="${M.labelW - 0.5}" y1="0" x2="${M.labelW - 0.5}" y2="${model.rowsHeight}" stroke="${pal.separator}"/>`);
    return out.join('');
}

function fillOf(item, pal) {
    switch (item.style) {
        case 'region': return pal.regionFill(item.depth);
        case 'build': return pal.buildFill;
        case 'build-failed': return pal.buildFailed;
        case 'comm': return pal.comm;
        default: return item.fill || '#778';
    }
}
const strokeOf = (item, pal) => item.style === 'region' ? pal.regionStroke : item.style === 'build' || item.style === 'build-failed' ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0.2)';
const WHITE = { r: 255, g: 255, b: 255 }, BLACK = { r: 0, g: 0, b: 0 };

/** The label colour of an item; on a bar with a colour from the data (call, transfer), white or black, whichever has more contrast. */
export function labelColor(item, pal) {
    if (item.style === 'region') return pal.regionText;
    if (item.style.startsWith('build')) return pal.buildText;
    if (item.lightText) return pal.lightBarText;
    const fill = item.fill && parseColor(item.fill);
    return fill && contrast(fill, BLACK) > contrast(fill, WHITE) ? pal.barTextDark : pal.barText;
}

function starPoints(cx, cy, outer, inner) {
    const pts = [];
    for (let i = 0; i < 10; i++) {
        const a = i * Math.PI / 5 - Math.PI / 2, r = i % 2 === 0 ? outer : inner;
        pts.push(`${num(cx + Math.cos(a) * r)},${num(cy + Math.sin(a) * r)}`);
    }
    return pts.join(' ');
}
const diamondPoints = (cx, cy, r) => `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;

function drawItem(item, model, view, win, pal, selectedId, out) {
    const row = model.rows[item.row], opacity = item.hidden ? ' opacity="0.3"' : '';
    const selected = item.id === selectedId ? ' gantt-selected' : '';
    const title = `<title>${esc(tooltip(item, model))}</title>`;
    if (item.shape === 'marker') {
        const cx = view.x(item.t0) - win.x0, cy = row.y + row.h / 2, m = MARKERS[item.style] ?? MARKERS.other;
        const pts = m.shape === 'star' ? starPoints(cx, cy, M.markerR, M.markerR * 0.42) : diamondPoints(num(cx), cy, M.markerR);
        out.push(`<polygon class="gantt-bar${selected}"${opacity} data-i="${item.idx}" points="${pts}" fill="${m.fill}" stroke="${m.stroke}" stroke-width="${m.strokeW}">${title}</polygon>`);
        return;
    }
    const y = row.y + M.barPad, h = row.h - 2 * M.barPad, fill = fillOf(item, pal), stroke = strokeOf(item, pal);
    const stroke05 = ` stroke="${stroke}" stroke-width="0.5"`;
    const clickable = item.style === 'comm' ? ' tl-click' : '';
    for (const s of item.spans) {
        const x1 = view.x(s.t0) - win.x0, w = Math.max(M.minBarW, view.x(s.t1) - view.x(s.t0));
        const light = s.light ? ` fill-opacity="${LIGHT_OPACITY}"` : '';
        out.push(`<rect class="gantt-bar${selected}${clickable}"${opacity} data-i="${item.idx}" x="${num(x1)}" y="${y}" width="${num(w)}" height="${h}" rx="${M.barRadius}" fill="${fill}"${light}${stroke05}>${title}</rect>`);
    }
    const x1 = view.x(item.t0) - win.x0, w = view.x(item.t1) - view.x(item.t0);
    if (w > M.minLabelW && item.label)
        out.push(`<svg data-li="${item.idx}" x="${num(x1 + 2)}" y="${y}" width="${num(w - 4)}" height="${h}" class="pointer-events-none"><text x="${M.labelPadX - 2}" y="${h / 2 + M.textDy}" fill="${labelColor(item, pal)}"${opacity}>${esc(item.label)}</text></svg>`);
}

/**
 * The items of the window `win` ({ x0, x1 }, x of the content) as the inside of an svg of width x1 - x0. Not here, because they do not depend on the
 * scroll: the rows, the grid, the end of the trace (layers of the page). A bar thinner than a pixel that starts inside the previous one of its row is left out.
 * The label of a bar is at its start (`data-li`); the view pins it to the left edge of the plot when the start is out of sight.
 */
export function drawPlot(model, view, win, pal, selectedId) {
    const out = [];
    const margin = M.markerR + 2, tLo = view.timeAt(win.x0 - margin), tHi = view.timeAt(win.x1 + margin);
    const loT = tLo + view.minT, hiT = tHi + view.minT;
    model.rows.forEach((row, r) => {
        let drawnTo = -Infinity;
        forEachItemIn(model, r, loT, hiT, item => {
            const x1 = view.x(item.t0), x2 = view.x(item.t1);
            if (item.shape === 'bar' && x2 - x1 < 1 && x1 < drawnTo) return;
            drawnTo = Math.max(drawnTo, x1 + (item.shape === 'bar' ? M.minBarW : 0));
            drawItem(item, model, view, win, pal, selectedId, out);
        });
    });
    return out.join('');
}

/** The tick times in the window; the marks are drawn by the page. */
export function drawAxis(model, view, win, pal) {
    const out = [];
    for (const t of windowTicks(view, win)) {
        const x = num(view.x(view.minT + t.t) - win.x0 + 0.5);   // t.t is after the start of the trace, x() takes the time of the data
        out.push(`<text x="${x}" y="${M.tickTextY}" text-anchor="${t.t === 0 ? 'start' : 'middle'}" fill="${pal.tickText}">${esc(t.label)}</text>`);
    }
    return out.join('');
}

/** The tick distance in px, which the grid and the marks repeat. */
export function tickStepPx(view) {
    return niceStep(M.tickPx / view.pxPerUs) * view.pxPerUs;
}
