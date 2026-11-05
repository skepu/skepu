// timeline/style.js — sizes and colours. Pure.
'use strict';

/** Sizes in px. The label column is sticky; the end padding follows the end of the trace. */
export const METRICS = {
    labelW:     72,
    endPad:     16,
    rowH:       22,   // least height; rows stretch with the pane
    maxRowH:    80,
    headerH:    14,   
    axisH:      18,   // ticks and cursor times
    barPad:     2,    
    barRadius:  2,
    minBarW:    2,    
    minLabelW:  24,   // no label on thinner bars
    labelPadX:  4,
    textDy:     4,    
    tickPx:     90,   
    tickLen:    3,
    tickTextY:  13,   
    markerR:    7,    
    chipH:      14,   
    chipY:      2,
    chipPadX:   4,
    deltaY:     9,    
    arrowHead:  5,
    lockHitPx:  4,    // click distance that counts as on the locked line
};

/** Opacity of the light part of a bar (waiting, communication). */
export const LIGHT_OPACITY = 0.38;

/** Colours of the chart that are not those of the data. `dark`: the palette of the dark mode. */
export function palette(dark) {
    const p = dark ? {
        rowA: '#1e1c2e', rowEven: '#1a1824', rowOdd: '#201e2c', rowRank: '#26233a', rowTransfer: '#221e2e', rowBuild: '#2e2a1e',
        rowComm: '#182529', rowEvent: '#1e1c28', beyond: 'rgba(0,0,0,0.28)',
        label: '#a09ab8', tickText: '#706880', grid: '#2a2838', separator: '#3a3848', axis: '#3a3848', tickMark: '#504860', end: '#504860',
        barText: 'rgba(255,255,255,0.9)', barTextDark: 'rgba(0,0,0,0.8)', lightBarText: 'rgba(255,255,255,0.9)', regionText: 'rgba(240,200,130,0.90)', buildText: 'rgba(0,0,0,0.8)',
        regionStroke: 'rgba(200,160,80,0.45)', buildFill: '#f0b44c', buildFailed: '#d9534f', comm: '#17a2b8',
        hoverStroke: '#504860', hoverChip: '#2a2838', hoverText: '#888090',
        lockStroke: '#8878a8', lockChip: '#22202a', lockChipStroke: '#604878', lockText: '#c0b8d8',
        delta: '#8878a8', deltaChip: '#22202a', deltaChipStroke: '#3a3848', deltaText: '#c0b8d8',
    } : {
        rowA: '#f0f7ff', rowEven: '#fff', rowOdd: '#f8f8f8', rowRank: '#e4eefb', rowTransfer: '#f8f4ff', rowBuild: '#fff9ef',
        rowComm: '#f0fafb', rowEvent: '#fff8f5', beyond: 'rgba(0,0,0,0.05)',
        label: '#555', tickText: '#777', grid: '#efefef', separator: '#ccc', axis: '#bbb', tickMark: '#aaa', end: '#bbb',
        barText: 'rgba(255,255,255,0.9)', barTextDark: 'rgba(0,0,0,0.8)', lightBarText: 'rgba(0,0,0,0.75)', regionText: 'rgba(0,70,160,0.80)', buildText: 'rgba(0,0,0,0.8)',
        regionStroke: 'rgba(70,155,220,0.5)', buildFill: '#f0b44c', buildFailed: '#d9534f', comm: '#17a2b8',
        hoverStroke: '#ccc', hoverChip: 'white', hoverText: '#aaa',
        lockStroke: '#888', lockChip: 'white', lockChipStroke: '#999', lockText: '#555',
        delta: '#888', deltaChip: 'white', deltaChipStroke: '#bbb', deltaText: '#555',
    };
    p.regionFill = depth => dark
        ? `rgba(${Math.min(185, 80 + depth * 25)},${Math.min(140, 60 + depth * 18)},${Math.min(70, 28 + depth * 10)},0.60)`
        : `rgba(${Math.max(20, 225 - depth * 34)},${Math.max(185, 240 - depth * 9)},255,0.55)`;
    return p;
}

/** Fill and stroke of the markers of the event row (allocation, deallocation, snapshot, other). */
export const MARKERS = {
    allocation:   { shape: 'star',    fill: 'white',   stroke: '#333',              strokeW: 1 },
    deallocation: { shape: 'star',    fill: 'black',   stroke: '#ddd',              strokeW: 1 },
    snapshot:     { shape: 'star',    fill: '#ffd43b', stroke: '#a67c00',           strokeW: 1 },
    other:        { shape: 'diamond', fill: 'purple',  stroke: 'rgba(0,0,0,0.25)',  strokeW: 0.5 },
};
