// editor.js — the page of the designer: a graph editor for a design (model.js), and the program of it (codegen.js) beside it.
// The graph is drawn with the style of SkeVU (script/graph_style.js), the program is shown by the source view of SkeVU
// (script/SourceCodeView.js: tabs, line numbers, badges, code paths), and the legend, the menus, the hover and the resizers are SkeVU's.
// All state is the design (state.js); everything else is made from it again when it changes. The pieces:
//   state.js      the design and what is made from it, helpers, `changed`
//   graphview.js  the graph: its nodes and edges, style, legend, hover, selection
//   codeview.js   the programs and the trace, shown by the source view
//   status.js     the problems and the statistics
//   inspector.js  the properties of what is selected
//   actions.js    what can be done: add, connect, group, duplicate, delete, arrange; the modes and the keys
//   menus.js      the menus that are the designer's own

import { S, $, stored, store, LAYOUT, changed, history } from './state.js';
import { initGraph } from './graphview.js';
import { initCode } from './codeview.js';
import { initActions, setMode } from './actions.js';
import { initMenus } from './menus.js';
import './status.js';                 // (these register what they do when the design changes: `hooks`)
import './inspector.js';
import { installMenus } from '../script/menu.js';
import { attachResizer } from '../script/resizer.js';

/** The gutter between the graph and the right pane, and the resizer between the code and the properties (the layout of SkeVU). */
function splitters() {
    const right = $('#right_pane'), info = $('#info_window');
    const layout = stored(LAYOUT, {});
    const remember = () => { layout.width = right.offsetWidth; layout.info = info.offsetHeight; store(LAYOUT, layout); };
    attachResizer($('#split-gutter'), {
        axis: 'x', min: 300, max: () => innerWidth - 300,
        size: () => right.offsetWidth, setSize: px => { right.style.flex = `0 0 ${px}px`; }, onEnd: remember,
    });
    attachResizer($('#info-resizer'), {
        axis: 'y', min: 60, max: () => right.offsetHeight - 120,
        size: () => info.offsetHeight, setSize: px => { info.style.flex = `0 0 ${px}px`; },
        onStart: () => info.classList.add('resizing'), onEnd: () => { info.classList.remove('resizing'); remember(); },
    });
    right.style.flex = `0 0 ${Math.round(Math.max(300, Math.min(layout.width ?? 560, innerWidth - 300)))}px`;
    info.style.flex = `0 0 ${Math.round(Math.max(60, layout.info ?? innerHeight * 0.4))}px`;
}

function start() {
    installMenus();
    initGraph();
    initCode();
    initActions();
    initMenus();
    splitters();
    setMode('explore');
    changed({ structure: true });
    history.reset(JSON.stringify(S.design));              // (what the page starts with is not a step: nothing to undo yet)
}

start();

// for the tests and for looking at the state from the console
window.designer = { get design() { return S.design; }, get result() { return S.result; }, get cy() { return S.cy; }, get programs() { return S.programs; }, get source() { return S.source; } };
