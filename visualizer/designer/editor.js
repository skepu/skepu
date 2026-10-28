// editor.js — the designer page: a graph editor for a design (model.js) beside its program (codegen.js).
// The graph uses the style of SkeVU (script/graph_style.js), the code the source view of SkeVU (script/SourceCodeView.js); the legend, menus, hover and
// resizers are SkeVU's. The design is the only state (state.js); everything else is derived from it on change. Modules:
//   state.js      the design, derived data, helpers, `changed`
//   graphview.js  graph nodes, edges, style, legend, hover, selection
//   codeview.js   programs and trace in the source view
//   status.js     problems and statistics
//   inspector.js  properties of the selection
//   actions.js    editing actions, modes, keys
//   menus.js      the designer's own menus

import { S, $, stored, store, LAYOUT, changed, history } from './state.js';
import { initGraph } from './graphview.js';
import { initCode } from './codeview.js';
import { initActions, setMode } from './actions.js';
import { initMenus } from './menus.js';
import './status.js';                 // register their `hooks`
import './inspector.js';
import { installMenus } from '../script/menu.js';
import { attachResizer } from '../script/resizer.js';

/** The resizers between the graph and the right pane, and between the code and the properties. */
function splitters() {
    const right = $('#right_pane'), info = $('#info_window');
    const layout = stored(LAYOUT, {});
    const remember = () => { layout.width = right.offsetWidth; layout.info = S.infoHeight = info.offsetHeight; store(LAYOUT, layout); };
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
    S.infoHeight = Math.round(Math.max(60, layout.info ?? innerHeight * 0.4));          // the pane opens to this height with a selection (inspector.js)
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
    history.reset(JSON.stringify(S.design));              // the initial design is not an undo step
}

start();

// state handles for tests and the console
window.designer = { get design() { return S.design; }, get result() { return S.result; }, get cy() { return S.cy; }, get programs() { return S.programs; }, get source() { return S.source; }, get legend() { return S.legend; } };
