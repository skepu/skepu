// menus.js — the designer's menus: Design (new, examples, open, save), Add, View, Generate and Code.
import { PATTERNS, emptyDesign, sampleDesign, allSkeletonsDesign, poolingDesign } from './model.js';
import { randomDesign } from './random.js';
import { writeAppearance, showDarkMode, buildThemeMenu, buildSpectrumMenu, showSpectrumMenu, showChoiceMenu, markChoice, HLJS_THEME_DEFAULT_LIGHT, HLJS_THEME_DEFAULT_DARK } from '../script/appearance.js';
import { hideTooltip } from '../script/tooltip.js';
import { $, CODE, S, changed, codeOptions, download, el, hooks, kindOf, normalize, say, setChecked, store, stored } from './state.js';
import { darkNow, provider, renderLegend, settings, stylesheet } from './graphview.js';
import { renderCode } from './codeview.js';
import { text } from './inspector.js';
import { addNode, arrange, replace, selectById, setMode } from './actions.js';
import { showContextMenu } from '../script/context_menu.js';

export function addMenu() {
    // the checked type is what a canvas click adds (add mode)
    const item = (type, text, title) => {
        const b = el('button', { className: 'menu-radio', title }, el('span', { className: 'menu-icon' }), text);
        b.dataset.add = type;
        return b;
    };
    const menu = $('#add-menu');
    // skeletons, then External and I/O (host blocks), then Region
    for (const p of Object.keys(PATTERNS).filter(p => !PATTERNS[p].host)) menu.append(item(p, p, PATTERNS[p].doc));
    menu.append(el('hr', { className: 'menu-sep' }), item('External', 'External', PATTERNS.External.doc), item('IO', 'I/O', PATTERNS.IO.doc),
                el('hr', { className: 'menu-sep' }), item('Region', 'Region', 'Group of calls (trace region)'));
    for (const b of document.querySelectorAll('#add-menu [data-add]')) b.onclick = () => setMode(S.adding === b.dataset.add ? 'explore' : 'add', b.dataset.add);
}

// ── Generate ─────────────────────────────────────────────────────────────────

export const GENERATE = 'skevu-designer-generate';

export const generating = { size: 'medium', arity: 'small', shape: 'balanced', seed: '', ...stored(GENERATE, {}) };

export function generateDesign() {
    if (S.design.containers.length + S.design.skeletons.length + S.design.loops.length && !confirm('Replace the design with a random one?')) return;
    const made = randomDesign(generating);
    replace(made.design);
    arrange();                                              // generated positions are rough
    say(`Generated: ${generating.size}, arity ${generating.arity}, ${generating.shape}, seed ${made.seed}`);
}

export function codeMenu() {
    const marks = () => { for (const b of document.querySelectorAll('[data-code]')) setChecked(b, codeOptions[b.dataset.code] === b.dataset.value); };
    for (const b of document.querySelectorAll('[data-code]')) b.onclick = () => { codeOptions[b.dataset.code] = b.dataset.value; store(CODE, codeOptions); marks(); renderCode(); hooks.renderInspector(true); };
    marks();
}

export function generateMenu() {
    const marks = () => { for (const b of document.querySelectorAll('[data-gen]')) setChecked(b, generating[b.dataset.gen] === b.dataset.value); };
    for (const b of document.querySelectorAll('[data-gen]')) b.onclick = () => { generating[b.dataset.gen] = b.dataset.value; store(GENERATE, generating); marks(); };
    const seed = $('#gen-seed');
    seed.value = generating.seed ?? '';
    seed.oninput = () => { generating.seed = seed.value; store(GENERATE, generating); };
    seed.onclick = e => e.stopPropagation();                // a click in the text field keeps the menu open
    seed.onkeydown = e => { if (e.key === 'Enter') { document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); generateDesign(); } e.stopPropagation(); };
    marks();
    $('#generate').onclick = generateDesign;
}

/**
 * The graph context menus (opened on press). On a node: the Edit actions on the selection (the node is selected first if it is not). On the canvas:
 * the Add items, which add at the click position without changing the mode.
 */
export function contextMenus() {
    const label = b => [...b.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('');
    const key = b => b.querySelector('.menu-key')?.textContent;
    // an item mirrors its Edit-menu button (action and enabled state)
    const editItem = id => { const b = $(id); return { label: key(b) ? `${label(b)}  (${key(b)})` : label(b), disabled: b.disabled, onClick: () => b.click() }; };
    const SELECTION = ['#duplicate', '#group-nodes', '#group-loop', '#ungroup', '#delete'];
    const separator = () => el('hr', { className: 'menu-sep' });

    S.cy.on('cxttapstart', evt => {
        hideTooltip();
        const mouse = evt.originalEvent, t = evt.target;
        if (!mouse) return;
        if (t === S.cy) {
            const groups = [];                                        // the Add menu sections
            for (const b of document.querySelectorAll('#add-menu [data-add]')) {
                const section = groups.at(-1);
                if (!section || b.previousElementSibling?.tagName === 'HR') groups.push([b]); else section.push(b);
            }
            const items = groups.flatMap((g, i) => [...(i ? [separator()] : []), ...g.map(b => ({ label: label(b), title: b.title, onClick: () => addNode(b.dataset.add, evt.position) }))]);
            showContextMenu(mouse.clientX, mouse.clientY, items);
        } else if (t.isNode() && kindOf(t.id())) {
            if (!t.selected()) selectById(t.id());
            showContextMenu(mouse.clientX, mouse.clientY, SELECTION.map(editItem));
        }
    });
}

/** Binds the menus. */
export function initMenus() {
    addMenu();
    const name = $('#design-name');
    name.value = S.design.name;
    name.onchange = e => { S.design.name = e.target.value || 'design'; changed(); };
    name.onclick = e => e.stopPropagation();                // (a click in the text entry does not close the menu)
    name.onkeydown = e => { if (e.key === 'Enter') { name.blur(); document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); } e.stopPropagation(); };

    // Design menu
    $('#new').onclick = () => { if (confirm('Start an empty design?')) replace(emptyDesign()); };
    $('#sample').onclick = () => { if (confirm('Replace the design with the Smoothing example?')) replace(sampleDesign()); };
    $('#pooling').onclick = () => { if (confirm('Replace the design with the Pooling and reductions example?')) replace(poolingDesign()); };
    $('#all-skeletons').onclick = () => { if (confirm('Replace the design with the All skeletons example?')) replace(allSkeletonsDesign()); };
    $('#save').onclick = () => download(`${S.design.name}.json`, JSON.stringify(S.design, null, 2), 'application/json');
    $('#load').onclick = () => $('#load-file').click();
    $('#load-file').onchange = async e => {
        const file = e.target.files[0];
        if (!file) return;
        try { replace(normalize(JSON.parse(await file.text()))); } catch { say('Not a design file.'); }
        e.target.value = '';
    };

    // View menu
    const colors = () => document.querySelectorAll('[data-color]').forEach(b => setChecked(b, b.dataset.color === settings['node-color-call']));
    document.querySelectorAll('[data-color]').forEach(b => {
        b.onclick = () => { settings['node-color-call'] = b.dataset.color; store('skevu-designer-color', b.dataset.color); colors(); S.cy.style().update(); renderLegend(); };
    });
    colors();
    const legendOn = stored('skevu-designer-legend', true);
    const showLegend = on => { $('#legend').hidden = !on; setChecked($('#menu-legend'), on); store('skevu-designer-legend', on); S.cy.resize(); };
    $('#menu-legend').onclick = () => showLegend($('#legend').hidden);
    showLegend(legendOn);
    const labels = stored('skevu-designer-edge-labels', false);
    const showEdgeLabels = on => { settings['edge-labels'] = on; setChecked($('#menu-edge-labels'), on); store('skevu-designer-edge-labels', on); S.cy.style(stylesheet()); };
    showEdgeLabels(labels);
    $('#menu-edge-labels').onclick = () => showEdgeLabels(!settings['edge-labels']);
    const paths = stored('skevu-designer-paths', true);
    S.source.setCodePaths(paths);
    setChecked($('#menu-paths'), paths);
    $('#menu-paths').onclick = () => { const on = !stored('skevu-designer-paths', true); store('skevu-designer-paths', on); setChecked($('#menu-paths'), on); S.source.setCodePaths(on); };
    setChecked($('#menu-dark'), darkNow());
    $('#menu-dark').onclick = () => {
        const on = !darkNow();
        showDarkMode(on);
        writeAppearance({ darkMode: on });
        S.source.setTheme(on ? HLJS_THEME_DEFAULT_DARK : HLJS_THEME_DEFAULT_LIGHT);        // the code theme follows the mode
        setChecked($('#menu-dark'), on);
        S.cy.style().update();
        renderLegend();
    };
    // code theme and color spectrum menus
    buildThemeMenu($('#hljs-theme-menu'), theme => S.source.setTheme(theme));
    S.source.syncThemeIcons();
    const pickSpectrum = name => {
        provider.colorSpectrum = name;
        writeAppearance({ colorSpectrum: name });
        markChoice('.spectrum-radio', 'spectrum', name);
        S.cy.style().update();
        renderLegend();
    };
    buildSpectrumMenu($('#spectrum-menu'), pickSpectrum);
    hooks.colorByMenu = (x, y) => showChoiceMenu(x, y, document.querySelectorAll('[data-color]'), b => b.dataset.color === settings['node-color-call']);      // right click on the color bar
    hooks.spectrumMenu = (x, y) => showSpectrumMenu(x, y, provider.colorSpectrum, pickSpectrum);          // click on the color bar
    markChoice('.spectrum-radio', 'spectrum', provider.colorSpectrum);

    generateMenu();
    codeMenu();
    contextMenus();
}
