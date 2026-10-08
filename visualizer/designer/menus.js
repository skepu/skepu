// menus.js — the menus that are the designer's own: Design (new, examples, open, save), Add, View (colours, code theme, dark mode), Generate and Code.
import { PATTERNS, emptyDesign, sampleDesign, allSkeletonsDesign } from './model.js';
import { randomDesign } from './random.js';
import { writeAppearance, showDarkMode, buildThemeMenu, buildSpectrumMenu, markChoice, HLJS_THEME_DEFAULT_LIGHT, HLJS_THEME_DEFAULT_DARK } from '../script/appearance.js';
import { $, CODE, S, changed, codeOptions, download, el, normalize, say, setChecked, store, stored } from './state.js';
import { darkNow, provider, renderLegend, settings } from './graphview.js';
import { renderCode } from './codeview.js';
import { text } from './inspector.js';
import { arrange, replace, setMode } from './actions.js';

export function addMenu() {
    // the types that can be added: the one that is checked is what a click on the canvas adds (the Add mode)
    const item = (type, text, title) => {
        const b = el('button', { className: 'menu-radio', title }, el('span', { className: 'menu-icon' }), text);
        b.dataset.add = type;
        return b;
    };
    const menu = $('#add-menu');
    // the skeletons; External and I/O (host blocks, drawn as in SkeVU) and Region (a group of calls) are sections of their own
    for (const p of Object.keys(PATTERNS).filter(p => !PATTERNS[p].host)) menu.append(item(p, p, PATTERNS[p].doc));
    menu.append(el('hr', { className: 'menu-sep' }), item('External', 'External', PATTERNS.External.doc), item('IO', 'I/O', PATTERNS.IO.doc),
                el('hr', { className: 'menu-sep' }), item('Region', 'Region', 'A group of calls (a trace region): click in a region to add to it, or connect it to the calls that are in it'));
    for (const b of document.querySelectorAll('#add-menu [data-add]')) b.onclick = () => setMode(S.adding === b.dataset.add ? 'explore' : 'add', b.dataset.add);
}

// ── Generate: a random design, as the settings of the Generate menu bias it ──

export const GENERATE = 'skevu-designer-generate';

export const generating = { size: 'medium', arity: 'small', shape: 'balanced', seed: '', ...stored(GENERATE, {}) };

export function generateDesign() {
    if (S.design.containers.length + S.design.skeletons.length + S.design.loops.length && !confirm('Replace the design with a random one?')) return;
    const made = randomDesign(generating);
    replace(made.design);
    arrange();                                              // (the design has places that are only rough)
    say(`Generated: ${generating.size}, arity ${generating.arity}, ${generating.shape}, seed ${made.seed}`);
}

export function codeMenu() {
    const marks = () => { for (const b of document.querySelectorAll('[data-code]')) setChecked(b, codeOptions[b.dataset.code] === b.dataset.value); };
    for (const b of document.querySelectorAll('[data-code]')) b.onclick = () => { codeOptions[b.dataset.code] = b.dataset.value; store(CODE, codeOptions); marks(); renderCode(); };
    marks();
}

export function generateMenu() {
    const marks = () => { for (const b of document.querySelectorAll('[data-gen]')) setChecked(b, generating[b.dataset.gen] === b.dataset.value); };
    for (const b of document.querySelectorAll('[data-gen]')) b.onclick = () => { generating[b.dataset.gen] = b.dataset.value; store(GENERATE, generating); marks(); };
    const seed = $('#gen-seed');
    seed.value = generating.seed ?? '';
    seed.oninput = () => { generating.seed = seed.value; store(GENERATE, generating); };
    seed.onclick = e => e.stopPropagation();                // (a click in the text entry does not close the menu)
    seed.onkeydown = e => { if (e.key === 'Enter') { document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); generateDesign(); } e.stopPropagation(); };
    marks();
    $('#generate').onclick = generateDesign;
}

/** The menus of the designer: their items do what they say. */
export function initMenus() {
    addMenu();
    const name = $('#design-name');
    name.value = S.design.name;
    name.onchange = e => { S.design.name = e.target.value || 'design'; changed(); };
    name.onclick = e => e.stopPropagation();                // (a click in the text entry does not close the menu)
    name.onkeydown = e => { if (e.key === 'Enter') { name.blur(); document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); } e.stopPropagation(); };

    // the Design menu
    $('#new').onclick = () => { if (confirm('Start an empty design?')) replace(emptyDesign()); };
    $('#sample').onclick = () => { if (confirm('Replace the design with the example?')) replace(sampleDesign()); };
    $('#all-skeletons').onclick = () => { if (confirm('Replace the design with the one that has every skeleton?')) replace(allSkeletonsDesign()); };
    $('#save').onclick = () => download(`${S.design.name}.json`, JSON.stringify(S.design, null, 2), 'application/json');
    $('#load').onclick = () => $('#load-file').click();
    $('#load-file').onchange = async e => {
        const file = e.target.files[0];
        if (!file) return;
        try { replace(normalize(JSON.parse(await file.text()))); } catch { say('That is not a design.'); }
        e.target.value = '';
    };

    // the View menu
    const colors = () => document.querySelectorAll('[data-color]').forEach(b => setChecked(b, b.dataset.color === settings['node-color-call']));
    document.querySelectorAll('[data-color]').forEach(b => {
        b.onclick = () => { settings['node-color-call'] = b.dataset.color; store('skevu-designer-color', b.dataset.color); colors(); S.cy.style().update(); renderLegend(); };
    });
    colors();
    const legendOn = stored('skevu-designer-legend', true);
    const showLegend = on => { $('#legend').hidden = !on; setChecked($('#menu-legend'), on); store('skevu-designer-legend', on); S.cy.resize(); };
    $('#menu-legend').onclick = () => showLegend($('#legend').hidden);
    showLegend(legendOn);
    const paths = stored('skevu-designer-paths', true);
    S.source.setCodePaths(paths);
    setChecked($('#menu-paths'), paths);
    $('#menu-paths').onclick = () => { const on = !stored('skevu-designer-paths', true); store('skevu-designer-paths', on); setChecked($('#menu-paths'), on); S.source.setCodePaths(on); };
    setChecked($('#menu-dark'), darkNow());
    $('#menu-dark').onclick = () => {
        const on = !darkNow();
        showDarkMode(on);
        writeAppearance({ darkMode: on });
        S.source.setTheme(on ? HLJS_THEME_DEFAULT_DARK : HLJS_THEME_DEFAULT_LIGHT);        // (the code theme of the mode, as SkeVU does)
        setChecked($('#menu-dark'), on);
        S.cy.style().update();
        renderLegend();
    };
    // the code theme and the colour spectrum: the menus of SkeVU
    buildThemeMenu($('#hljs-theme-menu'), theme => S.source.setTheme(theme));
    S.source.syncThemeIcons();
    buildSpectrumMenu($('#spectrum-menu'), name => {
        provider.colorSpectrum = name;
        writeAppearance({ colorSpectrum: name });
        markChoice('.spectrum-radio', 'spectrum', name);
        S.cy.style().update();
        renderLegend();
    });
    markChoice('.spectrum-radio', 'spectrum', provider.colorSpectrum);

    generateMenu();
    codeMenu();
}
