// status.js — what the designer says of the design: the problems (above the code) and the statistics (the ⓘ menu).
import { objectsOf, PATTERNS } from './model.js';
import { $, S, el, hooks } from './state.js';
import { selectById } from './actions.js';

/** The statistics of the design, in the ⓘ menu of the menu bar (as the statistics of a trace are in SkeVU). */
export function renderStatus() {
    const set = (id, value) => { $(`#stat-${id}`).textContent = String(value); };
    const prng = S.design.containers.filter(c => c.kind === 'PRNG').length, externals = S.design.skeletons.filter(s => PATTERNS[s.pattern]?.host).length;
    const nodes = S.design.containers.length - prng, objects = [...objectsOf(S.design).values()].filter(m => m[0].kind !== 'PRNG').length;
    set('data', nodes === objects ? nodes : `${objects} (${nodes} nodes: updates are the same data)`);
    set('streams', prng);
    set('skeletons', S.design.skeletons.length - externals);
    set('externals', externals);
    set('loops', S.design.loops.filter(l => !l.scope).length);
    set('regions', S.design.loops.filter(l => l.scope).length);
    set('edges', S.design.edges.length);
    set('problems', S.result.errors.length);
    set('warnings', S.result.warnings.length);
    set('events', S.programs?.trace?.events.length ?? '—');
}

export function renderProblems() {
    const box = $('#problems'), list = $('#problem-list');
    const all = [...S.result.errors.map(p => ({ ...p, error: true })), ...S.result.warnings];
    box.hidden = !all.length;
    box.classList.toggle('has-errors', S.result.errors.length > 0);
    $('summary', box).textContent = S.result.errors.length
        ? `${S.result.errors.length} problem${S.result.errors.length === 1 ? '' : 's'} in the design${S.lastGood ? ': the program is the last one that could be made' : ''}`
        : `${all.length} warning${all.length === 1 ? '' : 's'}`;
    list.replaceChildren(...all.map(p => {
        const row = el('div', { className: 'problem' + (p.error ? ' error' : ''), textContent: p.msg });
        row.onclick = () => selectById(p.id, 'code');
        return row;
    }));
}

hooks.renderProblems = renderProblems;
hooks.renderStatus = renderStatus;
