// codegen.js — the program of a lowered design (model.js lower()) as SkePU C++ and SkePY Python.
// Each emitter returns { text, lines, sites }: `lines[i]` has the design node ids of line i + 1 (for marking code and nodes), `sites` the lines where a
// skeleton is called or a loop is entered, as { type, id, line, count (runs), order }.
'use strict';
import { PATTERNS, readsOf, writesOf, overlapOf, poolOf, strideOf, stridesOf, dimsOf } from './model.js';

/**
 * Where containers are declared with `containerPlacement: 'first-use'`: before the first step that uses them, at the level where all uses are, outside loops
 * (a scope runs once and is entered). Random streams and unused data are declared at the start (`start`).
 * Returns { before: Map(step -> containers declared before it), start: [containers] }.
 */
export function containerPlacement(program) {
    const uses = new Map();                                    // object -> the step paths of the calls that use it (any update)
    walkCalls(program.steps, [], (call, path) => {
        for (const c of [...readsOf(call), ...writesOf(call)]) (uses.get(c.object) ?? uses.set(c.object, []).get(c.object)).push(path);
    });
    const before = new Map(), start = [];
    for (const c of program.containers) {
        const paths = uses.get(c.object);
        if (c.kind === 'PRNG' || !paths) { start.push(c); continue; }
        const step = stepBefore(program.steps, paths);
        (before.get(step) ?? before.set(step, []).get(step)).push(c);
    }
    return { before, start };
}

/** Calls fn(call, path) for each call, with the step path to it. */
function walkCalls(steps, path, fn) {
    for (const step of steps) {
        const here = [...path, step];
        if (step.type === 'call') fn(step.call, here); else walkCalls(step.steps, here, fn);
    }
}

/** The step before which something used by the calls at `paths` is declared: the first use, at the level shared by all, outside loops. */
function stepBefore(steps, paths) {
    let list = steps;
    for (let level = 0; ; level++) {
        const at = paths.map(p => list.indexOf(p[level]));
        const first = Math.min(...at), step = list[first];
        if (step.type === 'loop' && step.scope && at.every(i => i === first)) { list = step.steps; continue; }
        return step;
    }
}

/**
 * Where skeletons are declared with `skeletonPlacement: 'first-use'`: before their call, outside loops (as for containers).
 * Returns { before: Map(step -> calls whose skeletons are declared before it) }.
 */
export function skeletonPlacement(program) {
    const before = new Map();
    walkCalls(program.steps, [], (call, path) => {
        if (call.host) return;
        const step = stepBefore(program.steps, [path]);
        (before.get(step) ?? before.set(step, []).get(step)).push(call);
    });
    return { before };
}

/**
 * The layout of the program (`options.whitespace`): 'balanced' (default); 'readable' (blank lines between fragments of different nodes: skeleton
 * declarations, calls, regions; a call keeps its containers and setters); 'compact' (no generated comments; user comments stay).
 */
class Writer {
    constructor(whitespace = 'balanced') {
        this.out = []; this.ids = []; this.depth = 0; this.sites = []; this.times = 1;
        this.readable = whitespace === 'readable'; this.compact = whitespace === 'compact';
    }
    /** A generated comment (omitted in the compact form). */
    comment(text, ...ids) { if (!this.compact) this.line(text, ...ids); }
    /** A blank line between fragments (readable form), not after a blank line, a block opening or a comment. */
    gap() {
        const last = (this.out[this.out.length - 1] ?? '').trim();
        if (this.readable && last !== '' && !last.endsWith('{') && !last.endsWith(':') && !last.startsWith('//') && !last.startsWith('#')) this.line();
    }
    /** Records that the next line is a call ('skeleton_call') or a loop ('region'). */
    site(type, id) { this.sites.push({ type, id, line: this.out.length + 1, count: this.times, order: this.sites.length }); }
    line(text = '', ...ids) {
        const ind = text === '' ? '' : this.indent.repeat(this.depth);
        for (const l of String(text).split('\n')) { this.out.push(l === '' ? '' : ind + l); this.ids.push(ids.filter(Boolean)); }
    }
    get indent() { return this.style === 'py' ? '    ' : '\t'; }
    result() { return { text: this.out.join('\n') + '\n', lines: this.ids, sites: this.sites }; }
}

const pyNumber = (value, type) => {
    const v = String(value ?? '').trim();
    if (v === '') return type === 'int' ? '0' : '0.0';
    return v.replace(/^(-?[0-9.]+(?:e[-+]?[0-9]+)?)[fFlL]+$/, '$1');
};
const cppNumber = (value, type) => {
    const v = String(value ?? '').trim();
    if (v === '') return type === 'float' ? '0.0f' : type === 'double' ? '0.0' : '0';
    return v;
};

const cppParam = (p) => {
    switch (p.role) {
        case 'index':   return `skepu::${p.type} ${p.name}`;
        case 'random':  return 'skepu::Random<1> &random';
        case 'region':  return `skepu::${p.type}<${p.elementType}> ${p.name}`;
        case 'proxy':   return `skepu::${{ Matrix: 'Mat', Tensor3: 'Ten3', Tensor4: 'Ten4' }[p.container.kind] ?? 'Vec'}<${p.type}> ${p.name}`;
        default:        return `${p.type} ${p.name}`;
    }
};

const dims = c => c.size;

/** A stub without comments (compact form). */
const bare = (text, py) => text.split('\n').map(l => py ? l.replace(/\s*#.*$/, '') : l).filter(l => l.trim() && !l.trim().startsWith(py ? '#' : '//'));

/** The lines of a user function: the user's body, or the stub with a TODO. */
const cppBody = (f, compact) => {
    const user = (f.bodies.cpp ?? '').trim();
    return user ? user.split('\n') : compact ? bare(f.stub.cpp, false)
        : ['// TODO: ' + f.call.skeleton.label + (f.suffix === '_reduce' ? ' (reduction)' : f.suffix === '_map' ? ' (map)' : ''), ...f.stub.cpp.split('\n')];
};

/** Whether a user function is a lambda in its skeleton's declaration: the skeleton's setting, else the program option. */
export const isLambda = (f, options) => !f.host && (f.call.skeleton.lambda?.[f.suffix] ?? options.userFunctions === 'lambdas');

/** The skeleton type as the precompiler names it (Reduce1D, Reduce2D, MapOverlap2D, MapPool3D, Map, …); `call` is the lowered call, if any. */
export const skeletonTypeName = (s, call) => s.pattern === 'Reduce' ? ((call ? call.second : s.params.second) ? 'Reduce2D' : 'Reduce1D')
    : s.pattern === 'MapOverlap' || s.pattern === 'MapPool' ? `${s.pattern}${dimsOf(s)}D` : s.pattern;

/** Whether a skeleton is created at its call: its own setting, else the program option (never for host blocks). */
export const isAnonymous = (s, options) => !PATTERNS[s.pattern]?.host && (s.anonymous ?? options.skeletonInstancing === 'anonymous');

/** Whether a text property is set (not empty). */
const given = v => String(v ?? '').trim() !== '';
/** Whether a numeric text is set and not zero (the default value). */
const nonzero = v => given(v) && parseFloat(String(v).trim()) !== 0;
/** Whether a Map or MapReduce has a stride other than 1. */
const strided = call => stridesOf(call.skeleton, call.pattern === 'Map' ? 1 + call.elwise.length : call.elwise.length).some(x => x !== 1);

/** The trace name of an anonymous skeleton: the type and the call site. */
export const anonymousLabel = (s, call, file, line) => `${skeletonTypeName(s, call)}@${file}${line ? `:${line}` : ''}`;

/**
 * The program as SkePU C++ (for the precompiler).
 * `options.userFunctions`: 'functions' (default: named functions before main) or 'lambdas' (in the skeleton declaration), unless the skeleton sets it.
 * `options.skeletonInstancing`: 'named' (default) or 'anonymous' (created at the call with chained settings), unless the skeleton sets it.
 * `options.containerPlacement`: 'grouped' (default) or 'first-use'.
 */
export function emitCpp(program, options = {}) {
    const lambdaOf = f => isLambda(f, options);
    const w = new Writer(options.whitespace);
    w.style = 'cpp';
    w.comment(`// ${program.name}: generated by the SkeVU designer. Function bodies are stubs.`);
    w.line('#include <iostream>');
    w.line('#include <skepu>');
    w.line('#include <skepu-lib/io.hpp>');
    w.line();
    for (const f of program.functions.filter(f => !f.host && !lambdaOf(f))) {
        w.line(`${f.returns} ${f.name}(${f.params.map(cppParam).join(', ')})`, f.call.skeleton.id);
        w.line('{', f.call.skeleton.id);
        w.depth++;
        for (const l of cppBody(f, w.compact)) w.line(l, f.call.skeleton.id);
        w.depth--;
        w.line('}', f.call.skeleton.id);
        w.line();
    }
    w.line('int main(int argc, char *argv[])');
    w.line('{');
    w.depth++;
    w.line('if (argc < 2)');
    w.line('{');
    w.line('\tskepu::io::cout << "Usage: " << argv[0] << " backend\\n";');
    w.line('\texit(1);');
    w.line('}');
    w.line('auto spec = skepu::BackendSpec{argv[1]};');
    w.line('skepu::setGlobalBackendSpec(spec);');
    w.line();
    const declareContainer = c => {
        if (c.kind === 'PRNG') {                       // seed (if any), then the name
            const seed = String(c.init ?? '').trim(), label = c.label.replace(/"/g, '\\"');
            const args = [...(seed ? [seed] : []), ...(c.label === c.name ? [] : [`"${label}"`])];
            w.line(args.length ? `skepu::PRNG ${c.name}(${args.join(', ')});` : `skepu::PRNG ${c.name};`, ...c.nodes);
            return;
        }
        // the label (the name of the symbol, by default) and the initial value (zero) are left out when they are the defaults
        const zero = /^[-+]?0*(\.0*)?([eE][-+]?\d+)?[fFlL]?$/.test(String(c.init ?? '').trim());
        const tensor = c.kind === 'Tensor3' || c.kind === 'Tensor4';
        const args = [...dims(c)];
        if (c.label !== c.name || (tensor && !zero)) args.push(`"${c.label.replace(/"/g, '\\"')}"`);
        if (!zero) args.push(cppNumber(c.init, c.type));
        w.line(`skepu::${c.kind}<${c.type}> ${c.name}(${args.join(', ')});`, ...c.nodes);
    };
    const placement = options.containerPlacement === 'first-use' ? containerPlacement(program) : null;
    const atStart = placement ? placement.start : program.containers;
    if (atStart.length) w.comment(placement ? '// random streams and unused data' : '// containers');
    atStart.forEach(declareContainer);
    if (atStart.length) w.line();
    /** Writes `prefix`, the user functions (by name or as lambdas), then `)` and `tail`; `chain` lines follow, indented. */
    const declare = (prefix, fns, id, tail = ';', chain = []) => {
        let pending = prefix;
        fns.forEach((f, i) => {
            const last = i === fns.length - 1;
            if (!lambdaOf(f)) {
                pending += f.name + (last ? `)${tail}` : ', ');
                if (last) { w.line(pending, id); chainLines(); }
                return;
            }
            w.line(`${pending}[](${f.params.map(cppParam).join(', ')}) -> ${f.returns}`, id);
            pending = '';
            w.line('{', id);
            w.depth++;
            for (const l of cppBody(f, w.compact)) w.line(l, id);
            w.depth--;
            w.line(last ? `})${tail}` : '},', id);
            if (last) chainLines();
        });
        function chainLines() {            // settings and call of an anonymous skeleton, one per line
            w.depth++;
            for (const [text, ...ids] of chain) w.line(text, id, ...ids);
            w.depth--;
        }
    };
    /** The skeleton constructor (`head`, up to the functions) and its settings as [text, ...node ids]. */
    const skeletonParts = call => {
        const s = call.skeleton, id = s.id, settings = [];
        let head = '';
        switch (s.pattern) {
            case 'Map':
                head = `skepu::Map<${call.elwise.length}>(`;
                if (strided(call)) settings.push([`setStride(${stridesOf(s, 1 + call.elwise.length).join(', ')})`]);
                break;
            case 'MapReduce':
                head = `skepu::MapReduce<${call.elwise.length}>(`;
                if (strided(call)) settings.push([`setStride(${stridesOf(s, call.elwise.length).join(', ')})`]);
                if (given(s.params.start)) settings.push([`setStartValue(${cppNumber(s.params.start, call.elementType)})`]);
                break;
            case 'Reduce':
                head = 'skepu::Reduce(';
                if ((call.partial || call.second) && s.params.mode === 'ColWise') settings.push(['setReduceMode(skepu::ReduceMode::ColWise)']);
                if (given(s.params.start)) settings.push([`setStartValue(${cppNumber(s.params.start, call.elementType)})`]);
                break;
            case 'Scan':
                head = 'skepu::Scan(';
                if (s.params.mode === 'Exclusive') settings.push(['setScanMode(skepu::ScanMode::Exclusive)']);
                if (given(s.params.start)) settings.push([`setStartValue(${s.params.start})`]);
                break;
            case 'MapOverlap':
                head = 'skepu::MapOverlap(';
                if (overlapOf(s).some(o => o !== 1)) settings.push([`setOverlap(${overlapOf(s).join(', ')})`]);
                if (s.params.edge !== 'Duplicate') settings.push([`setEdgeMode(skepu::Edge::${s.params.edge})`]);
                if (s.params.edge === 'Pad' && nonzero(s.params.pad)) settings.push([`setPad(${cppNumber(s.params.pad, call.elementType)})`]);
                if (s.params.update && s.params.update !== 'Normal') settings.push([`setUpdateMode(skepu::UpdateMode::${s.params.update})`]);
                break;
            case 'MapPool': {
                head = 'skepu::MapPool(';
                const pool = poolOf(s), stride = strideOf(s);
                if (pool.join() === stride.join()) settings.push([`setPoolSizeAndStride(${pool.join(', ')})`]);
                else settings.push([`setPoolSize(${pool.join(', ')})`], [`setStride(${stride.join(', ')})`]);
                break;
            }
            case 'MapPairs':  head = 'skepu::MapPairs<1, 1>('; break;
            case 'MapPairsReduce':
                head = 'skepu::MapPairsReduce<1, 1>(';
                if (s.params.mode === 'ColWise') settings.push(['setReduceMode(skepu::ReduceMode::ColWise)']);
                if (given(s.params.start)) settings.push([`setStartValue(${cppNumber(s.params.start, call.elementType)})`]);
                break;
        }
        if (call.random) settings.push([`setPRNG(${call.random.name})`, call.random.id]);
        if (s.backend) settings.push([`setBackend(skepu::BackendSpec{"${s.backend}"})`]);
        return { head, settings };
    };
    const declareSkeleton = call => {
        const id = call.skeleton.id, fns = program.functions.filter(f => f.call === call);
        const { head, settings } = skeletonParts(call);
        declare(`auto ${call.name} = ${head}`, fns, id);
        for (const [text, ...ids] of settings) w.line(`${call.name}.${text};`, id, ...ids);
    };
    const skeletons = options.skeletonPlacement === 'first-use' ? skeletonPlacement(program) : null;
    if (!skeletons) {
        w.comment('// skeletons');
        let declared = 0;
        for (const call of program.calls) {
            if (call.host || isAnonymous(call.skeleton, options)) continue;
            if (declared++) w.gap();
            declareSkeleton(call);
        }
        w.line();
    }
    w.comment('// the program');
    const uniforms = call => call.skeleton.uniforms.filter(u => u.name).map(u => cppNumber(u.value, u.type));
    const emitCall = call => {
        const id = call.skeleton.id, names = cs => cs.map(c => c.name);
        const args = [...names(call.elwise), ...names(call.proxies), ...uniforms(call)];
        if (call.io) {                                                  // skepu::io::cout of the data and results
            w.site('external', id);
            const ids = [id, ...call.reads.map(c => c.id)];
            for (const c of call.reads) w.line(`skepu::io::cout << ${c.name} << "\\n";`, ...ids);
            for (const source of call.scalars) w.line(`skepu::io::cout << "${source.skeleton.label.replace(/"/g, '\\"')}: " << ${source.name}_result << "\\n";`, id, source.skeleton.id);
            if (!call.reads.length && !call.scalars.length) w.line(`skepu::io::cout << "${call.skeleton.label.replace(/"/g, '\\"')}" << "\\n";`, id);
            return;
        }
        if (call.host) {
            const f = program.functions.find(f => f.call === call && f.host);
            const user = (f.bodies.cpp ?? '').trim();
            const given = [`"${call.skeleton.label.replace(/"/g, '\\"')}"`];
            if (call.reads.length) given.push(`skepu::read(${names(call.reads).join(', ')})`);
            w.site('external', id);
            w.line(`skepu::external(${given.join(', ')}, [&]`, id, ...call.reads.map(c => c.id), ...call.writes.map(c => c.id));
            w.line('{', id);
            w.depth++;
            for (const l of user ? user.split('\n') : w.compact ? bare(f.stub.cpp, false) : f.stub.cpp.split('\n')) w.line(l, id);
            w.depth--;
            w.line(`}${call.writes.length ? `, skepu::write(${names(call.writes).join(', ')})` : ''});`, id);
            return;
        }
        w.site('skeleton_call', id);
        const callArgs = call.pattern === 'MapPairs' || call.pattern === 'MapPairsReduce'
            ? [call.out.name, call.vertical.name, call.horizontal.name, ...names(call.proxies), ...uniforms(call)]
            : call.scalar ? args : [call.out.name, ...args];
        const scalar = call.scalar ? `${call.elementType} ${call.name}_result = ` : '';
        if (isAnonymous(call.skeleton, options)) {                       // anonymous: created and called here
            const { head, settings } = skeletonParts(call);
            const fns = program.functions.filter(f => f.call === call), invoke = `(${callArgs.join(', ')});`;
            if (settings.length) declare(`${scalar}${head}`, fns, id, '', [...settings.map(([text, ...ids]) => [`.${text}`, ...ids]), [invoke]]);
            else declare(`${scalar}${head}`, fns, id, invoke);
            return;
        }
        w.line(`${scalar ? scalar + call.name : call.name}(${callArgs.join(', ')});`, id);
    };
    const emit = steps => {
        steps.forEach((step, i) => {
            if (i) w.gap();
            placement?.before.get(step)?.forEach(declareContainer);
            skeletons?.before.get(step)?.filter(call => !isAnonymous(call.skeleton, options)).forEach((call, k) => { if (k) w.gap(); declareSkeleton(call); });
            if (step.type === 'call') { emitCall(step.call); return; }
            const v = step.loops === 0 ? 'iteration' : `iteration_${step.loops + 1}`;
            if (!step.scope) w.line(`for (size_t ${v} = 0; ${v} < ${step.count}; ++${v})`, step.loop.id);
            w.line('{', step.loop.id);
            w.depth++;
            w.times *= step.count;
            w.site('region', step.loop.id);
            w.line(`SKEPU_TRACE_SCOPE("${step.label.replace(/"/g, '\\"')}");`, step.loop.id);
            emit(step.steps);
            w.times /= step.count;
            w.depth--;
            w.line('}', step.loop.id);
        });
    };
    emit(program.steps);
    w.line();
    w.line('return 0;');
    w.depth--;
    w.line('}');
    return w.result();
}

const pyUniformValues = call => call.skeleton.uniforms.filter(u => u.name).map(u => pyNumber(u.value, u.type));
const pyArg = (key, values) => values.length === 0 ? [] : [`${key}=${values.length === 1 ? values[0] : `(${values.join(', ')})`}`];

/** The expression of a user function whose body is `return expression` (ignoring comments and blank lines); null otherwise. */
const pyOneLiner = f => {
    const user = (f.bodies.py ?? '').trim();
    const lines = (user || f.stub.py).split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
    const m = lines.length === 1 ? /^return\s+(\S.*)$/.exec(lines[0]) : null;
    return m && !m[1].includes('#') ? m[1] : null;
};

/**
 * The program as SkePY Python. `options.userFunctions`: 'functions' (default) or 'lambdas': a function whose body is `return expression` becomes a
 * lambda in the skeleton declaration; others stay functions.
 */
export function emitPython(program, options = {}) {
    const inline = f => isLambda(f, options) ? pyOneLiner(f) : null;
    const used = f => { const e = inline(f); return e === null ? f.name : `lambda ${f.params.map(p => p.name).join(', ')}: ${e}`; };
    const w = new Writer(options.whitespace);
    w.style = 'py';
    w.comment(`# ${program.name}: generated by the SkeVU designer. Function bodies are stubs.`);
    w.line('import skepu');
    w.line('import sys');
    w.line();
    for (const f of program.functions.filter(f => !f.host && inline(f) === null)) {
        const user = (f.bodies.py ?? '').trim();
        const body = user || (w.compact ? bare(f.stub.py, true).join('\n') : f.stub.py);
        w.line(`def ${f.name}(${f.params.map(p => p.name).join(', ')}):`, f.call.skeleton.id);
        w.depth++;
        if (!user) w.comment('# TODO: ' + f.call.skeleton.label + (f.suffix === '_reduce' ? ' (reduction)' : f.suffix === '_map' ? ' (map)' : ''), f.call.skeleton.id);
        for (const l of body.split('\n')) w.line(l, f.call.skeleton.id);
        w.depth--;
        w.line();
    }
    w.line('def main():');
    w.depth++;
    w.line('if len(sys.argv) < 2:');
    w.line('    print("Usage:", sys.argv[0], "backend")');
    w.line('    sys.exit(1)');
    w.line('skepu.set_global_backend_spec(skepu.BackendSpec(sys.argv[1]))');
    w.line();
    const declareContainer = c =>
        w.line(c.kind === 'PRNG' ? `${c.name} = skepu.PRNG(${String(c.init ?? '').trim()})` : `${c.name} = skepu.${c.kind}(${[...c.size, pyNumber(c.init, c.type)].join(', ')})`, ...c.nodes);
    const placement = options.containerPlacement === 'first-use' ? containerPlacement(program) : null;
    const atStart = placement ? placement.start : program.containers;
    if (atStart.length) w.comment(placement ? '# random streams and unused data' : '# containers');
    atStart.forEach(declareContainer);
    if (atStart.length) w.line();
    const declareSkeleton = call => {
        const s = call.skeleton, fns = program.functions.filter(f => f.call === call), id = s.id;
        switch (s.pattern) {
            case 'Map': case 'MapPairs': case 'Scan': case 'MapOverlap': case 'MapPool':
                w.line(`${call.name} = skepu.${s.pattern}(${used(fns[0])})`, id); break;
            case 'Reduce': w.line(`${call.name} = skepu.Reduce(${fns.map(used).join(', ')})`, id); break;
            case 'MapReduce': w.line(`${call.name} = skepu.MapReduce(${used(fns[0])}, ${used(fns[1])})`, id); break;
            case 'MapPairsReduce': w.line(`${call.name} = skepu.MapPairsReduce(${used(fns[0])}, ${used(fns[1])})`, id); break;
        }
        if (s.pattern === 'MapPairsReduce') {
            if (s.params.mode === 'ColWise') w.line(`${call.name}.set_reduce_mode(skepu.REDUCEMODE_ColWise)`, id);
            if (given(s.params.start)) w.line(`${call.name}.set_start_value(${pyNumber(s.params.start, call.elementType)})`, id);
        }
        if (s.pattern === 'MapReduce' && given(s.params.start)) w.line(`${call.name}.set_start_value(${pyNumber(s.params.start, call.elementType)})`, id);
        if ((s.pattern === 'Map' || s.pattern === 'MapReduce') && strided(call)) w.comment(`# ${call.name}: strides are not supported by SkePY.`, id);
        if (s.pattern === 'Reduce') {
            if ((call.partial || call.second) && s.params.mode === 'ColWise') w.line(`${call.name}.set_reduce_mode(skepu.REDUCEMODE_ColWise)`, id);
            if (String(s.params.start ?? '').trim()) w.line(`${call.name}.set_start_value(${pyNumber(s.params.start, call.elementType)})`, id);
        }
        if (s.pattern === 'MapPool') {
            const pool = poolOf(s), stride = strideOf(s);
            if (pool.join() === stride.join()) w.line(`${call.name}.set_pool_size_and_stride(${pool.join(', ')})`, id);
            else w.line(`${call.name}.set_pool_size(${pool.join(', ')})`, id), w.line(`${call.name}.set_stride(${stride.join(', ')})`, id);
        }
        if (s.pattern === 'Scan') {
            if (s.params.mode === 'Exclusive') w.line(`${call.name}.set_scan_mode(skepu.SCANMODE_Exclusive)`, id);
            if (String(s.params.start ?? '').trim()) w.line(`${call.name}.set_start_value(${pyNumber(s.params.start, call.elementType)})`, id);
        }
        if (s.pattern === 'MapOverlap') {
            if (overlapOf(s).some(o => o !== 1)) w.line(`${call.name}.set_overlap(${overlapOf(s).join(', ')})`, id);
            if (s.params.edge !== 'Duplicate') w.line(`${call.name}.set_edge_mode(skepu.EDGE_${s.params.edge})`, id);
            if (s.params.edge === 'Pad' && nonzero(s.params.pad)) w.line(`${call.name}.set_pad(${pyNumber(s.params.pad, call.elementType)})`, id);
            if (s.params.update && s.params.update !== 'Normal') w.line(`${call.name}.set_update_mode(skepu.UPDATEMODE_${s.params.update})`, id);
        }
        if (call.random) w.line(`${call.name}.set_prng(${call.random.name})`, id, call.random.id);
        if (s.backend) w.line(`${call.name}.set_backend(skepu.BackendSpec("${s.backend}"))`, id);
    };
    const skeletons = options.skeletonPlacement === 'first-use' ? skeletonPlacement(program) : null;
    if (!skeletons) {
        w.comment('# skeletons');
        let declared = 0;
        for (const call of program.calls) {
            if (call.host) continue;
            if (declared++) w.gap();
            declareSkeleton(call);
        }
        w.line();
    }
    w.comment('# the program');
    const emitCall = call => {
        const id = call.skeleton.id, names = cs => cs.map(c => c.name);
        const kw = [...pyArg('proxies', names(call.proxies)), ...pyArg('uniforms', pyUniformValues(call))];
        if (call.io) {
            w.site('external', id);
            const ids = [id, ...call.reads.map(c => c.id)];
            for (const c of call.reads) w.line(`print(${c.name})`, ...ids);
            for (const source of call.scalars) w.line(`print("${source.skeleton.label.replace(/"/g, '\\"')}:", ${source.name}_result)`, id, source.skeleton.id);
            if (!call.reads.length && !call.scalars.length) w.line(`print("${call.skeleton.label.replace(/"/g, '\\"')}")`, id);
            return;
        }
        if (call.host) {
            const f = program.functions.find(f => f.call === call && f.host);
            const user = (f.bodies.py ?? '').trim();
            w.site('external', id);
            w.line(`with skepu.external(${[...pyArg('read', names(call.reads)), ...pyArg('write', names(call.writes)), `label="${call.skeleton.label.replace(/"/g, '\\"')}"`].join(', ')}):`,
                id, ...call.reads.map(c => c.id), ...call.writes.map(c => c.id));
            w.depth++;
            for (const l of user ? user.split('\n') : w.compact ? bare(f.stub.py, true) : f.stub.py.split('\n')) w.line(l, id);
            w.depth--;
            return;
        }
        w.site('skeleton_call', id);
        if (call.indexed) kw.push(`ix=skepu.${call.indexType}`);
        if (call.random) kw.push('rand=skepu.Random(1)');
        switch (call.pattern) {
            case 'Map':
                w.line(`${call.name}(${[call.out.name, ...pyArg('inargs', names(call.elwise)), ...kw].join(', ')})`, id); break;
            case 'MapReduce':
                w.line(`${call.name}_result = ${call.name}(${[...pyArg('inargs', names(call.elwise)), ...kw].join(', ')})`, id);
                break;
            case 'Reduce':
                if (call.partial) w.comment(`# ${call.name}: Reduce to a vector is not supported by SkePY.`, id);
                else w.line(`${call.name}_result = ${call.name}(${call.elwise[0].name})`, id);
                break;
            case 'Scan': w.line(`${call.name}(${call.out.name}, ${call.elwise[0].name})`, id); break;
            case 'MapOverlap': case 'MapPool': w.line(`${call.name}(${[call.out.name, call.elwise[0].name, ...kw].join(', ')})`, id); break;
            case 'MapPairs': case 'MapPairsReduce': w.line(`${call.name}(${[call.out.name, `first=${call.vertical.name}`, `second=${call.horizontal.name}`, ...kw].join(', ')})`, id); break;
        }
    };
    const emit = steps => {
        steps.forEach((step, i) => {
            if (i) w.gap();
            placement?.before.get(step)?.forEach(declareContainer);
            skeletons?.before.get(step)?.forEach((call, k) => { if (k) w.gap(); declareSkeleton(call); });
            if (step.type === 'call') { emitCall(step.call); return; }
            const v = step.loops === 0 ? 'iteration' : `iteration_${step.loops + 1}`;
            if (!step.scope) { w.line(`for ${v} in range(${step.count}):`, step.loop.id); w.depth++; }
            w.times *= step.count;
            w.site('region', step.loop.id);
            w.line(`with skepu.scope("${step.label.replace(/"/g, '\\"')}"):`, step.loop.id);
            w.depth++;
            emit(step.steps);
            w.times /= step.count;
            w.depth -= step.scope ? 1 : 2;
        });
    };
    emit(program.steps);
    w.depth--;
    w.line();
    w.line('main()');
    return w.result();
}

/** The emitters by language; each takes the program and the options. */
export const EMITTERS = { cpp: emitCpp, python: emitPython };
