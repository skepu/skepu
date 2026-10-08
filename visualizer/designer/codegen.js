// codegen.js — the program of a lowered design (model.js lower()), as SkePU C++ and as SkePY Python.
// Each emitter returns { text, lines, sites } where lines[i] is the ids of the design nodes that line (1-based: lines[0] is line 1) is for, so
// that the editor can mark the code of the selected node and the node of the selected line; and sites are the lines where a skeleton is called
// or a loop (a region) is entered, as the nodes of a trace are: { type, id, line (1-based), count (times it runs), order (in the program) }.
'use strict';
import { readsOf, writesOf, overlapOf } from './model.js';

/**
 * Where each container is declared when it is declared before its first use (`containerPlacement: 'first-use'`): before the first step that
 * uses it, in the list of steps where all its uses are, and not inside a loop (it would be made again for each iteration): a loop is
 * entered only for a scope, a group that runs once. Random streams, and data that no call uses, are declared at the start (`start`).
 * Returns { before: Map(step -> the containers declared before it), start: [containers] }.
 */
export function containerPlacement(program) {
    const uses = new Map();                                    // object -> the paths (the steps from the top) of the calls that use it (by any of its updates)
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

/** Calls fn(call, path) for each call, with the steps from the top that lead to it. */
function walkCalls(steps, path, fn) {
    for (const step of steps) {
        const here = [...path, step];
        if (step.type === 'call') fn(step.call, here); else walkCalls(step.steps, here, fn);
    }
}

/** The step that something used by the calls at `paths` is declared before: the first of them, in the list of steps where all of them are, and not inside a loop (a scope is entered when all of it is in one). */
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
 * Where each skeleton is declared when it is declared before its first use (`skeletonPlacement: 'first-use'`): before its call, not inside a loop
 * (as for containers). Returns { before: Map(step -> the calls whose skeletons are declared before it) }.
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
 * The whitespace of the program (`options.whitespace`): 'balanced' (the default), 'readable' (a blank line between the fragments that come from
 * different nodes: the declarations of skeletons, the calls and the regions; what a call needs, its containers if they are declared
 * before their first use, and the setters of a skeleton, stay with it) or 'compact' (no comments of its own: no note on top, no names of
 * the sections, no note that a function is a stub; the comments of the user stay).
 */
class Writer {
    constructor(whitespace = 'balanced') {
        this.out = []; this.ids = []; this.depth = 0; this.sites = []; this.times = 1;
        this.readable = whitespace === 'readable'; this.compact = whitespace === 'compact';
    }
    /** A comment of the program: not in the compact form. */
    comment(text, ...ids) { if (!this.compact) this.line(text, ...ids); }
    /** A blank line between two fragments, in the readable form: not after another blank line, an opening of a block or a comment. */
    gap() {
        const last = (this.out[this.out.length - 1] ?? '').trim();
        if (this.readable && last !== '' && !last.endsWith('{') && !last.endsWith(':') && !last.startsWith('//') && !last.startsWith('#')) this.line();
    }
    /** The next line is where a call (type 'skeleton_call') or a loop ('region') of the design is. */
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

/** A stub without the note that it is one (what the compact form has): the lines of comments are gone, and a comment after a statement. */
const bare = (text, py) => text.split('\n').map(l => py ? l.replace(/\s*#.*$/, '') : l).filter(l => l.trim() && !l.trim().startsWith(py ? '#' : '//'));

/** The lines of a user function: what the user wrote, or the stub with a note that it is one. */
const cppBody = (f, compact) => {
    const user = (f.bodies.cpp ?? '').trim();
    return user ? user.split('\n') : compact ? bare(f.stub.cpp, false)
        : ['// TODO: the function of ' + f.call.skeleton.label + (f.suffix === '_reduce' ? ' (the reduction)' : f.suffix === '_map' ? ' (the map)' : ''), ...f.stub.cpp.split('\n')];
};

/**
 * The program as SkePU C++ (to be built with the precompiler: one skeleton for each line, or each statement).
 * `options.userFunctions`: 'functions' (the default: a function for each user function, before main) or 'lambdas' (a lambda in the
 * declaration of the skeleton). `options.containerPlacement`: 'grouped' (the default: the containers are declared together, before the
 * skeletons) or 'first-use' (before the first use, see containerPlacement()).
 */
export function emitCpp(program, options = {}) {
    const lambdas = options.userFunctions === 'lambdas';
    const w = new Writer(options.whitespace);
    w.style = 'cpp';
    w.comment(`// ${program.name}: made by the SkeVU designer. The function bodies are stubs: write them, or`);
    w.comment('// give them in the designer. Build it with the SkePU precompiler (see examples/Makefile).');
    w.line('#include <iostream>');
    w.line('#include <skepu>');
    w.line('#include <skepu-lib/io.hpp>');
    w.line();
    for (const f of lambdas ? [] : program.functions.filter(f => !f.host)) {
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
        if (c.kind === 'PRNG') {                       // (the seed, if there is one, then the name: the stream has the default seed when it is left out)
            const seed = String(c.init ?? '').trim(), label = c.label.replace(/"/g, '\\"');
            w.line(`skepu::PRNG ${c.name}(${seed ? `${seed}, ` : ''}"${label}");`, ...c.nodes);
            return;
        }
        const size = dims(c).join(', ');
        w.line(`skepu::${c.kind}<${c.type}> ${c.name}(${size}, "${c.label.replace(/"/g, '\\"')}", ${cppNumber(c.init, c.type)});`, ...c.nodes);
    };
    const placement = options.containerPlacement === 'first-use' ? containerPlacement(program) : null;
    const atStart = placement ? placement.start : program.containers;
    if (atStart.length) w.comment(placement ? '// random streams, and data that no call uses' : '// containers');
    atStart.forEach(declareContainer);
    if (atStart.length) w.line();
    /** `prefix` up to the parenthesis of the skeleton; then the user functions, by name or as lambdas (with the type that they return). */
    const declare = (prefix, fns, id) => {
        if (!lambdas) { w.line(`${prefix}${fns.map(f => f.name).join(', ')});`, id); return; }
        fns.forEach((f, i) => {
            w.line(`${i === 0 ? prefix : ''}[](${f.params.map(cppParam).join(', ')}) -> ${f.returns}`, id);
            w.line('{', id);
            w.depth++;
            for (const l of cppBody(f, w.compact)) w.line(l, id);
            w.depth--;
            w.line(i < fns.length - 1 ? '},' : '});', id);
        });
    };
    const declareSkeleton = call => {
        const s = call.skeleton, fns = program.functions.filter(f => f.call === call);
        const id = s.id;
        switch (s.pattern) {
            case 'Map':       declare(`auto ${call.name} = skepu::Map<${call.elwise.length}>(`, fns, id); break;
            case 'MapReduce': declare(`auto ${call.name} = skepu::MapReduce<${call.elwise.length}>(`, fns, id); break;
            case 'Reduce':    declare(`auto ${call.name} = skepu::Reduce(`, fns, id); break;
            case 'Scan':
                declare(`auto ${call.name} = skepu::Scan(`, fns, id);
                if (s.params.mode === 'Exclusive') w.line(`${call.name}.setScanMode(skepu::ScanMode::Exclusive);`, id);
                if (String(s.params.start ?? '').trim()) w.line(`${call.name}.setStartValue(${s.params.start});`, id);
                break;
            case 'MapOverlap': {
                declare(`auto ${call.name} = skepu::MapOverlap(`, fns, id);
                w.line(`${call.name}.setOverlap(${overlapOf(s).join(', ')});`, id);
                w.line(`${call.name}.setEdgeMode(skepu::Edge::${s.params.edge});`, id);
                if (s.params.edge === 'Pad') w.line(`${call.name}.setPad(${cppNumber(s.params.pad, call.elementType)});`, id);
                break;
            }
            case 'MapPairs':  declare(`auto ${call.name} = skepu::MapPairs<1, 1>(`, fns, id); break;
            case 'MapPairsReduce':
                declare(`auto ${call.name} = skepu::MapPairsReduce<1, 1>(`, fns, id);
                w.line(`${call.name}.setReduceMode(skepu::ReduceMode::${s.params.mode === 'ColWise' ? 'ColWise' : 'RowWise'});`, id);
                if (String(s.params.start ?? '').trim()) w.line(`${call.name}.setStartValue(${cppNumber(s.params.start, call.elementType)});`, id);
                break;
        }
        if (call.random) w.line(`${call.name}.setPRNG(${call.random.name});`, id, call.random.id);
        if (s.backend) w.line(`${call.name}.setBackend(skepu::BackendSpec{"${s.backend}"});`, id);
    };
    const skeletons = options.skeletonPlacement === 'first-use' ? skeletonPlacement(program) : null;
    if (!skeletons) {
        w.comment('// skeletons');
        let declared = 0;
        for (const call of program.calls) {
            if (call.host) continue;
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
        if (call.io) {                                                  // skepu::io::cout of the data, and of the results that it prints
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
        switch (call.pattern) {
            case 'MapReduce': case 'Reduce': {
                const result = `${call.name}_result`;
                w.line(`${call.elementType} ${result} = ${call.name}(${args.join(', ')});`, id);
                if (call.printed) w.line(`skepu::io::cout << "${call.skeleton.label.replace(/"/g, '\\"')}: " << ${result} << "\\n";`, id);
                break;
            }
            case 'MapPairs': case 'MapPairsReduce': w.line(`${call.name}(${[call.out.name, call.vertical.name, call.horizontal.name, ...names(call.proxies), ...uniforms(call)].join(', ')});`, id); break;
            default: w.line(`${call.name}(${[call.out.name, ...args].join(', ')});`, id);
        }
    };
    const emit = steps => {
        steps.forEach((step, i) => {
            if (i) w.gap();
            placement?.before.get(step)?.forEach(declareContainer);
            skeletons?.before.get(step)?.forEach((call, k) => { if (k) w.gap(); declareSkeleton(call); });
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

/** The expression of a user function whose body is one line, `return expression` (comments and empty lines apart): what a lambda can have. Null if it is not. */
const pyOneLiner = f => {
    const user = (f.bodies.py ?? '').trim();
    const lines = (user || f.stub.py).split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
    const m = lines.length === 1 ? /^return\s+(\S.*)$/.exec(lines[0]) : null;
    return m && !m[1].includes('#') ? m[1] : null;
};

/**
 * The program as SkePY Python (the functions are plain Python, as Shed Skin takes them: build with `shedskin build`).
 * `options.userFunctions`: 'functions' (the default) or 'lambdas': a user function with a body of one line, `return expression`, is a
 * lambda in the declaration of the skeleton; one with more lines is a function, as before (a lambda cannot have statements).
 */
export function emitPython(program, options = {}) {
    const lambdas = options.userFunctions === 'lambdas';
    const inline = f => lambdas && !f.host ? pyOneLiner(f) : null;
    const used = f => { const e = inline(f); return e === null ? f.name : `lambda ${f.params.map(p => p.name).join(', ')}: ${e}`; };
    const w = new Writer(options.whitespace);
    w.style = 'py';
    w.comment(`# ${program.name}: made by the SkeVU designer. The function bodies are stubs: write them, or`);
    w.comment('# give them in the designer. Build it with SkePY (SKEPY.md): python3 -m shedskin build --trace this_file.py');
    w.line('import skepu');
    w.line('import sys');
    w.line();
    for (const f of program.functions.filter(f => !f.host && inline(f) === null)) {
        const user = (f.bodies.py ?? '').trim();
        const body = user || (w.compact ? bare(f.stub.py, true).join('\n') : f.stub.py);
        w.line(`def ${f.name}(${f.params.map(p => p.name).join(', ')}):`, f.call.skeleton.id);
        w.depth++;
        if (!user) w.comment('# TODO: the function of ' + f.call.skeleton.label + (f.suffix === '_reduce' ? ' (the reduction)' : f.suffix === '_map' ? ' (the map)' : ''), f.call.skeleton.id);
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
    if (atStart.length) w.comment(placement ? '# random streams, and data that no call uses' : '# containers');
    atStart.forEach(declareContainer);
    if (atStart.length) w.line();
    const declareSkeleton = call => {
        const s = call.skeleton, fns = program.functions.filter(f => f.call === call), id = s.id;
        switch (s.pattern) {
            case 'Map': case 'MapPairs': case 'Scan': case 'MapOverlap': case 'Reduce':
                w.line(`${call.name} = skepu.${s.pattern}(${used(fns[0])})`, id); break;
            case 'MapReduce': w.line(`${call.name} = skepu.MapReduce(${used(fns[0])}, ${used(fns[1])})`, id); break;
            case 'MapPairsReduce': w.line(`${call.name} = skepu.MapPairsReduce(${used(fns[0])}, ${used(fns[1])})`, id); break;
        }
        if (s.pattern === 'MapPairsReduce') {
            w.line(`${call.name}.set_reduce_mode(skepu.REDUCEMODE_${s.params.mode === 'ColWise' ? 'ColWise' : 'RowWise'})`, id);
            if (String(s.params.start ?? '').trim()) w.line(`${call.name}.set_start_value(${pyNumber(s.params.start, call.elementType)})`, id);
        }
        if (s.pattern === 'Scan') {
            if (s.params.mode === 'Exclusive') w.line(`${call.name}.set_scan_mode(skepu.SCANMODE_Exclusive)`, id);
            if (String(s.params.start ?? '').trim()) w.line(`${call.name}.set_start_value(${pyNumber(s.params.start, call.elementType)})`, id);
        }
        if (s.pattern === 'MapOverlap') {
            w.line(`${call.name}.set_overlap(${overlapOf(s).join(', ')})`, id);
            w.line(`${call.name}.set_edge_mode(skepu.EDGE_${s.params.edge})`, id);
            if (s.params.edge === 'Pad') w.line(`${call.name}.set_pad(${pyNumber(s.params.pad, call.elementType)})`, id);
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
                if (call.printed) w.line(`print("${call.skeleton.label.replace(/"/g, '\\"')}:", ${call.name}_result)`, id);
                break;
            case 'Reduce':
                w.line(`${call.name}_result = ${call.name}(${call.elwise[0].name})`, id);
                if (call.printed) w.line(`print("${call.skeleton.label.replace(/"/g, '\\"')}:", ${call.name}_result)`, id);
                break;
            case 'Scan': w.line(`${call.name}(${call.out.name}, ${call.elwise[0].name})`, id); break;
            case 'MapOverlap': w.line(`${call.name}(${[call.out.name, call.elwise[0].name, ...kw].join(', ')})`, id); break;
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

/** The emitters by language; each takes the program and the options (`userFunctions`, `containerPlacement`, `skeletonPlacement`, `whitespace`). */
export const EMITTERS = { cpp: emitCpp, python: emitPython };
