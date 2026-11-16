// test_skeleton_registry.js — the registry of the skeleton access-pattern visualizers in the user guide (user-guide/_extensions/skevu/skeletons.js):
// every entry resolves to skeletons whose defaults agree with their controls, and whose listing and description can be made.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SOURCE = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'user-guide', '_extensions', 'skevu', 'skeletons.js');

// the drawing needs a document; only the data of the entries is used here
const sandbox = { VizUtils: { buildMatrix() {}, makeMatrixGroup() {}, stack() {} } };
sandbox.window = sandbox;
vm.runInNewContext(readFileSync(SOURCE, 'utf8'), sandbox);
const { SkePUSkeletons, SkePUEntries, SkePUOperands } = sandbox;

const keysOf = entry => entry.variants ? entry.variants.map(v => v.key) : [entry.key];

describe('the entries of the visualizers', () => {
    it('have unique ids and resolve to skeletons', () => {
        expect(new Set(SkePUEntries.map(e => e.id)).size).toBe(SkePUEntries.length);
        for (const e of SkePUEntries) for (const k of keysOf(e)) expect(SkePUSkeletons[k], `${e.id}: ${k}`).toBeDefined();
    });

    it('cover every skeleton of the registry once', () => {
        const used = SkePUEntries.flatMap(keysOf).sort();
        expect(used).toEqual(Object.keys(SkePUSkeletons).sort());
    });

    it('start their variants with the same size, which the variant selector keeps', () => {
        for (const e of SkePUEntries.filter(x => x.variants)) {
            const sizes = new Set(keysOf(e).map(k => `${SkePUSkeletons[k].defaults.rows}x${SkePUSkeletons[k].defaults.cols}`));
            expect(sizes.size, e.id).toBe(1);
        }
    });
});

describe('the skeletons of the registry', () => {
    for (const [key, sk] of Object.entries(SkePUSkeletons)) {
        it(`${key}: the defaults are within the controls`, () => {
            expect(sk.defaults.rows).toBeGreaterThanOrEqual(1);
            expect(sk.defaults.rows).toBeLessThanOrEqual(12);
            expect(sk.defaults.cols).toBeGreaterThanOrEqual(1);
            expect(sk.defaults.cols).toBeLessThanOrEqual(12);
            for (const c of sk.extraControls || []) {
                const v = sk.defaults[c.id] ?? c.default;
                if (c.type === 'number') { expect(v, c.id).toBeGreaterThanOrEqual(c.min); expect(v, c.id).toBeLessThanOrEqual(c.max); }
                else expect(c.options.map(o => o.value), c.id).toContain(v);
            }
        });

        it(`${key}: has the listing, the description and the legend`, () => {
            const p = { ...sk.defaults };
            expect(typeof sk.codeGen(p)).toBe('string');
            expect(typeof (typeof sk.desc === 'function' ? sk.desc(p) : sk.desc)).toBe('string');
            expect(sk.legend.length).toBeGreaterThan(0);
        });
    }
});

describe('the outputs, proxies and uniforms', () => {
    const withOperands = Object.entries(SkePUSkeletons).filter(([, sk]) => sk.operands || sk.outputs);

    it('have controls whose defaults are within their range', () => {
        for (const c of [SkePUOperands.outputsControl, ...SkePUOperands.controls]) {
            if (c.type === 'number') { expect(c.default).toBeGreaterThanOrEqual(c.min); expect(c.default).toBeLessThanOrEqual(c.max); }
            else expect(c.options.map(o => o.value)).toContain(c.default);
        }
    });

    it('have a proxy of the size that is drawn for each option', () => {
        const options = SkePUOperands.controls.find(c => c.id === 'proxy').options.map(o => o.value).filter(v => v !== 'none');
        expect(Object.keys(SkePUOperands.proxies).sort()).toEqual(options.sort());
    });

    for (const [key, sk] of withOperands) {
        it(`${key}: the call lists the outputs first, then the inputs, the proxy and the uniforms`, () => {
            const code = sk.codeGen({ ...sk.defaults, outputs: sk.outputs ? 2 : 1, proxy: 'mat', uniforms: 2 });
            const call = code.split('\n').find(l => l.startsWith('skel('));
            const names = call.slice(5, -2).split(', ');
            expect(names.slice(-3)).toEqual(['proxy', 'u1', 'u2']);
            if (sk.outputs) expect(names.slice(0, 2)).toEqual(['out1', 'out2']);
            else expect(names[0]).toBe('out');
        });
    }
});

