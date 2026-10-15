// test_provider_events.js — the events of the TraceDataProvider. It ignores `on` and `_emit` of an event it does not know, so every event that
// the scripts use is checked to be one the provider has.
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TraceDataProvider } from '../../script/TraceDataProvider.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
function scripts(dir) {
    return readdirSync(dir).flatMap(f => {
        const p = join(dir, f);
        if (f === 'static') return [];
        return statSync(p).isDirectory() ? scripts(p) : p.endsWith('.js') ? [p] : [];
    });
}

describe('the events of the provider', () => {
    const provider = new TraceDataProvider({ fetchGraph: async () => ({}) });
    const known = Object.keys(provider._listeners);

    it('are the ones that the scripts emit and listen to', () => {
        const used = new Map();   // event -> the files that use it
        for (const file of scripts(join(ROOT, 'script'))) {
            const text = readFileSync(file, 'utf8');
            // (a provider is named provider, _provider, State.provider or _attachedProvider in the scripts)
            for (const m of text.matchAll(/[pP]rovider\.(?:on|off|_emit)\(\s*'(\w+)'/g))
                used.set(m[1], [...(used.get(m[1]) ?? []), file.slice(ROOT.length + 1)]);
        }
        expect(used.size).toBeGreaterThan(5);
        for (const [event, files] of used)
            expect(known, `the event '${event}' (${[...new Set(files)].join(', ')}) is not one that the provider has`).toContain(event);
    });

    it('call a listener of rendersettled (the splash fades and the panes slide from the first one on)', () => {
        let got = null;
        provider.on('rendersettled', stats => { got = stats; });
        provider._emit('rendersettled', { nodes: 3 });
        expect(got).toEqual({ nodes: 3 });
    });
});
