// test_examples_list.js — the built-in examples by language (the segmented control of the welcome panel).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { languageOf, exampleLanguages, examplesOfLanguage, initialLanguage, hasSnapshots } from '../../script/examples_list.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const list = [{ title: 'A' }, { title: 'B', language: 'python' }, { title: 'C', language: 'cpp' }, { title: 'D', language: 'python' }];

describe('the language of the examples', () => {
    it('is C++ unless the example says Python', () => {
        expect(list.map(languageOf)).toEqual(['cpp', 'python', 'cpp', 'python']);
    });

    it('the languages are those that the examples have, C++ first', () => {
        expect(exampleLanguages(list)).toEqual(['cpp', 'python']);
        expect(exampleLanguages([list[1]])).toEqual(['python']);
        expect(exampleLanguages([])).toEqual([]);
    });

    it('the examples of a language keep their index in the whole list (what loadExample takes)', () => {
        expect(examplesOfLanguage(list, 'python').map(e => [e.index, e.example.title])).toEqual([[1, 'B'], [3, 'D']]);
        expect(examplesOfLanguage(list, 'cpp').map(e => e.index)).toEqual([0, 2]);
    });

    it('starts with the language that was chosen, if there are examples of it', () => {
        expect(initialLanguage(list, 'python')).toBe('python');
        expect(initialLanguage(list, 'rust')).toBe('cpp');
        expect(initialLanguage(list, null)).toBe('cpp');
        expect(initialLanguage([list[1]], 'cpp')).toBe('python');
    });
});

describe('the shipped examples', () => {
    const examples = JSON.parse(readFileSync(join(ROOT, 'examples/examples.json'), 'utf8'));

    it('have both languages, and the Python examples are programs of Python', () => {
        expect(exampleLanguages(examples)).toEqual(['cpp', 'python']);
        for (const { example } of examplesOfLanguage(examples, 'python'))
            expect([].concat(example.cpp_files?.default ?? example.cpp_files).every(f => f.endsWith('.py'))).toBe(true);
        for (const { example } of examplesOfLanguage(examples, 'cpp'))
            expect(JSON.stringify(example.cpp_files)).not.toMatch(/\.py"/);
    });
});


describe('the examples that have snapshots of containers', () => {
    it('are marked for the backends that have an archive of them (the welcome screen shows a star there)', () => {
        const example = { title: 'Perlin', snapshot_archive: { openmp: 'a.zip', opencl: 'a.zip', mpi: 'm.zip' } };
        expect(hasSnapshots(example, 'openmp')).toBe(true);
        expect(hasSnapshots(example, 'mpi')).toBe(true);
        expect(hasSnapshots(example, 'cuda')).toBe(false);
    });

    it('are all the backends of an example with one archive', () => {
        expect(hasSnapshots({ snapshot_archive: 'cellular/snapshots.zip' }, 'opencl')).toBe(true);
        expect(hasSnapshots({ snapshot_archive: 'cellular/snapshots.zip' }, 'anything')).toBe(true);
    });

    it('are not marked for an example without an archive', () => {
        expect(hasSnapshots({ title: 'Horner' }, 'openmp')).toBe(false);
        expect(hasSnapshots({ snapshot_archive: '' }, 'openmp')).toBe(false);
        expect(hasSnapshots({ snapshot_archive: null }, 'openmp')).toBe(false);
        expect(hasSnapshots(undefined, 'openmp')).toBe(false);
    });

    it('are the examples of the list that name an archive for the backend of their traces', () => {
        const list = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../../examples/examples.json'), 'utf-8'));
        const marked = list.filter(e => Object.keys(e.trace).some(b => hasSnapshots(e, b))).map(e => e.title);
        expect(marked).toContain('Cellular Automaton');
        expect(marked).not.toContain('Dot Product');
    });
});
