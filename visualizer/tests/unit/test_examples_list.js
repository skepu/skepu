// test_examples_list.js — the built-in examples by language (the segmented control of the welcome panel).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { languageOf, exampleLanguages, examplesOfLanguage, initialLanguage } from '../../script/examples_list.js';

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
