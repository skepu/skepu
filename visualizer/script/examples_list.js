// examples_list.js — the list of the built-in examples by the language of the program (C++ or Python): pure, no DOM.
// An entry of examples.json has "language": "cpp" or "python" (C++ if it does not say).
'use strict';

export const LANGUAGE_LABELS = { cpp: 'C++', python: 'Python' };

/** The language of an example. */
export function languageOf(example) {
    return example && example.language === 'python' ? 'python' : 'cpp';
}

/** Whether an example has snapshots of containers for a backend (its trace has them, and the archive of the files): the welcome screen marks it. */
export function hasSnapshots(example, backend) {
    const archive = example && example.snapshot_archive;
    if (typeof archive === 'string') return archive !== '';
    return !!(archive && typeof archive === 'object' && archive[backend]);
}

/** The languages that the examples have, C++ first. */
export function exampleLanguages(examples) {
    const present = new Set((examples ?? []).map(languageOf));
    return Object.keys(LANGUAGE_LABELS).filter(l => present.has(l));
}

/** The examples of a language, each with its index in the list (what loadExample takes). */
export function examplesOfLanguage(examples, language) {
    return (examples ?? []).map((example, index) => ({ example, index })).filter(e => languageOf(e.example) === language);
}

/** The language to start with: the one that was chosen before if there are examples of it, else the first. */
export function initialLanguage(examples, remembered) {
    const languages = exampleLanguages(examples);
    return languages.includes(remembered) ? remembered : (languages[0] ?? 'cpp');
}
