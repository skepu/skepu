// permalink.js — the link of an example: pure, no DOM. The address of a loaded example is its hash,
//   #example=horner&backend=openmp
// the slug of the example and its backend, and nothing of the state of the view (the settings are the reader's own).
'use strict';

/** The slug of an example: its title in lower case, with a dash for what is not a letter or a digit ("Cellular Automaton" -> cellular-automaton). */
export function exampleSlug(title) {
  return String(title ?? '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

/** The index of the example that has the slug, or -1. */
export function findExample(examples, slug) {
  return (examples ?? []).findIndex(e => exampleSlug(e.title) === slug);
}

/** The hash for { slug, backend }: an empty string if there is no example. */
export function encodePermalink(state) {
  if (!state || !state.slug) return '';
  const p = new URLSearchParams({ example: state.slug });
  if (state.backend) p.set('backend', state.backend);
  return '#' + p.toString();
}

/** { slug, backend } of a hash, or null if it names no example. */
export function decodePermalink(hash) {
  const p = new URLSearchParams(String(hash ?? '').replace(/^#/, ''));
  const slug = p.get('example');
  return slug ? { slug, backend: p.get('backend') || undefined } : null;
}
