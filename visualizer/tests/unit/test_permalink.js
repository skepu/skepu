import { describe, it, expect } from 'vitest';
import { exampleSlug, findExample, encodePermalink, decodePermalink } from '../../script/permalink.js';

describe('permalink', () => {
  it('makes slugs of titles', () => {
    expect(exampleSlug('Cellular Automaton')).toBe('cellular-automaton');
    expect(exampleSlug(' N-Body (MPI) ')).toBe('n-body-mpi');
  });
  it('finds an example by its slug', () => {
    expect(findExample([{ title: 'Horner' }, { title: 'Dot Product' }], 'dot-product')).toBe(1);
    expect(findExample([{ title: 'Horner' }], 'nope')).toBe(-1);
  });
  it('has no hash without an example', () => {
    expect(encodePermalink(null)).toBe('');
    expect(encodePermalink({ slug: '' })).toBe('');
  });
  it('says the example and the backend only', () => {
    expect(encodePermalink({ slug: 'horner', backend: 'openmp' })).toBe('#example=horner&backend=openmp');
    expect(encodePermalink({ slug: 'horner' })).toBe('#example=horner');
  });
  it('round-trips, and ignores everything else in the hash', () => {
    expect(decodePermalink('#example=horner&backend=mpi&direction=horizontal')).toEqual({ slug: 'horner', backend: 'mpi' });
    expect(decodePermalink('#example=horner')).toEqual({ slug: 'horner', backend: undefined });
  });
  it('is null for a hash that names no example', () => {
    expect(decodePermalink('')).toBeNull();
    expect(decodePermalink('#section-2')).toBeNull();
  });
});
