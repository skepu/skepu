// test_early.js — early.js decides before the first paint whether the page has a splash: only a blank session has (a new tab); a link to an
// example, files that this tab kept, and an editor have a plain curtain (class no-splash, main.css hides the card).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../script/early.js', import.meta.url), 'utf8');

function run({ hash = '', search = '', kept = null, editor = false } = {}) {
  const classes = new Set();
  const env = {
    document: { documentElement: { classList: { add: c => classes.add(c) } } },
    location: { hash, search },
    sessionStorage: { getItem: k => (k === 'skevu-upload-kept' ? kept : null) },
    localStorage: { getItem: () => null },
  };
  if (editor) env.acquireVsCodeApi = () => ({});
  new Function(...Object.keys(env), source)(...Object.values(env));
  return classes;
}

describe('the splash of a page', () => {
  it('is there for a blank session', () => {
    expect(run().has('no-splash')).toBe(false);
    expect(run({ hash: '#section' }).has('no-splash')).toBe(false);
  });
  it('is not there for a link to an example', () => {
    expect(run({ hash: '#example=horner&backend=openmp' }).has('no-splash')).toBe(true);
    expect(run({ hash: '#x=1&example=horner' }).has('no-splash')).toBe(true);
    expect(run({ search: '?example=2&backend=openmp' }).has('no-splash')).toBe(true);
  });
  it('is not there when this tab kept files', () => {
    expect(run({ kept: '1' }).has('no-splash')).toBe(true);
    expect(run({ kept: null }).has('no-splash')).toBe(false);
  });
  it('is never there in an editor', () => {
    expect(run({ editor: true }).has('no-splash')).toBe(true);
  });
  it('does not stop for a storage that cannot be read', () => {
    const classes = new Set();
    new Function('document', 'location', 'sessionStorage', 'localStorage', source)(
      { documentElement: { classList: { add: c => classes.add(c) } } }, { hash: '', search: '' },
      { getItem() { throw new Error('blocked'); } }, { getItem() { throw new Error('blocked'); } });
    expect(classes.has('no-pane-anim')).toBe(true);
  });
});
