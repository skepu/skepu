import { describe, it, expect } from 'vitest';
import { createSessionStore, sessionId, MAX_AGE_MS, SESSION_KEY } from '../../script/session_store.js';

// A small IndexedDB: one store of records by sid, the requests answered in a later task as the real ones are.
function fakeIdb(records = new Map(), failOpen = false) {
  const later = (req, fn) => setTimeout(() => { req.result = fn(); req.onsuccess && req.onsuccess(); }, 0);
  const objectStore = () => ({
    put: (v) => { const r = {}; later(r, () => { records.set(v.sid, v); }); return r; },
    get: (k) => { const r = {}; later(r, () => records.get(k)); return r; },
    getAll: () => { const r = {}; later(r, () => [...records.values()]); return r; },
    delete: (k) => { const r = {}; later(r, () => { records.delete(k); }); return r; },
  });
  return {
    records,
    open() {
      const req = {};
      setTimeout(() => {
        if (failOpen) { req.error = new Error('blocked'); req.onerror && req.onerror(); return; }
        req.result = { createObjectStore() {}, transaction: () => ({ objectStore }) };
        req.onupgradeneeded && req.onupgradeneeded();
        req.onsuccess && req.onsuccess();
      }, 0);
      return req;
    },
  };
}

describe('session store', () => {
  it('keeps the files of a session and gives them back', async () => {
    const idb = fakeIdb();
    const files = { trace: { name: 't.json' }, cpp: [{ name: 'a.cpp' }], zip: null };
    await createSessionStore(idb, 's1').save(files);
    const back = await createSessionStore(idb, 's1').restore();
    expect(back.trace.name).toBe('t.json');
    expect(back.cpp).toHaveLength(1);
  });
  it('keeps the sessions of other tabs apart', async () => {
    const idb = fakeIdb();
    await createSessionStore(idb, 's1').save({ trace: { name: 'one' } });
    expect(await createSessionStore(idb, 's2').restore()).toBeNull();
  });
  it('forgets', async () => {
    const idb = fakeIdb();
    const s = createSessionStore(idb, 's1');
    await s.save({ trace: { name: 'one' } });
    await s.forget();
    expect(await s.restore()).toBeNull();
  });
  it('removes what is older than a day when it is opened', async () => {
    const idb = fakeIdb(new Map([['old', { sid: 'old', saved: 0 }], ['new', { sid: 'new', saved: MAX_AGE_MS }]]));
    await createSessionStore(idb, 'x', () => MAX_AGE_MS + 1).restore();
    expect([...idb.records.keys()].sort()).toEqual(['new']);
  });
  it('does not fail where the database cannot be used', async () => {
    const s = createSessionStore(fakeIdb(new Map(), true), 's1');
    expect(await s.save({ trace: {} })).toBeNull();
    expect(await s.restore()).toBeNull();
  });
  it('has the id of the tab', () => {
    const data = {};
    const storage = { getItem: (k) => data[k] ?? null, setItem: (k, v) => { data[k] = v; } };
    expect(sessionId(storage, () => 'abc')).toBe('abc');
    expect(sessionId(storage, () => 'other')).toBe('abc');
    expect(data[SESSION_KEY]).toBe('abc');
    expect(sessionId({ getItem() { throw 1; }, setItem() { throw 1; } }, () => 'z')).toBe('z');
  });
});
