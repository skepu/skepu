// session_store.js — the files that were uploaded in this tab, kept in IndexedDB so that a reload of the tab (or a debugging session) has them
// again. The key is an id that sessionStorage keeps: it is the tab's (a reload and a duplicated tab have it, a new tab does not), so the
// files go with the tab and need no way to be forgotten; what is older than a day is removed when the store is opened. The files stay in the
// browser: nothing is sent. Without IndexedDB (a private window, blocked storage) there is nothing kept, and nothing fails.
'use strict';

export const DB_NAME = 'skevu-session';
export const STORE   = 'uploads';
export const SESSION_KEY = 'skevu-session-id';
export const MAX_AGE_MS  = 24 * 60 * 60 * 1000;

/** The id of this tab's session (made the first time). */
export function sessionId(storage, makeId) {
  let id = null;
  try { id = storage.getItem(SESSION_KEY); } catch (e) { /* (no storage) */ }
  if (!id) {
    id = makeId();
    try { storage.setItem(SESSION_KEY, id); } catch (e) { /* (no storage: the id lasts until the page does) */ }
  }
  return id;
}

function request(req) {
  return new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); });
}

function open(idb) {
  return new Promise((resolve, reject) => {
    const req = idb.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'sid' });
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

/**
 * The store of a session: { save({trace, cpp, zip}), restore(), forget() }. `idb` is an IDBFactory, `now` a clock (for the tests).
 * Every call resolves (to null or nothing) when the database cannot be used.
 */
export function createSessionStore(idb, sid, now = Date.now) {
  let db = null;
  async function store(mode) {
    if (!db) {
      db = await open(idb);
      // (what is older than a day is not any tab's any more)
      const all = db.transaction(STORE, 'readwrite').objectStore(STORE);
      const records = await request(all.getAll());
      for (const r of records) if (now() - r.saved > MAX_AGE_MS) all.delete(r.sid);
    }
    return db.transaction(STORE, mode).objectStore(STORE);
  }
  const safe = async (fn) => { try { return await fn(); } catch (e) { return null; } };
  return {
    save: (files) => safe(async () => { await request((await store('readwrite')).put({ sid, saved: now(), ...files })); }),
    restore: () => safe(async () => (await request((await store('readonly')).get(sid))) ?? null),
    forget: () => safe(async () => { await request((await store('readwrite')).delete(sid)); }),
  };
}
