// Polyfill Web Crypto for Node.js < 19 (Vitest runs in Node).
// Node 19+ exposes globalThis.crypto natively; older LTS releases need this shim.
import { webcrypto } from 'node:crypto';
if (!globalThis.crypto) {
    globalThis.crypto = webcrypto;
} else if (!globalThis.crypto.randomUUID) {
    globalThis.crypto.randomUUID = webcrypto.randomUUID.bind(webcrypto);
}
