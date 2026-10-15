import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { SimulationEngine } from '../../script/simulation.js';

// graph.js drives the engine from the Actions menu: every method it calls must exist.
describe('simulation engine calls of graph.js', () => {
  it('only calls methods that SimulationEngine has', () => {
    const src = readFileSync(new URL('../../script/graph.js', import.meta.url), 'utf8');
    const called = new Set([...src.matchAll(/_simEngine\.([A-Za-z_]+)\(/g)].map(m => m[1]));
    expect(called.size).toBeGreaterThan(0);
    for (const name of called)
      expect(typeof SimulationEngine.prototype[name], name).toBe('function');
  });
});
