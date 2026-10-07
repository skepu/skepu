// nesting.js — the nesting of regions and fusion nodes, as the backend reports it (region_parents).
'use strict';

/**
 * The ids of everything inside `id`, at any depth.
 * @param {Object} parents  id -> id of the region or fusion node around it, or null
 */
export function descendantsOf(parents, id) {
  var children = {};
  Object.keys(parents || {}).forEach(function(k) { (children[parents[k]] = children[parents[k]] || []).push(k); });
  var out = [], todo = [id];
  while (todo.length) {
    (children[todo.pop()] || []).forEach(function(c) { out.push(c); todo.push(c); });
  }
  return out;
}
