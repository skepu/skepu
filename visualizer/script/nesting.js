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

/**
 * Of `ids`, the ones that are collapsed and drawn: no region or fusion node around them is collapsed (what is inside a
 * collapsed node is not drawn). Expanding these, and no others, opens one level.
 * @param {Object} parents    id -> id of the node around it, or null
 * @param {Object} collapsed  id -> true if collapsed
 */
export function drawnCollapsed(parents, collapsed, ids) {
  return ids.filter(function(id) {
    if (collapsed[id] !== true) return false;
    for (var p = (parents || {})[id]; p; p = parents[p])
      if (collapsed[p] === true) return false;
    return true;
  });
}
