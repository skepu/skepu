// graph_interaction.js — what the pointer and the selection do to a Cytoscape graph of SkeVU, for GraphView and the designer: which nodes and edges
// are marked as selected, and the hover.

/**
 * Mark the nodes `nodes` and the edges `edges` (collections) as selected, and nothing else: a selected node has its edges marked, a selected
 * edge has the nodes at its ends marked.
 */
export function highlightSelection(cy, nodes, edges) {
  cy.nodes().removeClass('selected_gutter');
  cy.edges().removeClass('edge-highlighted');
  nodes.addClass('selected_gutter');
  nodes.connectedEdges().addClass('edge-highlighted');
  edges.addClass('edge-highlighted');
  edges.connectedNodes().addClass('selected_gutter');
}

/**
 * The hover of the nodes and edges of `cy` (the element that it is in is `el`): the element under the pointer has the class `cy-hovered`.
 * handlers: over(ele, event), move(event), out(ele), and leave() (the pointer leaves the graph); the pointer pressed or the view moved: reset().
 * Returns a function that takes the listeners of `el` away again (the graph may be made again in the same element).
 */
export function bindHover(cy, el, handlers) {
  cy.on('mouseover', 'node, edge', function(event) {
    var target = event.target;
    if (target.hasClass('layout-phantom')) return;
    target.addClass('cy-hovered');
    if (handlers.over) handlers.over(target, event);
  });
  cy.on('mousemove', 'node, edge', function(event) { if (handlers.move) handlers.move(event); });
  cy.on('mouseout mousedown', 'node, edge', function(event) {
    event.target.removeClass('cy-hovered');
    if (handlers.out) handlers.out(event.target);
  });
  cy.on('mousedown viewport', function() { if (handlers.reset) handlers.reset(); });
  function onLeave() {
    cy.elements().removeClass('cy-hovered');
    if (handlers.leave) handlers.leave();
  }
  el.addEventListener('mouseleave', onLeave);
  return function unbind() { el.removeEventListener('mouseleave', onLeave); };
}
