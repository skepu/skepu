// context_menu.js — the HTML context menu of a graph node (right click). Items are { label, onClick, disabled? } or elements.
// It closes on a click, Escape, a scroll or a resize.

var _menu = null;

/** Created on first use. */
function menuElement() {
  if (_menu) return _menu;
  _menu = document.createElement('div');
  _menu.id = 'context-menu';
  _menu.className = 'dropdown context-menu';
  _menu.setAttribute('role', 'menu');
  document.body.appendChild(_menu);
  _menu.addEventListener('contextmenu', function(event) { event.preventDefault(); });   // macOS sends it on the press, with the menu already under the pointer
  document.addEventListener('click', hideContextMenu);
  document.addEventListener('keydown', function(event) { if (event.key === 'Escape') hideContextMenu(); });
  window.addEventListener('resize', hideContextMenu);
  window.addEventListener('blur', hideContextMenu);
  document.addEventListener('wheel', hideContextMenu, { passive: true, capture: true });
  return _menu;
}

/** A menu-styled button for `{ label, onClick, disabled }`. */
function makeItem(spec) {
  var button = document.createElement('button');
  button.className = 'menu-action';   // the menu bar item style, dark mode included
  button.setAttribute('role', 'menuitem');
  button.textContent = spec.label;
  button.disabled = !!spec.disabled;
  if (spec.title) button.title = spec.title;
  button.addEventListener('click', spec.onClick);
  return button;
}

/**
 * Shows the menu at the pointer (x, y in window coordinates), moved inside the window if needed. `items`: elements or `{ label, onClick, disabled }`.
 * Returns the buttons in item order, for callers that enable or disable one later.
 */
export function showContextMenu(x, y, items) {
  var menu = menuElement();
  var buttons = (items || []).map(function(item) { return item instanceof Element ? item : makeItem(item); });
  menu.replaceChildren.apply(menu, buttons);
  menu.classList.add('open');
  menu.style.left = Math.max(0, Math.min(x, window.innerWidth  - menu.offsetWidth  - 2)) + 'px';
  menu.style.top  = Math.max(0, Math.min(y, window.innerHeight - menu.offsetHeight - 2)) + 'px';
  return buttons;
}

export function hideContextMenu() {
  if (_menu) _menu.classList.remove('open');
}
