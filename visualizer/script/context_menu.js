// context_menu.js — the context menu of a graph node (right click), in HTML: the items are { label, onClick, disabled? }, or elements.
// The menu closes on a click, Escape, a scroll or a resize.

var _menu = null;

/** The menu element, made the first time (a click anywhere, Escape, a resize or a scroll closes it). */
function menuElement() {
  if (_menu) return _menu;
  _menu = document.createElement('div');
  _menu.id = 'context-menu';
  _menu.className = 'dropdown context-menu';
  _menu.setAttribute('role', 'menu');
  document.body.appendChild(_menu);
  _menu.addEventListener('contextmenu', function(event) { event.preventDefault(); });   // (macOS sends it on the press, when the menu is already under the pointer)
  document.addEventListener('click', hideContextMenu);
  document.addEventListener('keydown', function(event) { if (event.key === 'Escape') hideContextMenu(); });
  window.addEventListener('resize', hideContextMenu);
  window.addEventListener('blur', hideContextMenu);
  document.addEventListener('wheel', hideContextMenu, { passive: true, capture: true });
  return _menu;
}

/** A button that has the look of the items of a menu: for `{ label, onClick, disabled }`. */
function makeItem(spec) {
  var button = document.createElement('button');
  button.className = 'menu-action';   // (the look of the items of the menu bar, dark mode included)
  button.setAttribute('role', 'menuitem');
  button.textContent = spec.label;
  button.disabled = !!spec.disabled;
  if (spec.title) button.title = spec.title;
  button.addEventListener('click', spec.onClick);
  return button;
}

/**
 * Show the menu with its corner at the pointer (x, y in the window), moved in if it would be outside it. `items` are elements or
 * `{ label, onClick, disabled }`. Returns the buttons, in the order of the items (a caller can enable or disable one when it knows more).
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
