// menu.js — the menu bar menus, for SkeVU and the designer. Pressing a title opens its menu and closes the others; with a menu open, hovering another title
// opens that one; a click elsewhere closes them. Releasing the button over an item, after pressing on a title, activates the item; releasing on the title
// keeps the menu open. A click in a `.menu-field` (a text entry) keeps the menu open. An item that does nothing is `disabled` (greyed by main.css); a menu
// fires 'menuopen' when it opens, so that items can be set.

/** Close all the menus. */
export function menuClose() {
  document.querySelectorAll('.menu.open').forEach(function(m) { m.classList.remove('open'); });
  unhighlight();
}

/** The browser does not update :hover while the button is held after a press on a title, so the item under the pointer gets the class `menu-hot`. */
function unhighlight() {
  document.querySelectorAll('.menu-hot').forEach(function(e) { e.classList.remove('menu-hot'); });
}

function highlightAt(x, y) {
  var el = document.elementFromPoint(x, y);   // before the marks are removed: the pointer may be over a submenu that they keep open
  var item = el && el.closest('.menu.open .dropdown button:not(:disabled), .menu.open .dropdown .submenu-title');
  if (!item) {
    // Over no item (outside the menus, or on a separator): open submenus stay open, as in macOS, until the pointer is over another item; only the
    // highlight of a non-submenu item goes
    document.querySelectorAll('.menu-hot').forEach(function(e) { if (!e.classList.contains('submenu')) e.classList.remove('menu-hot'); });
    return;
  }
  unhighlight();
  item.classList.add('menu-hot');
  // the item's menus are open: hovering a submenu title opens it
  for (var sub = item.closest('.submenu'); sub; sub = sub.parentElement && sub.parentElement.closest('.submenu')) sub.classList.add('menu-hot');
}

/** Open a menu: the event 'menuopen' (on the menu, bubbling) says so, so that a page can set which of its items would do something now. */
function menuOpen(menu) {
  menu.classList.add('open');
  menu.dispatchEvent(new CustomEvent('menuopen', { bubbles: true }));
}

/** Open the menu of the title `btn`, or close it if it is open. */
export function menuToggle(event, btn) {
  event.stopPropagation();
  var menu   = btn.closest('.menu');
  var isOpen = menu.classList.contains('open');
  menuClose();
  if (!isOpen) menuOpen(menu);
}

/** Where the button is released after a press on a title: 'title', 'item' (a button of an open menu), 'menu' (elsewhere in one) or 'outside'. */
export function releaseOutcome(over) {
  return over === 'item' ? 'activate' : over === 'outside' ? 'close' : 'keep';
}

/** Make the menus of the page work: the titles (`.menu-title`) that are there now, and the closing by a click elsewhere. Also `window.menuToggle` and `window.menuClose` for the handlers in markup. */
export function installMenus() {
  var pressed = false;   // the button was pressed on a title: its release may activate an item
  // a title that has an action of its own, such as the ? of the shortcuts, opens no menu
  document.querySelectorAll('.menu-title:not([data-click])').forEach(function(btn) {
    btn.addEventListener('mousedown', function(event) {
      if (event.button !== 0) return;
      event.preventDefault();   // no text selection while moving to an item
      pressed = true;
      menuToggle(event, btn);
    });
    // the click after a press is not another toggle; a keyboard click (Enter, Space) has no press and is
    btn.addEventListener('click', function(event) {
      if (event.detail === 0) menuToggle(event, btn); else event.stopPropagation();
    });
    btn.addEventListener('mouseenter', function() {
      if (document.querySelector('.menu.open')) { menuClose(); menuOpen(btn.closest('.menu')); }
    });
  });
  document.addEventListener('mousemove', function(event) {
    if (pressed) highlightAt(event.clientX, event.clientY);
  });
  document.addEventListener('mouseup', function(event) {
    if (!pressed) return;
    pressed = false;
    // The element at the pointer, not event.target: some browsers send the release to the pressed element. Looked up before the item marks are removed: they
    // keep a submenu open, and its items are missing without them.
    var target = document.elementFromPoint(event.clientX, event.clientY) || event.target;
    unhighlight();
    var item = target && target.closest('.menu.open .dropdown button:not(:disabled):not(.submenu-title)');
    var over = target && target.closest('.menu-title') ? 'title' : item ? 'item' : target && target.closest('.menu.open') ? 'menu' : 'outside';
    var outcome = releaseOutcome(over);
    if (outcome === 'activate') item.click();
    else if (outcome === 'close') menuClose();
  });
  document.addEventListener('click', function(event) {
    if (event.target.closest && event.target.closest('.menu-field')) return;
    menuClose();
  });
  window.menuToggle = menuToggle;
  window.menuClose  = menuClose;
}

/** Enable or disable an item of a menu (a button, or the selector of one). */
export function enableMenuItem(item, on) {
  var b = typeof item === 'string' ? document.querySelector(item) : item;
  if (b) b.disabled = !on;
}
