// menu.js — the menus of the menu bar, for SkeVU and the designer: pressing the button on the title of a menu opens it (and closes the others), the pointer
// over another title while a menu is open opens that one, and a click anywhere else closes them. If the button is released over an item, after the pointer
// moved there with the button down, the item is activated; released on the title the menu stays open. A click inside a `.menu-field` (a text entry in a menu)
// does not close the menu. An item that would do nothing is `disabled` (greyed by main.css); a menu says 'menuopen' when it opens, to set that.

/** Close all the menus. */
export function menuClose() {
  document.querySelectorAll('.menu.open').forEach(function(m) { m.classList.remove('open'); });
  unhighlight();
}

/** While the button is held down after a press on a title the browser does not update :hover, so the item under the pointer is marked by a class (`menu-hot`). */
function unhighlight() {
  document.querySelectorAll('.menu-hot').forEach(function(e) { e.classList.remove('menu-hot'); });
}

function highlightAt(x, y) {
  var el = document.elementFromPoint(x, y);   // (before the marks are taken away: a submenu that is open by them is what the pointer may be over)
  var item = el && el.closest('.menu.open .dropdown button:not(:disabled), .menu.open .dropdown .submenu-title');
  if (!item) {
    // Over no item (outside the menus, or on a separator): the open submenus stay open, as in macOS, until the pointer is over another item; only the
    // highlight of an item that is not a submenu goes
    document.querySelectorAll('.menu-hot').forEach(function(e) { if (!e.classList.contains('submenu')) e.classList.remove('menu-hot'); });
    return;
  }
  unhighlight();
  item.classList.add('menu-hot');
  // (the menus that the item is in are open: a submenu opens by hovering its title)
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

/** What releasing the button does after a press on a title, by where it is released: 'title', 'item' (a button of an open menu), 'menu' (the rest of one) or 'outside'. */
export function releaseOutcome(over) {
  return over === 'item' ? 'activate' : over === 'outside' ? 'close' : 'keep';
}

/** Make the menus of the page work: the titles (`.menu-title`) that are there now, and the closing by a click elsewhere. Also `window.menuToggle` and `window.menuClose` for the handlers in markup. */
export function installMenus() {
  var pressed = false;   // the button was pressed on a title: its release may activate an item
  // (a title that has an action of its own, such as the ? of the shortcuts, opens no menu)
  document.querySelectorAll('.menu-title:not([data-click])').forEach(function(btn) {
    btn.addEventListener('mousedown', function(event) {
      if (event.button !== 0) return;
      event.preventDefault();   // (no selection of text while the pointer moves to an item)
      pressed = true;
      menuToggle(event, btn);
    });
    // (the click that follows a press is not another toggle; one from the keyboard, Enter or Space, has no press and is)
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
    // (the element at the pointer, not event.target: after a press some browsers send the release to the element that was pressed. It is looked for before the
    // marks of the item under the pointer are taken away: they keep a submenu open, and an item of it is not there without them)
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
