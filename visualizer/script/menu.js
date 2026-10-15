// menu.js — the menus of the menu bar, for SkeVU and the designer: a click on the title of a menu opens it (and closes the others), the pointer
// over another title while a menu is open opens that one, and a click anywhere else closes them. A click inside a `.menu-field` (a text entry in
// a menu) does not close the menu. An item that would do nothing is `disabled` (greyed by main.css); a menu says 'menuopen' when it opens, to set that.

/** Close all the menus. */
export function menuClose() {
  document.querySelectorAll('.menu.open').forEach(function(m) { m.classList.remove('open'); });
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

/** Make the menus of the page work: the titles (`.menu-title`) that are there now, and the closing by a click elsewhere. Also `window.menuToggle` and `window.menuClose` for the handlers in markup. */
export function installMenus() {
  // (a title that has an action of its own, such as the ? of the shortcuts, opens no menu)
  document.querySelectorAll('.menu-title:not([data-click])').forEach(function(btn) {
    btn.addEventListener('click', function(event) { menuToggle(event, btn); });
    btn.addEventListener('mouseenter', function() {
      if (document.querySelector('.menu.open')) { menuClose(); menuOpen(btn.closest('.menu')); }
    });
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
