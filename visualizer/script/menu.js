// menu.js — the menu bar menus, for SkeVU and the designer. Pressing a title opens its menu and closes the others; with a menu open, hovering another title
// opens that one; a click elsewhere closes them. Releasing the button over an item, after pressing on a title, activates the item; releasing on the title
// keeps the menu open. A click in a `.menu-field` (a text entry) keeps the menu open. An item that does nothing is `disabled` (greyed by main.css); a menu
// fires 'menuopen' when it opens, so that items can be set.
//
// When the titles do not fit the bar, the ones at the right collapse into a "…" menu, each as a submenu with the same items (the items are moved, not copied, so
// that they keep their state and their handlers). The logo, the ? and the ⓘ stay (the classes menu-anfang, menu-help-btn and menu-info-btn, and the render
// indicator), as does any menu that is not a title with a dropdown.

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

/**
 * How many of the titles, from the right, collapse into the "…" menu. `widths` are those of the titles that can collapse, left to right; `fixed` is the width of
 * everything else in the bar; `more` is the width of the "…" title, which is there when any collapses.
 */
export function collapseCount(widths, fixed, more, available) {
  var sum = function(list) { return list.reduce(function(a, b) { return a + b; }, 0); };
  if (fixed + sum(widths) <= available) return 0;
  for (var n = 1; n < widths.length; n++)
    if (fixed + more + sum(widths.slice(0, widths.length - n)) <= available) return n;
  return widths.length;
}

var FIXED_MENUS = '.menu-help-btn, .menu-info-btn, #render-indicator';

/** The menus of the bar that can collapse: a title with a dropdown, not the logo, the ?, the ⓘ and the render indicator. */
function collapsibleMenus(nav) {
  return Array.prototype.filter.call(nav.children, function(m) {
    return m.classList.contains('menu') && !m.matches(FIXED_MENUS) && !m.classList.contains('menu-overflow')
      && m.querySelector(':scope > .menu-title') && m.querySelector(':scope > .dropdown');
  });
}

/** Collapse the titles of `nav` that do not fit into a "…" menu, and keep doing so as the bar changes size. */
export function installMenuOverflow(nav) {
  if (!nav || nav.querySelector('.menu-overflow')) return;
  var menus = collapsibleMenus(nav);
  if (!menus.length) return;
  var more = document.createElement('div');
  more.className = 'menu menu-overflow';
  more.hidden = true;
  more.innerHTML = '<button class="menu-title" title="More menus" aria-label="More menus">…</button><div class="dropdown"></div>';
  menus[menus.length - 1].after(more);
  var list = more.querySelector('.dropdown');
  var moved = [];   // { menu, dropdown, submenu } of the collapsed titles, in the bar's order

  function restore() {
    moved.forEach(function(r) { r.menu.appendChild(r.dropdown); r.submenu.remove(); r.menu.hidden = false; });
    moved = [];
    more.hidden = true;
  }
  function collapse(menu) {
    var dropdown = menu.querySelector(':scope > .dropdown');
    var submenu = document.createElement('div');
    submenu.className = 'submenu';
    var title = document.createElement('div');
    title.className = 'submenu-title';
    title.innerHTML = '<span></span><span class="menu-arrow">▸</span>';
    title.firstChild.textContent = menu.querySelector(':scope > .menu-title').textContent.trim();
    submenu.appendChild(title);
    submenu.appendChild(dropdown);
    list.appendChild(submenu);
    menu.hidden = true;
    moved.push({ menu: menu, dropdown: dropdown, submenu: submenu });
  }

  var pending = false;
  function layout() {
    pending = false;
    menuClose();
    restore();
    var shown = menus.filter(function(m) { return m.offsetWidth > 0; });   // a menu that the page leaves out (data-only) is not counted
    var widths = shown.map(function(m) { return m.offsetWidth; });
    var total = Array.prototype.reduce.call(nav.children, function(a, c) { return a + c.offsetWidth; }, 0);
    var fixed = total - widths.reduce(function(a, b) { return a + b; }, 0);
    more.hidden = false;
    var moreWidth = more.offsetWidth;
    more.hidden = true;
    var n = collapseCount(widths, fixed, moreWidth, nav.clientWidth);
    if (n === 0) return;
    shown.slice(shown.length - n).forEach(collapse);
    more.hidden = false;
  }
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(layout);
  }

  // the items of a collapsed menu are set when it opens: the menu that had them says 'menuopen'
  more.addEventListener('menuopen', function() {
    more.classList.toggle('flip', more.getBoundingClientRect().left > window.innerWidth / 2);   // opens to the left where there is no room to the right
    moved.forEach(function(r) { r.menu.dispatchEvent(new CustomEvent('menuopen', { bubbles: true })); });
  });

  if (typeof ResizeObserver !== 'undefined') {
    var observer = new ResizeObserver(schedule);
    observer.observe(nav);
    Array.prototype.forEach.call(nav.querySelectorAll(FIXED_MENUS), function(m) { observer.observe(m); });
  }
  window.addEventListener('resize', schedule);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(schedule);
  schedule();
}

/** Make the menus of the page work: the titles (`.menu-title`) that are there now, and the closing by a click elsewhere. Also `window.menuToggle` and `window.menuClose` for the handlers in markup. */
export function installMenus() {
  installMenuOverflow(document.getElementById('menubar'));
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
