// shortcuts.js — the keyboard shortcuts of the viewer in a dialog (the (?) in the menu bar, or the key ?). The list is made from the menus, where an
// item has its key (`.menu-key`): the dialog cannot be out of date with them; what the menus do not say (keys without an item, the mouse) is below.
'use strict';

/** What the keys do that no menu item says, and the mouse and the trackpad. Groups of { keys, label }; `handles` names the keys (event.key) that the page handles for it. */
export const EXTRA_SHORTCUTS = [
  { group: 'General', keys: '?', label: 'Show shortcuts', handles: ['?'] },
  { group: 'General', keys: 'Esc', label: 'Close dialog or pop-up', handles: ['Escape'] },
  { group: 'General', keys: 'X', label: 'Close pop-up', handles: ['x'] },
  { group: 'Actions', keys: 'Shift+Tab', label: 'Previous source tab', handles: [], only: 'web' },
  { group: 'Actions', keys: '←, →', label: 'Previous / next simulation step or heat map iteration', handles: ['ArrowLeft', 'ArrowRight'] },
  { group: 'Mouse and trackpad', keys: 'Scroll, swipe', label: 'Pan' },
  { group: 'Mouse and trackpad', keys: 'Shift+scroll', label: 'Pan sideways' },
  { group: 'Mouse and trackpad', keys: 'Ctrl or ⌘+scroll, pinch', label: 'Zoom' },
  { group: 'Mouse and trackpad', keys: 'Click a node', label: 'Show in info pane, timeline and source' },
  { group: 'Mouse and trackpad', keys: 'Shift+click a call', label: 'Also show the skeleton instance' },
  { group: 'Mouse and trackpad', keys: 'Alt+click a call', label: 'Also show a user function (repeat: the next)' },
  { group: 'Mouse and trackpad', keys: 'Right-click a node', label: 'Context menu' },
];

/** Groups of items, in the order of their first appearance: [{ group, entries: [{ keys, label }] }]. */
export function groupShortcuts(items) {
  const groups = [];
  for (const item of items) {
    let g = groups.find(x => x.group === item.group);
    if (!g) { g = { group: item.group, entries: [] }; groups.push(g); }
    g.entries.push({ keys: item.keys, label: item.label });
  }
  return groups;
}

function text(el) {
  const clone = el.cloneNode(true);
  clone.querySelectorAll('.menu-key, .menu-icon').forEach(function(n) { n.remove(); });
  return clone.textContent.replace(/\s+/g, ' ').trim();
}

/** The shortcuts of the menus of the page (a menu is a group), and the extra ones. */
export function collectShortcuts(root, editor) {
  const variant = (editor === undefined ? document.documentElement.classList.contains('editor') : editor) ? 'editor' : 'web';
  const items = [];
  (root || document).querySelectorAll('.menu-key').forEach(function(key) {
    const button = key.closest('button');
    const menu = key.closest('.menu');
    const title = menu && menu.querySelector(':scope > .menu-title');
    if (!button || !title) return;
    const only = button.closest('[data-only]');   // (an item for the browser, or for the editor, only)
    if (only && only.dataset.only !== variant) return;
    items.push({ group: text(title), keys: key.textContent.trim(), label: text(button) });
  });
  return groupShortcuts(items.concat(EXTRA_SHORTCUTS.filter(function(e) { return !e.only || e.only === variant; })));
}

// ── The dialog ────────────────────────────────────────────────────────────────

let _previousFocus = null;

function build(modal) {
  const body = modal.querySelector('.shortcuts-body');
  body.textContent = '';
  collectShortcuts().forEach(function(g) {
    const section = document.createElement('section');
    const h = document.createElement('h3');
    h.textContent = g.group;
    section.appendChild(h);
    const dl = document.createElement('dl');
    g.entries.forEach(function(e) {
      const dt = document.createElement('dt');
      e.keys.split(/,\s*/).forEach(function(k, i) {
        if (i) dt.appendChild(document.createTextNode(' '));
        const kbd = document.createElement('kbd');
        kbd.textContent = k;
        dt.appendChild(kbd);
      });
      const dd = document.createElement('dd');
      dd.textContent = e.label;
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    section.appendChild(dl);
    body.appendChild(section);
  });
}

function modalEl() { return document.getElementById('shortcuts-modal'); }

export function shortcutsOpen() {
  const m = modalEl();
  return !!m && !m.hidden;
}

export function showShortcuts() {
  const m = modalEl();
  if (!m || !m.hidden) return;
  if (typeof window.menuClose === 'function') window.menuClose();
  build(m);
  _previousFocus = document.activeElement;
  m.hidden = false;
  const close = m.querySelector('.shortcuts-close');
  if (close) close.focus();
}

export function hideShortcuts() {
  const m = modalEl();
  if (!m || m.hidden) return;
  m.hidden = true;
  if (_previousFocus && typeof _previousFocus.focus === 'function') _previousFocus.focus();
  _previousFocus = null;
}

export function toggleShortcuts() {
  if (shortcutsOpen()) hideShortcuts(); else showShortcuts();
}

/** Closing: the backdrop, Esc, and the button. (Call once, when the page is made.) */
export function installShortcuts() {
  const m = modalEl();
  if (!m) return;
  m.addEventListener('click', function(event) { if (event.target === m) hideShortcuts(); });
  window.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && shortcutsOpen()) { event.preventDefault(); hideShortcuts(); }
  });
  window.showShortcuts   = showShortcuts;
  window.hideShortcuts   = hideShortcuts;
  window.toggleShortcuts = toggleShortcuts;
}
