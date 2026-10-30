// actions.js — the controls of the page in markup, without inline handlers (a page with a content security policy, such as an editor's webview, does
// not run them). `data-click="a b"` says what a click does: actions a and b, in order; an element with a value says what a change does with
// `data-input="a"` (the action receives the value). An action is a function of the page (`window.name`, made by the page's modules for the markup) or
// one of the few below. `bindActions(root)` binds the elements of root (the document, or markup just made); an element is bound once.
import { menuClose } from './menu.js';

/** Actions that are not functions of the page by their name. The element and the event are the arguments. */
export const ACTIONS = {
  // the click is not for the elements around: a menu stays open, as it is a click inside a field of it
  stop: function(el, event) { event.stopPropagation(); },
  menuClose: function() { menuClose(); },
  // the settings of a control: the control is the argument
  menuSelectRadio: function(el) { window.menuSelectRadio(el); },
  menuToggleCheck: function(el) { window.menuToggleCheck(el); },
  loadFilesFromForm: function() { window.loadFiles(null); },
  loadExample: function(el) { window.loadExample(Number(el.dataset.example), el.dataset.backend); },
  nextSourceFile: function() {
    var tabs = Array.from(document.querySelectorAll('.cpp-tab'));
    if (tabs.length < 2) return;
    var cur  = tabs.findIndex(function(t) { return t.classList.contains('active'); });
    var next = (cur + 1) % tabs.length;
    window.switchTab(tabs[next], next);
  },
};

function run(names, el, event, withValue) {
  names.split(/\s+/).filter(Boolean).forEach(function(name) {
    var own = ACTIONS[name], fn = own || window[name];
    if (typeof fn !== 'function') { console.error('actions: there is no action ' + name); return; }
    // A function of the page is called with no arguments, or with the value of the control.
    if (withValue) fn(el.value); else if (own) own(el, event); else fn();
  });
}

var _bound = new WeakSet();

export function bindActions(root) {
  (root || document).querySelectorAll('[data-click], [data-input]').forEach(function(el) {
    if (_bound.has(el)) return;
    _bound.add(el);
    if (el.dataset.click !== undefined) el.addEventListener('click', function(event) { run(el.dataset.click, el, event, false); });
    if (el.dataset.input !== undefined) el.addEventListener('input', function(event) { run(el.dataset.input, el, event, true); });
  });
}
