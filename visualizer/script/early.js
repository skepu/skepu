// early.js — a classic script for the <head> of a page: puts the page in dark mode before it is painted, if the dark mode is what is remembered
// (the entry of appearance.js), so that there is no flash of the light page. (A module is run after the page is parsed: too late for this.)
(function() {
  // (the panes are put in place without sliding until the first trace has been drawn: ui.js takes the class away)
  document.documentElement.classList.add('no-pane-anim');
  try {
    var s = JSON.parse(localStorage.getItem('skepu-visualizer-settings-v1'));
    if (s && s.layout && s.layout['dark-mode']) document.documentElement.classList.add('dark-mode');
  } catch (e) { /* nothing remembered, or not readable: the light page */ }
})();
