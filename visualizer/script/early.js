// early.js — a classic script for the <head> of a page: puts the page in dark mode before it is painted, if the dark mode is what is remembered
// (the entry of appearance.js), so that there is no flash of the light page. (A module is run after the page is parsed: too late for this.)
(function() {
  // (the panes are put in place without sliding until the first trace has been drawn: ui.js takes the class away)
  document.documentElement.classList.add('no-pane-anim');
  // The splash is for a blank session (a new tab). With a trace on its way (a link to an example, files kept for this tab) or in an editor
  // (VS Code webview) there is none: the panel stays blank until the first trace is drawn, then fades (reveal.js). worker-loader.js
  // removes the class if no trace comes.
  try {
    if (typeof acquireVsCodeApi === 'function' || /[#&]example=/.test(location.hash) || /[?&]example=/.test(location.search)
        || sessionStorage.getItem('skevu-upload-kept') === '1')
      document.documentElement.classList.add('no-splash');
  } catch (e) { /* (no storage: the splash) */ }
  try {
    var s = JSON.parse(localStorage.getItem('skepu-visualizer-settings-v1'));
    if (s && s.layout && s.layout['dark-mode']) document.documentElement.classList.add('dark-mode');
  } catch (e) { /* nothing remembered, or not readable: the light page */ }
})();
