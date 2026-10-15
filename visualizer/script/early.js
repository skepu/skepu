// early.js — a classic script for the <head> of a page: puts the page in dark mode before it is painted, if the dark mode is what is remembered
// (the entry of appearance.js), so that there is no flash of the light page. (A module is run after the page is parsed: too late for this.)
(function() {
  // (the panes are put in place without sliding until the first trace has been drawn: ui.js takes the class away)
  document.documentElement.classList.add('no-pane-anim');
  // The splash (the welcome panel) is for a blank session, a new tab. Where a trace is on its way (a link to an example, files kept for this tab) or
  // the editor has its own (a VS Code webview, which has acquireVsCodeApi), there is no splash: the panel is blank until the first trace has been drawn
  // (the card is hidden: main.css), and fades out then as it would (reveal.js). worker-loader.js takes the class away if the trace does not come.
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
