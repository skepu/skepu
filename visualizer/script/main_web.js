// main_web.js — the start of the page in a browser: the controls of the markup (data-click), and the first load once the page is there.
import { bindActions } from './actions.js';
import { load } from './ui.js';

bindActions();
window.addEventListener('load', function() { load(); });
