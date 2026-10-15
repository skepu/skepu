// theme_vscode.js — in an editor the viewer follows the theme of VS Code (the classes that the body of the webview has) and has no dark mode switch of its own.
import { setDarkMode } from './ui.js';
import { isDarkTheme } from './theme_core.js';

/** Puts the viewer in the theme of the editor now and whenever the editor's theme changes. */
export function followEditorTheme(body = document.body) {
  const apply = () => setDarkMode(isDarkTheme(body.classList));
  apply();
  new MutationObserver(apply).observe(body, { attributes: true, attributeFilter: ['class'] });
}
