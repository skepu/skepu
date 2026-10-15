// theme_core.js — the theme of the editor (VS Code) as the viewer needs it. No DOM, so that it can be tested (tests/unit/test_theme_vscode.js).

/** Whether a theme is dark, from the classes of the body of a webview (vscode-dark, vscode-light, vscode-high-contrast, vscode-high-contrast-light). */
export function isDarkTheme(classes) {
  const has = c => classes.contains(c);
  if (has('vscode-light') || has('vscode-high-contrast-light')) return false;
  return has('vscode-dark') || has('vscode-high-contrast');
}
