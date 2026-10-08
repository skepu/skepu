// appearance.js — what makes the look of SkeVU and of the designer the same: the dark mode, the theme of the code, and the colour spectrum, as the
// View and Colors menus offer them and as the browser remembers them (one entry of localStorage for both pages), and the menus themselves.

import { COLOR_SPECTRUMS } from './TraceDataProvider.js';

export const SETTINGS_KEY = 'skepu-visualizer-settings-v1';

// ── The themes of the code ───────────────────────────────────────────────────

/** The themes of the code that there are (static/highlight/styles/<id>.css); `dark`: for a dark background. */
export const HLJS_THEMES = [
  { id: 'default',          label: 'Default (light)',        dark: false },
  { id: 'github',           label: 'GitHub (light)',         dark: false },
  { id: 'intellij-light',   label: 'IntelliJ (light)',       dark: false },
  { id: 'vs',               label: 'Visual Studio (light)',  dark: false },
  { id: 'atom-one-dark',    label: 'Atom One Dark',          dark: true },
  { id: 'github-dark',      label: 'GitHub Dark',            dark: true },
  { id: 'monokai',          label: 'Monokai',                dark: true },
  { id: 'nord',             label: 'Nord',                   dark: true },
  { id: 'tokyo-night-dark', label: 'Tokyo Night',            dark: true },
  { id: 'vs2015',           label: 'VS 2015 (dark)',         dark: true },
];
export const HLJS_THEME_DEFAULT_LIGHT = 'default';
export const HLJS_THEME_DEFAULT_DARK  = 'tokyo-night-dark';

export function isDarkTheme(id) { return HLJS_THEMES.some(function(t) { return t.id === id && t.dark; }); }
/** The theme if there is one of that name, the default otherwise (a theme that is remembered may be one that was taken away). */
export function knownTheme(id) { return HLJS_THEMES.some(function(t) { return t.id === id; }) ? id : HLJS_THEME_DEFAULT_LIGHT; }

// ── The spectra of the colours ───────────────────────────────────────────────

const SPECTRUM_LABELS = {
  'default': 'Red–Green', 'viridis': 'Viridis', 'plasma': 'Plasma', 'inferno': 'Inferno',
  'cividis': 'Cividis', 'cool-warm': 'Cool–Warm', 'grayscale': 'Grayscale', 'rainbow': 'Rainbow',
};

function spectrumGradient(stops) {
  return 'linear-gradient(to right,' + stops.map(function(c) { return 'rgb(' + c.join(',') + ')'; }).join(',') + ')';
}

// ── What is remembered ───────────────────────────────────────────────────────

function readStored() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}; } catch (e) { return {}; }
}

/** The dark mode, the theme of the code and the spectrum of the colours that are remembered ({ darkMode, hljsTheme, colorSpectrum }). */
export function readAppearance() {
  var layout = readStored().layout || {};
  return {
    darkMode:      !!layout['dark-mode'],
    hljsTheme:     knownTheme(layout['hljs-theme']),
    colorSpectrum: COLOR_SPECTRUMS[layout['color-spectrum']] ? layout['color-spectrum'] : 'default',
  };
}

/** Remember some of them (the others, and the rest of what SkeVU remembers, stay as they are). */
export function writeAppearance(patch) {
  var s = readStored();
  s.layout = s.layout || {};
  if ('darkMode'      in patch) s.layout['dark-mode']      = !!patch.darkMode;
  if ('hljsTheme'     in patch) s.layout['hljs-theme']     = patch.hljsTheme;
  if ('colorSpectrum' in patch) s.layout['color-spectrum'] = patch.colorSpectrum;
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch (e) { /* private window: it is not remembered */ }
}

/** Turn the dark mode on or off in the page. */
export function showDarkMode(on) { document.documentElement.classList.toggle('dark-mode', !!on); }

// ── The menus ────────────────────────────────────────────────────────────────

function item(className, text, data) {
  var b = document.createElement('button');
  b.className = className;
  Object.keys(data).forEach(function(k) { b.dataset[k] = data[k]; });
  var icon = document.createElement('span');
  icon.className = 'menu-icon';
  b.append(icon, text);
  return b;
}

/**
 * Fill `container` with the items of the Source Code Theme menu (`.hljs-theme-radio`, which SourceCodeView marks): the light themes, a line, the
 * dark ones. `onPick(id)` is called when one is chosen.
 */
export function buildThemeMenu(container, onPick) {
  container.replaceChildren();
  HLJS_THEMES.forEach(function(theme, i) {
    if (i > 0 && theme.dark !== HLJS_THEMES[i - 1].dark) {
      var sep = document.createElement('div');
      sep.className = 'menu-sep';
      container.append(sep);
    }
    var b = item('hljs-theme-radio', theme.label, { theme: theme.id });
    b.onclick = function() { onPick(theme.id); };
    container.append(b);
  });
}

/** Fill `container` with the items of the Colors menu (`.spectrum-radio`, with a swatch each). `onPick(name)` is called when one is chosen. */
export function buildSpectrumMenu(container, onPick) {
  container.replaceChildren();
  Object.keys(COLOR_SPECTRUMS).forEach(function(name) {
    var b = item('spectrum-radio', (SPECTRUM_LABELS[name] || name) + ' ', { spectrum: name });
    var swatch = document.createElement('span');
    swatch.className = 'spectrum-swatch';
    swatch.style.background = spectrumGradient(COLOR_SPECTRUMS[name].stops);
    b.append(swatch);
    b.onclick = function() { onPick(name); };
    container.append(b);
  });
}

/** Mark the item of a menu whose `data-<key>` is `value` (a check by it, none by the others). */
export function markChoice(selector, key, value) {
  document.querySelectorAll(selector).forEach(function(b) {
    var icon = b.querySelector('.menu-icon');
    if (icon) icon.textContent = b.dataset[key] === value ? '✓' : '';
  });
}
