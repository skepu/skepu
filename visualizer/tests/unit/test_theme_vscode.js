// test_theme_vscode.js — the viewer in an editor is dark when the theme of VS Code is (script/theme_core.js).
import { describe, it, expect } from 'vitest';
import { isDarkTheme } from '../../script/theme_core.js';

const classes = (...names) => ({ contains: c => names.includes(c) });

describe('the theme of the editor', () => {
    it('is dark for the dark and the high contrast themes', () => {
        expect(isDarkTheme(classes('vscode-dark'))).toBe(true);
        expect(isDarkTheme(classes('vscode-high-contrast'))).toBe(true);
    });

    it('is light for the light themes, also the high contrast one that is light', () => {
        expect(isDarkTheme(classes('vscode-light'))).toBe(false);
        expect(isDarkTheme(classes('vscode-high-contrast', 'vscode-high-contrast-light'))).toBe(false);
    });

    it('is light when the body says nothing', () => {
        expect(isDarkTheme(classes())).toBe(false);
    });
});
