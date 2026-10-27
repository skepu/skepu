// test_menu.js — the menus of the menu bar open when the button is pressed on a title; releasing the button over an item activates it, on the title
// keeps the menu open, and outside the menus closes them.
import { describe, it, expect } from 'vitest';
import { releaseOutcome } from '../../script/menu.js';

describe('releasing the button after a press on a title of a menu', () => {
    it('activates an item that it is over', () => expect(releaseOutcome('item')).toBe('activate'));
    it('keeps the menu open on a title (a click) and on the rest of a menu', () => {
        expect(releaseOutcome('title')).toBe('keep');
        expect(releaseOutcome('menu')).toBe('keep');
    });
    it('closes the menus outside them', () => expect(releaseOutcome('outside')).toBe('close'));
});
