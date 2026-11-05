// timeline/zoombar.js — vertical logarithmic zoom slider.
'use strict';

export const MIN_SCALE = 1;
export const MAX_SCALE = 1000;
const KEY_FACTOR = 1.25;

/** Position on the strip (0: bottom, 1: top) of a scale, and back. */
export const scaleToFraction = (scale, min = MIN_SCALE, max = MAX_SCALE) => Math.log(Math.min(max, Math.max(min, scale)) / min) / Math.log(max / min);
export const fractionToScale = (f, min = MIN_SCALE, max = MAX_SCALE) => min * Math.pow(max / min, Math.min(1, Math.max(0, f)));

/** The scale as text: "1.0×", "12×", "1000×". */
export const formatScale = scale => (scale < 10 ? scale.toFixed(1) : Math.round(scale)) + '×';

export class ZoomBar {
    /** `el` gets the label (above the strip: a rounded window corner would cut it below), the track and the thumb. */
    constructor(el, { onChange = () => {}, min = MIN_SCALE, max = MAX_SCALE } = {}) {
        this._el = el; this._min = min; this._max = max; this._onChange = onChange; this._value = min;
        el.classList.add('tl-zoom');
        el.setAttribute('role', 'slider');
        el.setAttribute('tabindex', '0');
        el.setAttribute('aria-label', 'Timeline zoom');
        el.setAttribute('aria-valuemin', min);
        el.setAttribute('aria-valuemax', max);
        el.title = 'Zoom (ctrl + wheel, double-click for 1×)';
        el.innerHTML = '<div class="tl-zoom-text"></div><div class="tl-zoom-track"><div class="tl-zoom-fill"></div><div class="tl-zoom-thumb"></div></div>';
        this._track = el.querySelector('.tl-zoom-track');
        this._thumb = el.querySelector('.tl-zoom-thumb');
        this._fill = el.querySelector('.tl-zoom-fill');
        this._text = el.querySelector('.tl-zoom-text');

        const fromPointer = ev => {
            const r = this._track.getBoundingClientRect();
            this._move(fractionToScale(1 - (ev.clientY - r.top) / r.height, this._min, this._max));
        };
        el.addEventListener('pointerdown', ev => {
            if (ev.button !== 0) return;
            el.setPointerCapture(ev.pointerId);
            this._dragging = true;
            fromPointer(ev);
            ev.preventDefault();
        });
        el.addEventListener('pointermove', ev => { if (this._dragging) fromPointer(ev); });
        const stop = () => { this._dragging = false; };
        el.addEventListener('pointerup', stop);
        el.addEventListener('pointercancel', stop);
        el.addEventListener('dblclick', () => this._move(this._min));
        el.addEventListener('keydown', ev => {
            const to = { ArrowUp: this._value * KEY_FACTOR, ArrowRight: this._value * KEY_FACTOR, ArrowDown: this._value / KEY_FACTOR, ArrowLeft: this._value / KEY_FACTOR,
                         Home: this._min, End: this._max }[ev.key];
            if (to === undefined) return;
            ev.preventDefault();
            this._move(to);
        });
        this.setValue(min);
    }

    get value() { return this._value; }

    /** Shows `scale` without telling anyone. */
    setValue(scale) {
        this._value = Math.min(this._max, Math.max(this._min, scale));
        const percent = scaleToFraction(this._value, this._min, this._max) * 100;
        this._thumb.style.bottom = percent + '%';
        this._fill.style.height = percent + '%';
        this._text.textContent = formatScale(this._value);
        this._el.setAttribute('aria-valuenow', this._value.toFixed(2));
    }

    _move(scale) {
        const before = this._value;
        this.setValue(scale);
        if (this._value !== before) this._onChange(this._value);
    }
}
