/*
 * VizUtils — mounts the access-pattern visualizers of the registry (skeletons.js) into <div data-skeleton-viz="map,reduce">,
 * which the shortcode skeleton-viz (skevu.lua) writes.
 */
window.VizUtils = (function () {
  const MAX_DIM = 12;

  /* labelFn(r, c) overrides the default label: the index for a vector, "r,c" for a matrix. */
  function buildMatrix(el, rows, cols, labelFn, onEnter, onLeave) {
    el.dataset.rows = rows;
    el.dataset.cols = cols;
    el.style.gridTemplateColumns = `repeat(${cols}, var(--cell-size))`;
    el.innerHTML = "";
    const cells = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.r = r;
        cell.dataset.c = c;
        cell.textContent = labelFn ? labelFn(r, c)
          : rows === 1 ? String(c) : cols === 1 ? String(r) : `${r},${c}`;
        if (onEnter) cell.addEventListener("mouseenter", () => onEnter(r, c));
        if (onLeave) cell.addEventListener("mouseleave", onLeave);
        el.appendChild(cell);
        row.push(cell);
      }
      cells.push(row);
    }
    return cells;
  }

  /* `wrapperClass`: viz-vec-group, for the vectors of MapPairs. A label that starts with OUT or INTERMEDIATE marks the matrix as hoverable (data-role). */
  function makeMatrixGroup(labelText, wrapperClass) {
    const group = document.createElement("div");
    if (wrapperClass) group.className = wrapperClass;
    const lbl = document.createElement("div");
    lbl.className = "matrix-label";
    lbl.textContent = labelText;
    const mat = document.createElement("div");
    mat.className = "matrix";
    if (/^OUT/.test(labelText)) mat.dataset.role = "out";
    else if (/^INTERMEDIATE/.test(labelText)) mat.dataset.role = "inter";
    group.appendChild(lbl);
    group.appendChild(mat);
    return { group, mat };
  }

  function stack(areaEl, groups, className = "viz-stack") {
    const container = document.createElement("div");
    container.className = className;
    groups.forEach(g => container.appendChild(g));
    areaEl.innerHTML = "";
    areaEl.appendChild(container);
  }

  /* entries: [{ bg, border, label }] */
  function buildLegend(entries, el) {
    el.innerHTML = "";
    (entries || []).forEach(({ bg, border, label }) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      const swatch = document.createElement("div");
      swatch.className = "legend-swatch";
      swatch.style.background = bg;
      swatch.style.borderColor = border;
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(label));
      el.appendChild(item);
    });
  }

  /* textContent: the <T> of template arguments is not parsed as HTML. */
  function setCode(el, text) {
    el.innerHTML = "";
    const pre = document.createElement("pre");
    pre.className = "viz-code";
    const code = document.createElement("code");
    code.textContent = text;
    pre.appendChild(code);
    el.appendChild(pre);
  }

  /* min and max are the attributes of the input; `fallback` if it is empty. */
  function clampedInt(el, fallback) {
    const val = parseInt(el.value);
    const min = parseInt(el.min);
    const max = parseInt(el.max);
    const v = isNaN(val) ? fallback : val;
    return Math.max(isNaN(min) ? -Infinity : min, Math.min(isNaN(max) ? Infinity : max, v));
  }

  function element(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
  }

  function labelled(text, control) {
    const label = element("label");
    label.appendChild(element("span", null, text));
    label.appendChild(control);
    return label;
  }

  function numberInput(min, max, value) {
    const input = element("input");
    input.type = "number";
    input.min = min;
    input.max = max;
    input.value = value;
    return input;
  }

  function selectOf(options, value) {
    const sel = element("select");
    options.forEach(o => {
      const opt = new Option(o.label, o.value);
      opt.selected = o.value === value;
      sel.appendChild(opt);
    });
    return sel;
  }

  function section(title, collapsed) {
    const root = element("div", "viz-section");
    const bar = element("button", "viz-section-bar");
    const titleEl = element("span", "viz-section-title", title);
    bar.appendChild(titleEl);
    bar.appendChild(element("span", "viz-chevron" + (collapsed ? " viz-chevron-collapsed" : ""), "▾"));
    const body = element("div", "viz-section-body");
    if (collapsed) body.style.display = "none";
    root.appendChild(bar);
    root.appendChild(body);
    return { root, titleEl, body };
  }

  /* The controls of a skeleton: its own, with the number of outputs first and the proxy and the uniforms last, as in the call. */
  function controlsOf(sk) {
    return [...(sk.outputs ? [SkePUOperands.outputsControl] : []), ...(sk.extraControls || []), ...(sk.operands ? SkePUOperands.controls : [])];
  }

  /* What a call has besides what the skeleton draws: more outputs (copies of the first, hovered together with it), a proxy and the uniforms.
   * The reads of the proxy and the uniforms follow the hovered output; see hover(). */
  function addOperands(areaEl, p) {
    const ops = areaEl._ops = { clones: [], proxy: null, uniforms: null };
    const primary = areaEl.querySelector('.matrix[data-role="out"]');
    if (primary && p.outputs > 1) {
      const label = primary.previousElementSibling;
      label.textContent = label.textContent.replace(/^OUT/, "OUT 1");
      primary.parentElement.classList.add("viz-multi");
      for (let k = 2; k <= p.outputs; k++) {
        const lbl = element("div", "matrix-label", `OUT ${k}`);
        const mat = element("div", "matrix");
        mat.dataset.clone = "1";
        mat.dataset.role = "out";
        const cells = buildMatrix(mat, +primary.dataset.rows, +primary.dataset.cols, (r, c) => primary.children[r * primary.dataset.cols + c].textContent);
        primary.parentElement.append(lbl, mat);
        ops.clones.push(cells);
      }
    }
    const groups = [];
    const proxy = SkePUOperands.proxies[p.proxy];
    if (proxy) {
      const { group, mat } = makeMatrixGroup(`PROXY — ${proxy.label}`);
      ops.proxy = { spec: proxy, cells: buildMatrix(mat, proxy.rows, proxy.cols, proxy.rows === 1 ? (r, c) => `p${c}` : null) };
      groups.push(group);
    }
    if (p.uniforms > 0) {
      const { group, mat } = makeMatrixGroup(p.uniforms === 1 ? "UNIFORM" : "UNIFORMS");
      const row = element("div", "viz-uniforms");
      ops.uniforms = [];
      for (let i = 1; i <= p.uniforms; i++) {
        const one = element("div", "matrix");
        ops.uniforms.push(buildMatrix(one, 1, 1, () => `u${i}`)[0][0]);
        row.appendChild(one);
      }
      mat.replaceWith(row);
      groups.push(group);
    }
    if (groups.length) {
      const host = areaEl.querySelector(".viz-stack") || areaEl.appendChild(element("div", "viz-stack"));
      groups.forEach(g => host.appendChild(g));
    }
  }

  /* The hovered cell `cell` of an output: the copies of it in the other outputs, and what the function may read of the proxy and the uniforms. */
  function hover(areaEl, cell, on) {
    const ops = areaEl._ops;
    const mat = cell.parentElement;
    if (!ops || !mat.dataset.role) return;
    const r = +cell.dataset.r, c = +cell.dataset.c;
    ops.clones.forEach(cells => cells[r][c].classList.toggle("output-hover", on));
    if (mat.dataset.clone) {
      const primary = areaEl.querySelector('.matrix[data-role="out"]:not([data-clone])');
      primary.children[r * primary.dataset.cols + c].dispatchEvent(new MouseEvent(on ? "mouseenter" : "mouseleave"));
    }
    if (ops.proxy) {
      const { spec, cells } = ops.proxy;
      cells.forEach((row, i) => row.forEach((el, j) =>
        el.classList.toggle("highlight-proxy", on && (spec.access === "all" || (spec.access === "row" ? i === r % spec.rows : j === c % spec.cols)))));
    }
    if (ops.uniforms) ops.uniforms.forEach(el => el.classList.toggle("highlight-uniform", on));
  }

  /* `ids` are of SkePUEntries; several give a tab bar. */
  function mount(root, ids) {
    const entries = ids.map(id => {
      const e = SkePUEntries.find(x => x.id === id);
      if (!e) throw new Error(`skeleton-viz: no skeleton "${id}"`);
      return e;
    });
    const tabs = entries.length > 1;

    root.className = "viz-wrap";
    root.innerHTML = "";
    const tabBar = element("div", "skeleton-select");
    const controls = element("div", "controls");
    const rowsEl = numberInput(1, MAX_DIM, 1);
    const colsEl = numberInput(1, MAX_DIM, 1);
    const extraEl = element("span");
    controls.append(labelled("Rows", rowsEl), labelled("Cols", colsEl), extraEl);
    const main = section("", false);
    const descEl = element("p", "desc");
    const areaEl = element("div");
    const legendEl = element("div", "legend");
    main.body.append(descEl, areaEl, legendEl);
    const source = section("Source", true);

    if (tabs) root.appendChild(tabBar);
    root.append(controls, main.root, source.root);

    let entry, key;
    let extras = {};   // control id → element

    function params() {
      const sk = SkePUSkeletons[key];
      const p = { rows: clampedInt(rowsEl, sk.defaults.rows), cols: clampedInt(colsEl, sk.defaults.cols) };
      for (const ctrl of controlsOf(sk)) {
        const el = extras[ctrl.id];
        p[ctrl.id] = ctrl.type === "number" ? clampedInt(el, ctrl.default) : el.value;
      }
      return p;
    }

    function draw() {
      const sk = SkePUSkeletons[key];
      const p = params();
      main.titleEl.textContent = sk.name;
      descEl.innerHTML = typeof sk.desc === "function" ? sk.desc(p) : sk.desc;
      sk.render(areaEl, p);
      addOperands(areaEl, p);
      buildLegend([...sk.legend, ...(SkePUOperands.proxies[p.proxy] ? [SkePUOperands.proxyLegend] : []), ...(p.uniforms > 0 ? [SkePUOperands.uniformLegend] : [])], legendEl);
      if (sk.codeGen) setCode(source.body, sk.codeGen(p));
    }

    function buildExtras() {
      const sk = SkePUSkeletons[key];
      extraEl.innerHTML = "";
      extras = {};
      if (entry.variants) {
        const sel = selectOf(entry.variants.map(v => ({ value: v.key, label: v.name })), key);
        sel.addEventListener("change", () => { key = sel.value; buildExtras(); draw(); });
        extraEl.appendChild(labelled("Mode", sel));
      }
      for (const ctrl of controlsOf(sk)) {
        const dflt = sk.defaults[ctrl.id] ?? ctrl.default;
        const el = ctrl.type === "number"
          ? numberInput(ctrl.min, ctrl.max, dflt)
          : selectOf(ctrl.options, dflt);
        el.addEventListener("change", () => {
          if (ctrl.type === "number") el.value = clampedInt(el, ctrl.default);
          draw();
        });
        extras[ctrl.id] = el;
        extraEl.appendChild(labelled(ctrl.label || ctrl.id, el));
      }
    }

    function select(e) {
      entry = e;
      key = e.variants ? e.variants[0].key : e.key;
      const sk = SkePUSkeletons[key];
      rowsEl.value = sk.defaults.rows;
      colsEl.value = sk.defaults.cols;
      tabBar.querySelectorAll(".sk-btn").forEach(b => b.classList.toggle("active", b.dataset.id === e.id));
      buildExtras();
      draw();
    }

    for (const e of entries) {
      const btn = element("button", "sk-btn", e.variants ? e.name : SkePUSkeletons[e.key].name);
      btn.dataset.id = e.id;
      btn.addEventListener("click", () => select(e));
      tabBar.appendChild(btn);
    }
    areaEl.addEventListener("mouseover", e => { const cell = e.target.closest(".cell"); if (cell) hover(areaEl, cell, true); });
    areaEl.addEventListener("mouseout", e => { const cell = e.target.closest(".cell"); if (cell) hover(areaEl, cell, false); });
    rowsEl.addEventListener("change", draw);
    colsEl.addEventListener("change", draw);
    select(entries[0]);
  }

  document.addEventListener("click", e => {
    const bar = e.target.closest(".viz-section-bar");
    if (!bar || !bar.nextElementSibling) return;
    const body = bar.nextElementSibling;
    const hidden = body.style.display === "none";
    body.style.display = hidden ? "" : "none";
    bar.querySelector(".viz-chevron").classList.toggle("viz-chevron-collapsed", !hidden);
  });

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-skeleton-viz]").forEach(el => {
      const ids = el.dataset.skeletonViz === "all"
        ? SkePUEntries.map(e => e.id)
        : el.dataset.skeletonViz.split(",").map(s => s.trim());
      mount(el, ids);
    });
  });

  return { buildMatrix, makeMatrixGroup, stack, buildLegend, setCode, mount };
}());
