/*
 * SkePUSkeletons — the registry of the access-pattern visualizers; skeleton-viz.js mounts them.
 *
 * An entry:
 *   name          — label of the tab and the section
 *   desc          — string, or function(params) => string for text that depends on a mode
 *   legend        — [{bg, border, label}]
 *   extraControls — [{type: "number", id, label, min, max, default} | {type: "toggle", id, label, options, default}], beyond rows and cols
 *   defaults      — {rows, cols, ...values of the extra controls}
 *   codeGen(params), render(areaEl, params) — the listing and the drawing; render rebuilds its DOM each call
 *
 * SkePUEntries (below) lists what has a tab: a skeleton, or a skeleton with variants that are entries of their own.
 */
window.SkePUSkeletons = {};

const { buildMatrix, makeMatrixGroup, stack } = VizUtils;

/* Produce a typed container declaration based on dimensions. */
function containerDecl(name, rows, cols) {
  return rows === 1
    ? `skepu::Vector<int> ${name}(${cols});`
    : `skepu::Matrix<int> ${name}(${rows}, ${cols});`;
}

/* The outputs, proxies and uniforms of a call, in the order of the signature: output(s), element-wise inputs, proxies, uniforms.
 * Returns the return type and extra parameters of the user function, the names of the outputs, and the declarations and arguments of the proxy and the uniforms. */
function operandCode({ outputs = 1, proxy = "none", uniforms = 0 } = {}) {
  const proxies = { vec: ["skepu::Vec<int>", "skepu::Vector<int> proxy(3);"], mat: ["skepu::Mat<int>", "skepu::Matrix<int> proxy(3, 3);"],
                    matrow: ["skepu::MatRow<int>", "skepu::Matrix<int> proxy(3, 3);"], matcol: ["skepu::MatCol<int>", "skepu::Matrix<int> proxy(3, 3);"] };
  const names = Array.from({ length: uniforms }, (_, i) => uniforms === 1 ? "u" : `u${i + 1}`);
  return {
    ret: outputs === 1 ? "int" : `skepu::multiple<${Array(outputs).fill("int").join(", ")}>`,
    retExpr: e => outputs === 1 ? e : `skepu::ret(${Array(outputs).fill(e).join(", ")})`,
    params: [...(proxy in proxies ? [`${proxies[proxy][0]} p`] : []), ...names.map(n => `int ${n}`)],
    outNames: Array.from({ length: outputs }, (_, i) => outputs === 1 ? "out" : `out${i + 1}`),
    decls: [...(proxy in proxies ? [proxies[proxy][1]] : []), ...names.map((n, i) => `int ${n} = ${i + 1};`)],
    args: [...(proxy in proxies ? ["proxy"] : []), ...names],
  };
}

/* Produce a C-style user function with int params and a dummy body. */
function funcDecl(name, paramNames, o = operandCode()) {
  const params = [...paramNames.map(p => `int ${p}`), ...o.params].join(", ");
  return `${o.ret} ${name}(${params}) { return ${o.retExpr(paramNames.join(" + "))}; }`;
}

/* ---- Map ---------------------------------------------------------------- */
SkePUSkeletons.map = {
  name: "Map",
  operands: true, outputs: true,
  codeGen(p) {
    const { rows, cols, numInputs: n = 2 } = p;
    const o = operandCode(p);
    const inNames  = Array.from({ length: n }, (_, i) => n === 1 ? "in" : `in${i + 1}`);
    const argNames = Array.from({ length: n }, (_, i) => String.fromCharCode(97 + i)); // a, b, c, d
    return [
      funcDecl("f", argNames, o),
      "",
      "auto skel = skepu::Map(f);",
      "",
      ...o.outNames.map(name => containerDecl(name, rows, cols)),
      ...inNames.map(name => containerDecl(name, rows, cols)),
      ...o.decls,
      "",
      `skel(${[...o.outNames, ...inNames, ...o.args].join(", ")});`,
    ].join("\n");
  },
  desc: "Each output cell reads exactly one cell at the same position from every input container. Multiple inputs are shown side by side.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output cell" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Direct read (same position)" },
  ],
  extraControls: [
    { type: "number", id: "numInputs", label: "Inputs", min: 1, max: 4, default: 2 },
  ],
  defaults: { rows: 4, cols: 6, numInputs: 2 },
  render(areaEl, { rows, cols, numInputs = 2 }) {
    let allInputCells = [], outputCells = [];

    function clearHighlights() {
      allInputCells.forEach(cells =>
        cells.forEach(row => row.forEach(c => c.classList.remove("highlight-primary"))));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(r, c) {
      clearHighlights();
      outputCells[r][c].classList.add("output-hover");
      allInputCells.forEach(cells => cells[r][c].classList.add("highlight-primary"));
    }

    const inputsRow = document.createElement("div");
    inputsRow.className = "viz-row";
    for (let i = 0; i < numInputs; i++) {
      const { group, mat } = makeMatrixGroup(numInputs === 1 ? "IN" : `IN ${i + 1}`);
      allInputCells.push(buildMatrix(mat, rows, cols, null, null, null));
      inputsRow.appendChild(group);
    }
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");
    outputCells = buildMatrix(outEl, rows, cols, null, onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, inputsRow]);
  },
};

/* ---- MapOverlap --------------------------------------------------------- */
SkePUSkeletons.mapoverlap = {
  name: "MapOverlap",
  operands: true, outputs: true,
  codeGen(p) {
    const { rows, cols } = p;
    const o = operandCode(p);
    const is1D   = rows === 1;
    const region = is1D ? "skepu::Region1D<int>" : "skepu::Region2D<int>";
    const center = is1D ? "r(0)" : "r(0, 0)";
    return [
      `${o.ret} f(${[`${region} r`, ...o.params].join(", ")}) { return ${o.retExpr(center)}; }`,
      "",
      "auto skel = skepu::MapOverlap(f);",
      "",
      ...o.outNames.map(name => containerDecl(name, rows, cols)),
      containerDecl("in", rows, cols),
      ...o.decls,
      "",
      `skel(${[...o.outNames, "in", ...o.args].join(", ")});`,
    ].join("\n");
  },
  desc: "Each output cell reads its input cell plus a surrounding halo. Row and column overlap radii are configurable independently; a radius of 0 disables overlap in that dimension.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output cell" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Direct read (same index)" },
    { bg: "#C0DD97", border: "#639922", label: "Halo read (overlap region)" },
  ],
  extraControls: [
    { type: "number", id: "rowRadius", label: "Row overlap", min: 0, max: 3, default: 1 },
    { type: "number", id: "colRadius", label: "Col overlap", min: 0, max: 3, default: 1 },
  ],
  defaults: { rows: 4, cols: 6, rowRadius: 1, colRadius: 1 },
  render(areaEl, { rows, cols, rowRadius = 1, colRadius = 1 }) {
    let inputCells = [], outputCells = [];

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c =>
        c.classList.remove("highlight-primary", "highlight-halo")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(outR, outC) {
      clearHighlights();
      outputCells[outR][outC].classList.add("output-hover");
      for (let dr = -rowRadius; dr <= rowRadius; dr++) {
        for (let dc = -colRadius; dc <= colRadius; dc++) {
          const nr = outR + dr, nc = outC + dc;
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
          inputCells[nr][nc].classList.add(
            dr === 0 && dc === 0 ? "highlight-primary" : "highlight-halo");
        }
      }
    }

    const { group: inGroup, mat: inEl }   = makeMatrixGroup("IN");
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");
    inputCells  = buildMatrix(inEl,  rows, cols, null, null, null);
    outputCells = buildMatrix(outEl, rows, cols, null, onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, inGroup]);
  },
};

/* ---- Reduce — full ------------------------------------------------------ */
SkePUSkeletons.reduce_full = {
  name: "Reduce — full",
  codeGen({ rows, cols }) {
    return [
      funcDecl("f", ["a", "b"]),
      "",
      "auto skel = skepu::Reduce(f);",
      "",
      containerDecl("in", rows, cols),
      "",
      "int out = skel(in);",
    ].join("\n");
  },
  desc: "Full reduction: every input element contributes to a single scalar output. Hover the output cell to see all reads.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Output scalar" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed element" },
  ],
  extraControls: [],
  defaults: { rows: 4, cols: 6 },
  render(areaEl, { rows, cols }) {
    let inputCells = [], outputCells = [];

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c => c.classList.remove("highlight-primary")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover() {
      clearHighlights();
      outputCells[0][0].classList.add("output-hover");
      inputCells.forEach(row => row.forEach(c => c.classList.add("highlight-primary")));
    }

    const { group: inGroup, mat: inEl }   = makeMatrixGroup("IN");
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");
    inputCells  = buildMatrix(inEl,  rows, cols, null,     null, null);
    outputCells = buildMatrix(outEl, 1,    1,    () => "Σ", onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, inGroup]);
  },
};

/* ---- Reduce — rowwise --------------------------------------------------- */
SkePUSkeletons.reduce_row = {
  name: "Reduce — rowwise",
  codeGen({ rows, cols }) {
    return [
      funcDecl("f", ["a", "b"]),
      "",
      "auto skel = skepu::Reduce(f);",
      "skel.setReduceMode(skepu::ReduceMode::RowWise);",
      "",
      containerDecl("in", rows, cols),
      `skepu::Vector<int> out(${rows});`,
      "",
      "skel(out, in);",
    ].join("\n");
  },
  desc: "Rowwise reduction: each input row is reduced to a single output element (N×1 column vector). Hover an output cell to see its contributing row.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output element" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed row" },
  ],
  extraControls: [],
  defaults: { rows: 4, cols: 6 },
  render(areaEl, { rows, cols }) {
    let inputCells = [], outputCells = [];

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c => c.classList.remove("highlight-primary")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(outR) {
      clearHighlights();
      outputCells[outR][0].classList.add("output-hover");
      for (let c = 0; c < cols; c++) inputCells[outR][c].classList.add("highlight-primary");
    }

    const { group: inGroup, mat: inEl }   = makeMatrixGroup("IN");
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");
    inputCells  = buildMatrix(inEl,  rows, cols, null,           null, null);
    outputCells = buildMatrix(outEl, rows, 1,    r => `r${r}`, onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, inGroup]);
  },
};

/* ---- Reduce — colwise --------------------------------------------------- */
SkePUSkeletons.reduce_col = {
  name: "Reduce — colwise",
  codeGen({ rows, cols }) {
    return [
      funcDecl("f", ["a", "b"]),
      "",
      "auto skel = skepu::Reduce(f);",
      "skel.setReduceMode(skepu::ReduceMode::ColWise);",
      "",
      containerDecl("in", rows, cols),
      `skepu::Vector<int> out(${cols});`,
      "",
      "skel(out, in);",
    ].join("\n");
  },
  desc: "Colwise reduction: each input column is reduced to a single output element (1×N row vector). Hover an output cell to see its contributing column.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output element" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed column" },
  ],
  extraControls: [],
  defaults: { rows: 4, cols: 6 },
  render(areaEl, { rows, cols }) {
    let inputCells = [], outputCells = [];

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c => c.classList.remove("highlight-primary")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(outR, outC) {
      clearHighlights();
      outputCells[0][outC].classList.add("output-hover");
      for (let r = 0; r < rows; r++) inputCells[r][outC].classList.add("highlight-primary");
    }

    const { group: inGroup, mat: inEl }   = makeMatrixGroup("IN");
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");
    inputCells  = buildMatrix(inEl,  rows, cols, null,                null, null);
    outputCells = buildMatrix(outEl, 1,    cols, (r, c) => `c${c}`, onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, inGroup]);
  },
};

/* ---- Reduce — 2-operator ------------------------------------------------ */
SkePUSkeletons.reduce_2op = {
  name: "Reduce — 2-op",
  codeGen({ rows, cols }) {
    return [
      funcDecl("f1", ["a", "b"]),
      funcDecl("f2", ["a", "b"]),
      "",
      "auto skel = skepu::Reduce(f1, f2);",
      "",
      containerDecl("in", rows, cols),
      "",
      "int out = skel(in);",
    ].join("\n");
  },
  desc: "Two-operator reduction: the first operator reduces each row to a partial result (N×1); the second reduces those partials to a scalar. Hover a partial cell to see its contributing row; hover the scalar to see the full read path.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output cell" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed by first operator (row → partial)" },
    { bg: "#C0DD97", border: "#639922", label: "Accessed by second operator (partial → scalar)" },
  ],
  extraControls: [],
  defaults: { rows: 4, cols: 6 },
  render(areaEl, { rows, cols }) {
    let inputCells = [], partialCells = [], outputCells = [];

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c =>
        c.classList.remove("highlight-primary", "highlight-secondary")));
      partialCells.forEach(row => row.forEach(c =>
        c.classList.remove("highlight-primary", "highlight-secondary", "output-hover")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onPartialHover(r) {
      clearHighlights();
      partialCells[r][0].classList.add("output-hover");
      for (let c = 0; c < cols; c++) inputCells[r][c].classList.add("highlight-primary");
    }
    function onOutputHover() {
      clearHighlights();
      outputCells[0][0].classList.add("output-hover");
      for (let r = 0; r < rows; r++) {
        partialCells[r][0].classList.add("highlight-secondary");
        for (let c = 0; c < cols; c++) inputCells[r][c].classList.add("highlight-primary");
      }
    }

    const { group: inGroup,      mat: inEl }      = makeMatrixGroup("IN");
    const { group: partialGroup, mat: partialEl } = makeMatrixGroup("PARTIAL (op 1)");
    const { group: outGroup,     mat: outEl }     = makeMatrixGroup("OUT (op 2)");

    inputCells   = buildMatrix(inEl,      rows, cols, null,        null,           null);
    partialCells = buildMatrix(partialEl, rows, 1,    r => `r${r}`, onPartialHover, clearHighlights);
    outputCells  = buildMatrix(outEl,     1,    1,    () => "Σ",    onOutputHover,  clearHighlights);

    stack(areaEl, [outGroup, partialGroup, inGroup]);
  },
};

/* ---- MapPairs ----------------------------------------------------------- */
SkePUSkeletons.mappairs = {
  name: "MapPairs",
  operands: true, outputs: true,
  codeGen(p) {
    const { rows, cols, nvert = 1, nhoriz = 1 } = p;
    const o = operandCode(p);
    const vertNames  = Array.from({ length: nvert  }, (_, i) => `vert${nvert  === 1 ? "" : i + 1}`);
    const horizNames = Array.from({ length: nhoriz }, (_, i) => `horiz${nhoriz === 1 ? "" : i + 1}`);
    const vArgs = Array.from({ length: nvert  }, (_, i) => nvert  === 1 ? "v" : `v${i + 1}`);
    const hArgs = Array.from({ length: nhoriz }, (_, i) => nhoriz === 1 ? "h" : `h${i + 1}`);
    return [
      funcDecl("f", [...vArgs, ...hArgs], o),
      "",
      "auto skel = skepu::MapPairs(f);",
      "",
      ...o.outNames.map(name => containerDecl(name, rows, cols)),
      ...vertNames.map(n  => `skepu::Vector<int> ${n}(${rows});`),
      ...horizNames.map(n => `skepu::Vector<int> ${n}(${cols});`),
      ...o.decls,
      "",
      `skel(${[...o.outNames, ...vertNames, ...horizNames, ...o.args].join(", ")});`,
    ].join("\n");
  },
  desc: "Output cell (i, j) reads element i from every vertical input vector and element j from every horizontal input vector — a generalised outer-product computation. Hover an output cell to see its accessed elements.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output cell" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed element in vertical input(s)" },
    { bg: "#C0DD97", border: "#639922", label: "Accessed element in horizontal input(s)" },
  ],
  extraControls: [
    { type: "number", id: "nvert",  label: "Vert inputs",  min: 1, max: 3, default: 1 },
    { type: "number", id: "nhoriz", label: "Horiz inputs", min: 1, max: 3, default: 1 },
  ],
  defaults: { rows: 4, cols: 5, nvert: 1, nhoriz: 1 },
  render(areaEl, { rows, cols, nvert = 1, nhoriz = 1 }) {
    let vertCells = [], horizCells = [], outputCells = [];

    function clearHighlights() {
      vertCells.forEach(vec =>
        vec.forEach(row => row.forEach(c => c.classList.remove("highlight-vert"))));
      horizCells.forEach(vec =>
        vec.forEach(row => row.forEach(c => c.classList.remove("highlight-horiz"))));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(r, c) {
      clearHighlights();
      outputCells[r][c].classList.add("output-hover");
      vertCells.forEach(vec  => vec[r][0].classList.add("highlight-vert"));
      horizCells.forEach(vec => vec[0][c].classList.add("highlight-horiz"));
    }

    const corner = document.createElement("div");

    const horizArea = document.createElement("div");
    horizArea.className = "mp-horiz-area";
    horizCells = [];
    for (let h = 0; h < nhoriz; h++) {
      const { group, mat } = makeMatrixGroup(
        nhoriz === 1 ? "HORIZ IN" : `HORIZ IN ${h + 1}`, "viz-vec-group");
      horizCells.push(buildMatrix(mat, 1, cols, (r, c) => `h${c}`, null, null));
      horizArea.appendChild(group);
    }

    const vertArea = document.createElement("div");
    vertArea.className = "mp-vert-area";
    vertCells = [];
    for (let v = 0; v < nvert; v++) {
      const { group, mat } = makeMatrixGroup(
        nvert === 1 ? "VERT IN" : `VERT ${v + 1}`, "viz-vec-group");
      vertCells.push(buildMatrix(mat, rows, 1, (r) => `v${r}`, null, null));
      vertArea.appendChild(group);
    }

    const { group: outGroup, mat: outMat } = makeMatrixGroup(
      "OUT", "viz-vec-group");
    outputCells = buildMatrix(outMat, rows, cols, null, onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, vertArea, horizArea, corner], "mp-grid");
  },
};

/* ---- MapPairsReduce ----------------------------------------------------- */
SkePUSkeletons.mappairsreduce = {
  name: "MapPairsReduce",
  operands: true,
  codeGen(p) {
    const { rows, cols, nvert = 1, nhoriz = 1, mode = "row" } = p;
    const o = operandCode({ ...p, outputs: 1 });
    const vertNames  = Array.from({ length: nvert  }, (_, i) => `vert${nvert  === 1 ? "" : i + 1}`);
    const horizNames = Array.from({ length: nhoriz }, (_, i) => `horiz${nhoriz === 1 ? "" : i + 1}`);
    const vArgs = Array.from({ length: nvert  }, (_, i) => nvert  === 1 ? "v" : `v${i + 1}`);
    const hArgs = Array.from({ length: nhoriz }, (_, i) => nhoriz === 1 ? "h" : `h${i + 1}`);
    return [
      funcDecl("f1", [...vArgs, ...hArgs], o),
      funcDecl("f2", ["a", "b"]),
      "",
      "auto skel = skepu::MapPairsReduce(f1, f2);",
      "",
      `skepu::Vector<int> out(${mode === "row" ? rows : cols});`,
      ...vertNames.map(n  => `skepu::Vector<int> ${n}(${rows});`),
      ...horizNames.map(n => `skepu::Vector<int> ${n}(${cols});`),
      ...o.decls,
      "",
      `skel(${["out", ...vertNames, ...horizNames, ...o.args].join(", ")});`,
    ].join("\n");
  },
  desc({ mode = "row" } = {}) {
    return mode === "row"
      ? "MapPairsReduce (row-wise): each output element at row i reduces all intermediate elements in row i of the MapPairs matrix. Hover an intermediate cell to see its MapPairs inputs; hover an output cell to see the full chain."
      : "MapPairsReduce (col-wise): each output element at column j reduces all intermediate elements in column j of the MapPairs matrix. Hover an intermediate cell to see its MapPairs inputs; hover an output cell to see the full chain.";
  },
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered cell" },
    { bg: "#FAEBC4", border: "#C49600", label: "Intermediate cells reduced into output" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed vertical input element(s)" },
    { bg: "#C0DD97", border: "#639922", label: "Accessed horizontal input element(s)" },
  ],
  extraControls: [
    { type: "number", id: "nvert",  label: "Vert inputs",  min: 1, max: 3, default: 1 },
    { type: "number", id: "nhoriz", label: "Horiz inputs", min: 1, max: 3, default: 1 },
    { type: "toggle", id: "mode", label: "Mode",
      options: [{ value: "row", label: "Row-wise" }, { value: "col", label: "Col-wise" }],
      default: "row" },
  ],
  defaults: { rows: 4, cols: 5, nvert: 1, nhoriz: 1, mode: "row" },
  render(areaEl, { rows, cols, nvert = 1, nhoriz = 1, mode = "row" }) {
    let vertCells = [], horizCells = [], interCells = [], outputCells = [];

    function clearHighlights() {
      vertCells.forEach(vec =>
        vec.forEach(row => row.forEach(c => c.classList.remove("hl-vert"))));
      horizCells.forEach(vec =>
        vec.forEach(row => row.forEach(c => c.classList.remove("hl-horiz"))));
      interCells.forEach(row =>
        row.forEach(c => c.classList.remove("hl-inter", "output-hover")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onInterHover(r, c) {
      clearHighlights();
      interCells[r][c].classList.add("output-hover");
      vertCells.forEach(vec  => vec[r][0].classList.add("hl-vert"));
      horizCells.forEach(vec => vec[0][c].classList.add("hl-horiz"));
    }
    function onOutputHover(r, c) {
      clearHighlights();
      outputCells[r][c].classList.add("output-hover");
      if (mode === "row") {
        for (let dc = 0; dc < cols; dc++) interCells[r][dc].classList.add("hl-inter");
        vertCells.forEach(vec  => vec[r][0].classList.add("hl-vert"));
        horizCells.forEach(vec => vec[0].forEach(cell => cell.classList.add("hl-horiz")));
      } else {
        for (let dr = 0; dr < rows; dr++) interCells[dr][c].classList.add("hl-inter");
        vertCells.forEach(vec  => vec.forEach(row => row[0].classList.add("hl-vert")));
        horizCells.forEach(vec => vec[0][c].classList.add("hl-horiz"));
      }
    }

    const corner = document.createElement("div");

    const horizArea = document.createElement("div");
    horizArea.className = "mpr-horiz-area";
    horizCells = [];
    for (let h = 0; h < nhoriz; h++) {
      const { group, mat } = makeMatrixGroup(
        nhoriz === 1 ? "HORIZ IN" : `HORIZ IN ${h + 1}`, "viz-vec-group");
      horizCells.push(buildMatrix(mat, 1, cols, (r, c) => `h${c}`, null, null));
      horizArea.appendChild(group);
    }

    const vertArea = document.createElement("div");
    vertArea.className = "mpr-vert-area";
    vertCells = [];
    for (let v = 0; v < nvert; v++) {
      const { group, mat } = makeMatrixGroup(
        nvert === 1 ? "VERT IN" : `VERT IN ${v + 1}`, "viz-vec-group");
      vertCells.push(buildMatrix(mat, rows, 1, (r) => `v${r}`, null, null));
      vertArea.appendChild(group);
    }

    const { group: interGroup, mat: interMat } = makeMatrixGroup(
      "INTERMEDIATE", "viz-vec-group");
    interCells = buildMatrix(interMat, rows, cols, null, onInterHover, clearHighlights);

    const isRow = mode === "row";
    const { group: outGroup, mat: outMat } = makeMatrixGroup(
      "OUT", "viz-vec-group");
    outputCells = buildMatrix(
      outMat,
      isRow ? rows : 1,
      isRow ? 1 : cols,
      isRow ? (r) => `r${r}` : (r, c) => `c${c}`,
      onOutputHover, clearHighlights);

    const blank = document.createElement("div");
    // the output first; the intermediate result, the vertical and the horizontal inputs aligned with it
    if (isRow) stack(areaEl, [outGroup, interGroup, vertArea, blank, horizArea, corner], "mpr-grid mpr-by-row");
    else       stack(areaEl, [blank, outGroup, vertArea, interGroup, corner, horizArea], "mpr-grid mpr-by-col");
  },
};

/* ---- MapPool ------------------------------------------------------------ */
SkePUSkeletons.mappool = {
  name: "MapPool",
  operands: true, outputs: true,
  codeGen(p) {
    const { rows, cols, poolH = 2, poolW = 2 } = p;
    const o = operandCode(p);
    const outRows = Math.floor(rows / poolH);
    const outCols = Math.floor(cols / poolW);
    return [
      funcDecl("f", ["a", "b"], o),
      "",
      "auto skel = skepu::MapPool(f);",
      "",
      ...o.outNames.map(name => containerDecl(name, outRows, outCols)),
      containerDecl("in", rows, cols),
      ...o.decls,
      "",
      `skel(${[...o.outNames, "in", ...o.args].join(", ")});`,
    ].join("\n");
  },
  desc: "Each output cell (i, j) reduces a ph × pw block of input cells starting at (i·ph, j·pw). Output dimensions are ⌊rows/ph⌋ × ⌊cols/pw⌋. Faded input cells fall outside all pools. Hover an output cell to see its pool.",
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output cell" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accessed pool region" },
  ],
  extraControls: [
    { type: "number", id: "poolH", label: "Pool rows", min: 1, max: 4, default: 2 },
    { type: "number", id: "poolW", label: "Pool cols", min: 1, max: 4, default: 2 },
  ],
  defaults: { rows: 6, cols: 8, poolH: 2, poolW: 2 },
  render(areaEl, { rows, cols, poolH = 2, poolW = 2 }) {
    const outRows = Math.floor(rows / poolH);
    const outCols = Math.floor(cols / poolW);
    let inputCells = [], outputCells = [];

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c => c.classList.remove("highlight-pool")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(oR, oC) {
      clearHighlights();
      outputCells[oR][oC].classList.add("output-hover");
      for (let dr = 0; dr < poolH; dr++)
        for (let dc = 0; dc < poolW; dc++)
          inputCells[oR * poolH + dr][oC * poolW + dc].classList.add("highlight-pool");
    }

    const { group: inGroup, mat: inEl }   = makeMatrixGroup("IN");
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");

    stack(areaEl, [outGroup, inGroup]);

    inputCells  = buildMatrix(inEl,  rows,    cols,    null, null, null);
    outputCells = buildMatrix(outEl, outRows, outCols, null, onOutputHover, clearHighlights);

    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        if (r >= outRows * poolH || c >= outCols * poolW)
          inputCells[r][c].classList.add("out-of-pool");
  },
};

/* ---- Scan --------------------------------------------------------------- */
SkePUSkeletons.scan = {
  name: "Scan",
  codeGen({ rows, cols }) {
    return [
      funcDecl("f", ["a", "b"]),
      "",
      "auto skel = skepu::Scan(f);",
      "",
      containerDecl("in",  rows, cols),
      containerDecl("out", rows, cols),
      "",
      "skel(out, in);",
    ].join("\n");
  },
  desc({ mode = "inclusive" } = {}) {
    return mode === "inclusive"
      ? "Inclusive prefix sum: output element k holds the sum of input elements 0 … k (row-major order). Hover an output cell to see which inputs accumulate into it."
      : "Exclusive prefix sum: output element k holds the sum of input elements 0 … k−1 (row-major order). The first element receives the start value (identity). Hover an output cell to see which inputs accumulate into it.";
  },
  legend: [
    { bg: "#378ADD", border: "#185FA5", label: "Hovered output element" },
    { bg: "#B5D4F4", border: "#378ADD", label: "Accumulated input elements" },
  ],
  extraControls: [
    { type: "toggle", id: "mode", label: "Mode",
      options: [{ value: "inclusive", label: "Inclusive" }, { value: "exclusive", label: "Exclusive" }],
      default: "inclusive" },
  ],
  defaults: { rows: 3, cols: 5, mode: "inclusive" },
  render(areaEl, { rows, cols, mode = "inclusive" }) {
    let inputCells = [], outputCells = [];
    const labelFn = (r, c) => String(r * cols + c);

    function clearHighlights() {
      inputCells.forEach(row => row.forEach(c => c.classList.remove("highlight-primary")));
      outputCells.forEach(row => row.forEach(c => c.classList.remove("output-hover")));
    }
    function onOutputHover(oR, oC) {
      clearHighlights();
      const limit = oR * cols + oC;
      outputCells[oR][oC].classList.add("output-hover");
      inputCells.forEach((row, r) =>
        row.forEach((cell, c) => {
          const idx = r * cols + c;
          if (mode === "inclusive" ? idx <= limit : idx < limit)
            cell.classList.add("highlight-primary");
        })
      );
    }

    const { group: inGroup, mat: inEl }   = makeMatrixGroup("IN");
    const { group: outGroup, mat: outEl } = makeMatrixGroup("OUT");
    inputCells  = buildMatrix(inEl,  rows, cols, labelFn, null, null);
    outputCells = buildMatrix(outEl, rows, cols, labelFn, onOutputHover, clearHighlights);

    stack(areaEl, [outGroup, inGroup]);
  },
};

window.SkePUEntries = [
  { id: "map", key: "map" },
  { id: "mapoverlap", key: "mapoverlap" },
  { id: "reduce", name: "Reduce", variants: [
    { key: "reduce_full", name: "Full" },
    { key: "reduce_row",  name: "Row-wise" },
    { key: "reduce_col",  name: "Col-wise" },
    { key: "reduce_2op",  name: "2-operator" },
  ] },
  { id: "mappairs", key: "mappairs" },
  { id: "mappairsreduce", key: "mappairsreduce" },
  { id: "mappool", key: "mappool" },
  { id: "scan", key: "scan" },
];

/* Controls and proxies that every skeleton with the flags `outputs` and `operands` has, beyond its own.
 * A proxy is a fixed container, 3 elements or 3×3, that the user function may read at any index (Vec, Mat) or the row or column of the output's index (MatRow, MatCol). */
window.SkePUOperands = {
  outputsControl: { type: "number", id: "outputs", label: "Outputs", min: 1, max: 3, default: 1 },
  controls: [
    { type: "toggle", id: "proxy", label: "Proxy", default: "none",
      options: [{ value: "none", label: "None" }, { value: "vec", label: "Vec" }, { value: "mat", label: "Mat" },
                { value: "matrow", label: "MatRow" }, { value: "matcol", label: "MatCol" }] },
    { type: "number", id: "uniforms", label: "Uniforms", min: 0, max: 3, default: 0 },
  ],
  proxyLegend: { bg: "#DCD3F2", border: "#8C79C7", label: "Proxy element(s) that the function may read" },
  uniformLegend: { bg: "#F6D5C6", border: "#D0805A", label: "Uniform" },
  proxies: {
    vec:    { rows: 1, cols: 3, access: "all", label: "Vec" },
    mat:    { rows: 3, cols: 3, access: "all", label: "Mat" },
    matrow: { rows: 3, cols: 3, access: "row", label: "MatRow" },
    matcol: { rows: 3, cols: 3, access: "col", label: "MatCol" },
  },
};
