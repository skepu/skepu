--[[
  skevu.lua — Quarto shortcodes embedding SkeVU trace panels in the book.

    {{< trace-embed example="3" backend="openmp" views="graph,source,timeline"
                    label="Dot Product" controls="direction,node-color" >}}
        A fixed trace: entry `example` (index) of the visualizer's examples.json.

    {{< trace-wasm js="../demos/wasm/map_squaring.js" src="demos/map_squaring.cpp" argv="10"
                   views="graph,source" label="Map: element-wise squaring" >}}
        A WebAssembly program with a Run button; its trace.json is visualized after each run.
        With views="none" only the program is shown, add code="true" to list its source above it.

  Parameters
  ----------
  views     Comma-separated subset of graph, source, timeline, or none.  Default: graph.
  height    Height in pixels of the graph and source panes.  Default: 420.
  label     Heading bar.
  controls  Interactive controls to offer: regions, collapse-iter, element-accesses,
            anti-deps, direction, node-color.  Default: none (a read-only panel).
  opt-*     Initial settings: opt-regions, opt-collapse-iter, opt-element-accesses,
            opt-anti-deps (true/false), opt-direction (vertical|horizontal),
            opt-node-color (duration-total|dag-depth|total-order|fixed).
  example, backend   (trace-embed) Example index and backend name.  Default: 0, openmp.
  js        (trace-wasm) Emscripten glue .js, relative to the rendered page.
  src       (trace-wasm) C++ source, relative to the project root: read at build time to be
            shown in the source view, and listed above the program with code="true".
  argv      (trace-wasm) Default arguments.

  The runtime is skevu.js; the panels load the visualizer from the viz/ directory of the
  book (installed by scripts/install_viz.py).
--]]

local OPT_SETTINGS = {
  ["opt-regions"]          = "show_regions",
  ["opt-collapse-iter"]    = "collapse_iteration",
  ["opt-element-accesses"] = "element-accesses",
  ["opt-anti-deps"]        = "anti-deps",
  ["opt-direction"]        = "direction",
  ["opt-node-color"]       = "node-color-call",
}

local counter = 0

local function param(kwargs, name, default)
  local v = kwargs[name] and pandoc.utils.stringify(kwargs[name]) or ""
  if v == "" then return default end
  return v
end

local function split_list(s)
  local list = {}
  for item in s:gmatch("[^,]+") do
    item = item:match("^%s*(.-)%s*$")
    if item ~= "" then table.insert(list, item) end
  end
  return list
end

local function html_escape(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub('"', "&quot;"))
end

local function read_file(path)
  local fh = io.open(path, "r")
  if not fh then return nil end
  local text = fh:read("*a")
  fh:close()
  return text
end

-- Parameters common to both shortcodes.
local function common(kwargs)
  local c = {
    label    = param(kwargs, "label", nil),
    views    = split_list(param(kwargs, "views", "graph")),   -- "none": no views
    controls = split_list(param(kwargs, "controls", "")),
    height   = tonumber(param(kwargs, "height", "420")) or 420,
    settings = {},
  }
  for name, key in pairs(OPT_SETTINGS) do
    local v = param(kwargs, name, nil)
    if v == "true" then c.settings[key] = true
    elseif v == "false" then c.settings[key] = false
    elseif v then c.settings[key] = v end
  end
  if c.views[1] == "none" then c.views = {} end
  c.has = {}
  for _, v in ipairs(c.views) do c.has[v] = true end
  -- The pages mirror the source directory depth: viz/ is next to the top-level pages.
  local file = quarto.doc.input_file or ""
  local root = quarto.project.directory
  if root and file:sub(1, #root) == root then file = file:sub(#root + 2) end
  local depth = select(2, file:gsub("[/\\]", ""))
  c.vizBase = string.rep("../", depth) .. "viz/"
  return c
end

-- The panel's runtime and style sheet, added once per page.
local function add_dependency()
  quarto.doc.add_html_dependency({
    name = "skevu", version = "2.0.0",
    scripts = { "skevu.js" }, stylesheets = { "skevu.css" },
  })
end

-- The views of a panel: graph (with statistics, details and controls), source and timeline.
local function views_html(id, c)
  local h = {}
  local function add(s) table.insert(h, s) end
  add('<div class="' .. (c.has.source and "te-main te-main-split" or "te-main") .. '">')
  if c.has.graph then
    add('<div class="te-graph-col">')
    add('<div class="te-graph-wrap">')
    add('<div id="' .. id .. '-cy" class="te-cy" style="height:' .. c.height .. 'px"></div>')
    add('<div id="' .. id .. '-loading" class="te-loading"><div class="te-spinner"></div><span>Loading…</span></div>')
    add('<div id="' .. id .. '-error" class="te-error"></div>')
    add('</div>')
    add('<div id="' .. id .. '-stats" class="te-stats-bar"></div>')
    add('<div id="' .. id .. '-info" class="te-info-box te-info-empty">Click a node to see details</div>')
    if #c.controls > 0 then add('<div id="' .. id .. '-controls" class="te-controls-row"></div>') end
    add('</div>')
  end
  if c.has.source then
    add('<div class="te-src-col">')
    add('<div id="' .. id .. '-tab-bar" class="te-tab-bar"></div>')
    add('<div id="' .. id .. '-tab-panes" class="te-tab-panes" style="height:' .. c.height .. 'px"></div>')
    add('</div>')
  end
  add('</div>')
  if c.has.timeline then
    add('<div class="te-timeline-wrap"><div id="' .. id .. '-timeline" class="te-timeline-scroll"></div></div>')
  end
  return table.concat(h, "\n")
end

local function label_html(c)
  return c.label and ('<div class="te-label">' .. html_escape(c.label) .. '</div>') or ""
end

-- The script that starts the panel: skevu.js is already loaded, the panel's DOM precedes it.
local function mount_script(id, c, extra)
  local config = {
    views = c.views, controls = c.controls, settings = c.settings, vizBase = c.vizBase,
  }
  for k, v in pairs(extra) do config[k] = v end
  return pandoc.RawBlock("html",
    '<script>skevu.mount("' .. id .. '", ' .. quarto.json.encode(config) .. ');</script>')
end

return {
  ["trace-embed"] = function(args, kwargs)
    if not quarto.doc.is_format("html") then return {} end
    add_dependency()
    local c = common(kwargs)
    counter = counter + 1
    local id = "te-" .. counter
    local panel = pandoc.Div(
      { pandoc.RawBlock("html", label_html(c) .. views_html(id, c)) },
      pandoc.Attr(id, { "te-panel" }))
    return { panel, mount_script(id, c, {
      example = tonumber(param(kwargs, "example", "0")) or 0,
      backend = param(kwargs, "backend", "openmp"),
    }) }
  end,

  ["trace-wasm"] = function(args, kwargs)
    if not quarto.doc.is_format("html") then return {} end
    add_dependency()
    local c = common(kwargs)
    counter = counter + 1
    local id = "te-" .. counter

    local src_path = param(kwargs, "src", nil)
    local source = src_path and (read_file(src_path) or ("// source file not found: " .. src_path))
    local blocks = { pandoc.RawBlock("html", label_html(c)) }
    -- A real code block, so that Quarto highlights it.
    if source and param(kwargs, "code", "false") == "true" then
      table.insert(blocks, pandoc.CodeBlock(source, pandoc.Attr("", { "cpp" })))
    end
    table.insert(blocks, pandoc.RawBlock("html", table.concat({
      '<div class="te-wasm-section">',
      '  <div class="te-wasm-run-row">',
      '    <input id="' .. id .. '-argv" class="te-wasm-argv" type="text" value="'
        .. html_escape(param(kwargs, "argv", "")) .. '" placeholder="arguments…">',
      '    <button id="' .. id .. '-run" class="te-wasm-run">▶ Run</button>',
      '    <button id="' .. id .. '-clear" class="te-wasm-clear">Clear</button>',
      '  </div>',
      '  <pre id="' .. id .. '-output" class="te-wasm-output" aria-live="polite"></pre>',
      '</div>',
    }, "\n")))
    if #c.views > 0 then
      table.insert(blocks, pandoc.RawBlock("html", table.concat({
        '<div class="te-trace-section">',
        '  <div id="' .. id .. '-placeholder" class="te-trace-placeholder">▶ Run the program above to visualize its execution trace.</div>',
        '  <div id="' .. id .. '-trace" style="display:none">',
        views_html(id, c),
        '  </div>',
        '</div>',
      }, "\n")))
    end

    local panel = pandoc.Div(blocks, pandoc.Attr(id, { "te-panel", "te-wasm-panel" }))
    return { panel, mount_script(id, c, { wasm = {
      jsUrl = param(kwargs, "js", ""),
      srcContent = source,
      srcFilename = src_path and (src_path:match("[^/\\]+$") or src_path) or "source.cpp",
    } }) }
  end,
}
