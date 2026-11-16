--[[
  listings.lua — a runner after each listing marked to run (```{.cpp .run}) that scripts/build_listings.py built; see README.md.
]]

local skevu = dofile(quarto.utils.resolve_path("skevu.lua"))

local function exists(path)
  local fh = io.open(path, "r")
  if fh then fh:close() end
  return fh ~= nil
end

local occurrence = {}

-- A runner is not for the simple edition and not for print.
local function runners_wanted()
  if not quarto.doc.is_format("html") then return false end
  for _, profile in ipairs(quarto.project.profile or {}) do
    if profile == "simple" then return false end
  end
  return true
end

local function runner(el)
  if not runners_wanted() then return nil end
  local hash = pandoc.sha1(el.text):sub(1, 10)
  local root = quarto.project.directory or "."
  if not exists(root .. "/listings/wasm/" .. hash .. ".js") then return nil end

  -- the pages mirror the source directory depth: listings/ is next to the top-level pages
  local file = quarto.doc.input_file or ""
  if file:sub(1, #root) == root then file = file:sub(#root + 2) end
  local depth = select(2, file:gsub("[/\\]", ""))

  occurrence[hash] = (occurrence[hash] or 0) + 1
  local function value(name, default)
    local v = el.attributes[name]
    return pandoc.Inlines({ pandoc.Str(v or default) })
  end
  local kwargs = {
    id = pandoc.Inlines({ pandoc.Str("te-" .. hash .. "-" .. occurrence[hash]) }),
    js = pandoc.Inlines({ pandoc.Str(string.rep("../", depth) .. "listings/wasm/" .. hash .. ".js") }),
    argv = value("argv", ""),
    -- the field of the arguments is for a listing that says what they are
    ["hide-argv"] = pandoc.Inlines({ pandoc.Str(el.attributes["argv"] == nil and "true" or "false") }),
    views = value("views", "none"),
    label = el.attributes["label"] and value("label", "") or nil,
  }
  local blocks = skevu["trace-wasm"]({}, kwargs)
  -- without the class, the walk that follows does not make a runner of the listing again
  local shown = el:clone()
  shown.classes = shown.classes:filter(function(c) return c ~= "run" end)
  table.insert(blocks, 1, shown)
  -- one box: the runner is attached to its listing
  return pandoc.Div(blocks, pandoc.Attr("", { "te-listing" }))
end

-- The expected output of a listing, for a page without its runner.
function CodeBlock(el)
  if not el.classes:includes("output") then return nil end
  return pandoc.Div({ pandoc.Para({ pandoc.Emph({ pandoc.Str("Output") }) }), pandoc.CodeBlock(el.text, pandoc.Attr("", { "text" })) },
    pandoc.Attr("", { "te-output" }))
end

local function is_output(block)
  return block ~= nil and block.t == "Div" and block.classes:includes("te-output")
end

-- Where a runner is attached to the listing, the output block after it is left out. After a captioned listing (a float, which this walk does
-- not see as a pair with its output) the stylesheet hides it.
local function process(blocks)
  local result = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local block = blocks[i]
    if block.t == "CodeBlock" and block.classes:includes("run") then
      local box = runner(block)
      result:insert(box or block)
      if box and is_output(blocks[i + 1]) then i = i + 1 end
    else
      result:insert(block)
    end
    i = i + 1
  end
  return result
end

function Blocks(blocks)
  return process(blocks)
end

-- A listing with an id (lst-...) is a float that Quarto makes its own node, which the generic walk does not enter.
function FloatRefTarget(float)
  local content = float.content
  if content.t == "CodeBlock" and content.classes:includes("run") then
    float.content = runner(content) or content
  elseif content.t == "Div" then
    content.content = process(content.content)
  end
  return float
end
