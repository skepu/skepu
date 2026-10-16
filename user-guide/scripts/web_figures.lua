-- web_figures.lua — for the simple edition of the book (_quarto-simple.yml): a figure that is a PDF is shown from an SVG of the same name next to it, if there is
-- one. A browser engine without a PDF viewer (the webview of an editor) shows an <embed> of a PDF as nothing at all. Without the SVG the PDF stays.

local function exists(path)
  local fh = io.open(path, "r")
  if fh then fh:close() end
  return fh ~= nil
end

local function directory_of(file)
  return (file or ""):match("^(.*)[/\\][^/\\]*$") or "."
end

function Image(img)
  local svg, n = img.src:gsub("%.pdf$", ".svg")
  if n == 0 then return nil end
  if exists(directory_of(quarto.doc.input_file) .. "/" .. svg) then
    img.src = svg
    return img
  end
  return nil
end
