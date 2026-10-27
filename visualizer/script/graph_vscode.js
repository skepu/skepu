import { setInfoViewOnHighlightNode, setInfoViewOnShowSource, setInfoViewCanShowSource, setInfoViewOnDefinitions } from './graph.js'
import { definitionMarks } from './badges.js'

function highlightVSCodeText(data) {
  if (!data) return;

  var idx = (data.trace_indices && data.trace_indices.length)
    ? data.trace_indices[0]
    : data.trace_index;
  if (idx != null) {
    window.skevu.vscode.postMessage({
        type: 'highlight_trace_file',
        node_id: data.node_id
    });
  }

  var line = data.line != null ? data.line : data.Line;
  var file = data.file         ? data.file : data.File;
  if (line != null && line !== -1 && file) {
    window.skevu.vscode.postMessage({
        type: 'highlight_source_file',
        source_file: file,
        source_file_line_index: line,
        node_id: data.node_id
    });
  }
}

setInfoViewOnHighlightNode(highlightVSCodeText);

// the links to the definitions of a call (and Shift- and Alt-click on a node): the lines are shown in the editor, not in a source pane
setInfoViewOnShowSource(function(place) {
  window.skevu.vscode.postMessage({
    type: 'reveal_source',
    source_file: place.file,
    line: place.line,
    end_line: place.end_line
  });
});

// Shades the places of a call lightly in the editor (instance, user functions, the call itself), as the source pane does in the browser.
// No marks removes the shading.
setInfoViewOnDefinitions(function(definitions, call) {
  var marks = definitionMarks(definitions);
  if (marks.length && call && call.file && Number.isInteger(call.line) && call.line > 0)
    marks.push({ role: 'call', file: call.file, from: call.line, to: call.line });
  window.skevu.vscode.postMessage({ type: 'show_definitions', marks: marks });
});

// a link is for a file that the extension has found (with the hash of the trace): its names come as 'source_files' messages (skevu.js)
setInfoViewCanShowSource(function(file) {
  var names = window.skevu.sourceFiles;
  return !!names && !!file && names.has(String(file).split('/').pop().split('\\').pop());
});

