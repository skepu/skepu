import { setInfoViewOnHighlightNode, setInfoViewOnShowSource, setInfoViewCanShowSource } from './graph.js'

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

// a link is for a file that the extension has found (with the hash of the trace): its names come as 'source_files' messages (skevu.js)
setInfoViewCanShowSource(function(file) {
  var names = window.skevu.sourceFiles;
  return !!names && !!file && names.has(String(file).split('/').pop().split('\\').pop());
});

