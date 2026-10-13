import { setInfoViewOnHighlightNode } from './graph.js'

function highlightVSCodeText(data) {
  if (!data) return;

  window.skevu.vscode.postMessage({
      type: 'status_message',
      text: data
  });

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