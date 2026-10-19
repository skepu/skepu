// skevu.js — the entry of the viewer in the webview of an editor (bundled by esbuild.js into dist/skevu.js, which the page that src/webviewHtml.ts
// makes from index.html runs). The page, its controls and its menus are those of the browser page (script/ui.js, script/actions.js); this is what
// is the editor's: the trace comes as a message from the extension, and what the viewer selects goes back as messages (script/graph_vscode.js).
import './script/util_vscode.js';
import './script/workers/worker-loader-vscode.js';
import './script/graph_vscode.js';
import { load } from './script/ui.js';
import { bindActions } from './script/actions.js';
import { loadFiles } from './script/workers/worker-loader.js';
import { State } from './script/state.js';
import { followEditorTheme } from './script/theme_vscode.js';

// The API to post messages to the extension.
// @ts-ignore
const vscode = acquireVsCodeApi();
window.skevu = Object.assign(window.skevu || {}, { vscode });

bindActions();
load();
followEditorTheme();

// A click in the viewer brings the viewer forward among the editors.
document.addEventListener('click', () => {
	vscode.postMessage({ type: 'focus_skevu_editor' });
});

// The graph that the viewer has drawn is what the extension needs to put the badges at the lines of the sources (and of the trace).
State.provider?.on('data', (rawData) => {
	// (the regions have badges only if they are shown, as in the viewer: badgeNodesToShow)
	vscode.postMessage({ type: 'build_documents', rawData, showRegions: !!State.settings['show_regions'] });
});

/** Show the trace that is the text of the document. */
function updateContent(/** @type {string} */ text) {
	loadFiles(text);
}

window.addEventListener('message', event => {
	const message = event.data;
	switch (message.type) {
		case 'update':
			updateContent(message.text);
			// (the state is what a webview that is made again when it is shown again gets back)
			vscode.setState({ text: message.text });
			return;

		case 'source_files':
			// (the names of the source files that the extension has found: the links of the info pane are for those)
			window.skevu.sourceFiles = new Set(message.files.map((f) => String(f).split('/').pop().split('\\').pop()));
			return;

		case 'highlight_node':
			State.provider?._emit('select', {
				nodeId:  message.node_id,
				source:  'code-listing',
				focusId: message.node_id,
				allIds:  [message.node_id],
				iterIdx: 0,
			});
			return;
	}
});

// A webview is torn down when it is not visible, and made again when it is: the trace it had is in the state.
const state = vscode.getState();
if (state)
	updateContent(state.text);
