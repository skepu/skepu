// @ts-check



import './script/util_vscode.js';
import './script/workers/worker-loader-vscode.js';
import './script/graph_vscode.js';
import { renderImage, setHljsTheme, simStop, simToggle, stopRendering, switchTab } from './script/graph.js';
import { menuRestoreDefaults, menuSelectRadio, menuToggleCheck } from './script/settings.js';
import { fetchViewDataAndRender, recenterGraph, toggleCodePaths, toggleGanttPane, toggleLegend, toggleSourcePane, load, zoomGanttToMeasuredInterval, collapseSelectedRecursively, expandSelectedRecursively, collapseSelected, expandSelected, toggleDarkMode, collapseAllRegions, expandAllRegions, collapseAllFusions, expandAllFusions, colorSpectrumChanged } from './script/ui.js';
import { loadFiles } from './script/workers/worker-loader.js';
import { State } from './script/state.js';

/**
 * @param {{ stopPropagation: () => void; }} event
 * @param {{ closest: (arg0: string) => any; }} btn
 */
function menuToggle(event, btn) {
	event.stopPropagation();
	var menu   = btn.closest('.menu');
	var isOpen = menu.classList.contains('open');
	menuClose();
	if (!isOpen) menu.classList.add('open');
}

function menuClose() {
	document.querySelectorAll('.menu.open').forEach(function(m) { m.classList.remove('open'); });
}

function syncExternalMenuIcons() {
	var lb = document.getElementById('menu-check-legend');
	if (lb) lb.querySelector('.menu-icon').textContent = State._legendOpen ? '✓' : '';
	var tb = document.getElementById('menu-check-timeline');
	if (tb) tb.querySelector('.menu-icon').textContent = State._ganttPaneOpen ? '✓' : '';
	var sb = document.getElementById('menu-check-source');
	if (sb) sb.querySelector('.menu-icon').textContent = State._sourcePaneOpen ? '✓' : '';
	var cp = document.getElementById('menu-check-code-paths');
	if (cp) cp.querySelector('.menu-icon').textContent = State._codePaths ? '✓' : '';
	var db = document.getElementById('menu-check-dark');
	if (db) db.querySelector('.menu-icon').textContent = State._darkMode ? '✓' : '';
	// hljs theme icons are owned by SourceCodeView (window.setHljsTheme /
	// window._applyHljsTheme wired in graph.js).
	if (State.sourceView) State.sourceView.syncThemeIcons();
}

function openInfoPane() {
	var rightEl = document.getElementById('right_pane');
	if (State._infoPaneRatio > 0 && rightEl) {
		var h = Math.round(State._infoPaneRatio * rightEl.offsetHeight);
		document.getElementById('info_window')?.style.setProperty('--flex-basis', h + 'px');
	} else {
		document.getElementById('info_window')?.style.setProperty('--flex-basis', '10em');
	}
}

// Script run within the webview itself.
(function () {

	// Get a reference to the VS Code webview api.
	// We use this API to post messages back to our extension.

	// @ts-ignore
	const vscode = acquireVsCodeApi();

	window.skevu = window.skevu || {};
	window.skevu.vscode = vscode;

	load();

	document.addEventListener('click', () => {
		menuClose();
		window.skevu.vscode.postMessage({
			type: 'focus_skevu_editor'
		});
	});

	document.getElementById('file-menu')?.addEventListener('click', event => {
		event.stopPropagation();
	});

	document.getElementById('load-trace-btn')?.addEventListener('click', () => {
		//loadFiles();
		menuClose();
	});

	document.querySelectorAll('.menu-title').forEach(button => {
		button.addEventListener('click', event => {
			menuToggle(event, button);
		});
	});

	document.querySelectorAll('.menu-radio').forEach(button => {
		button.addEventListener('click', () => {
			menuSelectRadio(button);
		});
	});

	document.querySelectorAll('.menu-toggle-check').forEach(button => {
		button.addEventListener('click', () => {
			menuToggleCheck(button);
		});
	});

	document.getElementById('menu-action-apply')?.addEventListener('click', () => {
		fetchViewDataAndRender();
		menuClose();
	});

	document.getElementById('menu-action-restore-defaults')?.addEventListener('click', () => {
		menuRestoreDefaults();
		fetchViewDataAndRender();
	});

	document.getElementById('menu-action-reset-view')?.addEventListener('click', () => {
		recenterGraph();
		menuClose();
	});

	document.getElementById('menu-action-save-image')?.addEventListener('click', () => {
		renderImage();
		menuClose();
	});

	document.getElementById('menu-check-legend')?.addEventListener('click', () => {
		toggleLegend();
		menuClose();
	});

	document.getElementById('menu-check-timeline')?.addEventListener('click', () => {
		toggleGanttPane();
		menuClose();
	});

	document.getElementById('menu-check-source')?.addEventListener('click', () => {
		toggleSourcePane();
		menuClose();
	});

	document.getElementById('menu-check-code-paths')?.addEventListener('click', () => {
		toggleCodePaths();
		menuClose();
	});

	document.querySelectorAll('.hljs-theme-radio').forEach(button => {
		button.addEventListener('click', event => {
			setHljsTheme(event.currentTarget?.dataset.theme);
			menuClose();
    	});
	});

	document.getElementById('sim-btn-playpause')?.addEventListener('click', () => {
		simToggle();
		menuClose();
	});

	document.getElementById('sim-btn-stop')?.addEventListener('click', () => {
		simStop();
		menuClose();
	});

	document.getElementById('gantt-btn-zoom')?.addEventListener('click', () => {
		zoomGanttToMeasuredInterval();
		menuClose();
	});

	document.getElementById('btn-next-source-file')?.addEventListener('click', () => {
		var tabs = Array.from(document.querySelectorAll('.cpp-tab'));
		if (tabs.length > 1) {
			var cur = tabs.findIndex((t) => {
				return t.classList.contains('active');
			});
			var next = (cur+1) % tabs.length;
			switchTab(tabs[next], next);
		}
		menuClose();
	});

	document.getElementById('btn-collapse-recursive')?.addEventListener('click', () => {
		collapseSelectedRecursively();
		menuClose();
	});

	document.getElementById('btn-expand-recursive')?.addEventListener('click', () => {
		expandSelectedRecursively();
		menuClose();
	});

	document.getElementById('btn-collapse-once')?.addEventListener('click', () => {
		collapseSelected();
		menuClose();
	});

	document.getElementById('btn-expand-once')?.addEventListener('click', () => {
		expandSelected();
		menuClose();
	});

	document.getElementById('gantt-timeline-scale-div')?.addEventListener('click', event => {
		event.stopPropagation();
	});

	document.getElementById('menu-check-dark')?.addEventListener('click', () => {
		toggleDarkMode();
		menuClose();
	});

	document.getElementById('btn-collapse-all-regions')?.addEventListener('click', () => {
		collapseAllRegions();
		menuClose();
	});

	document.getElementById('btn-expand-all-regions')?.addEventListener('click', () => {
		expandAllRegions();
		menuClose();
	});

	document.getElementById('btn-collapse-all-fusions')?.addEventListener('click', () => {
		collapseAllFusions();
		menuClose();
	});

	document.getElementById('btn-expand-all-fusions')?.addEventListener('click', () => {
		expandAllFusions();
		menuClose();
	});

	document.querySelectorAll('.spectrum-radio').forEach(button => {
		button.addEventListener('click', event => {
			colorSpectrumChanged(event.currentTarget?.dataset.spectrum);
			menuClose();
    	});
	});

	document.getElementById('render-indicator-btn-stop-rendering')?.addEventListener('click', () => {
		stopRendering();
	});

	document.getElementById('cy-stop-btn')?.addEventListener('click', () => {
		stopRendering();
	});

	document.querySelectorAll('.menu-title').forEach(function(btn) {
		btn.addEventListener('mouseenter', function() {
			if (document.querySelector('.menu.open')) {
				menuClose();
				btn.closest('.menu').classList.add('open');
			}
		});
	});

	/**
	 * Render the document in the webview.
	 */
	function updateContent(/** @type {string} */ text) {
		loadFiles(text);
		State.provider?.on('data', (rawData, minmax) => {
			window.skevu.vscode.postMessage({
				type: 'status_message',
				text: rawData
			});
			window.skevu.vscode.postMessage({
				type: 'build_documents',
				rawData: rawData
			});
		});
		/*let json;
		try {
			if (!text) {
				text = '{}';
			}
			json = JSON.parse(text);
		} catch {
			notesContainer.style.display = 'none';
			errorContainer.innerText = 'Error: Document is not valid json';
			errorContainer.style.display = '';
			return;
		}
		notesContainer.style.display = '';
		errorContainer.style.display = 'none';

		// Render the scratches
		notesContainer.innerHTML = '';
		for (const note of json.scratches || []) {
			const element = document.createElement('div');
			element.className = 'note';
			notesContainer.appendChild(element);

			const text = document.createElement('div');
			text.className = 'text';
			const textContent = document.createElement('span');
			textContent.innerText = note.text;
			text.appendChild(textContent);
			element.appendChild(text);

			const created = document.createElement('div');
			created.className = 'created';
			created.innerText = new Date(note.created).toUTCString();
			element.appendChild(created);

			const deleteButton = document.createElement('button');
			deleteButton.className = 'delete-button';
			deleteButton.addEventListener('click', () => {
				vscode.postMessage({ type: 'delete', id: note.id, });
			});
			element.appendChild(deleteButton);
		}

		notesContainer.appendChild(addButtonContainer);*/
	}

	// Handle messages sent from the extension to the webview
	window.addEventListener('message', event => {
		const message = event.data; // The json data that the extension sent
		switch (message.type) {
			case 'update':
				const text = message.text;

				// Update our webview's content
				updateContent(text);

				// Then persist state information.
				// This state is returned in the call to `vscode.getState` below when a webview is reloaded.
				vscode.setState({ text });

				return;
			
			case 'file_content':
				//cppFileResolveReady(message.text);
                return;

			case 'file_error':
				//cppFileRejectReady(message.text);
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

	// Webviews are normally torn down when not visible and re-created when they become visible again.
	// State lets us save information across these re-loads
	const state = vscode.getState();
	if (state) {
		updateContent(state.text);
	}
}());