import { ExtensionContext, Webview, Uri } from 'vscode';

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

export function hashContent(content: string): string {
	return createHash("sha256").update(content, "utf8").digest("hex");
}

export function hashFile(path: string): Promise<string> {
    return new Promise((resolve, reject) => {
        const hash = createHash("sha256");
        const stream = createReadStream(path);

        stream.on("data", chunk => hash.update(chunk));
        stream.on("end", () => resolve(hash.digest("hex")));
        stream.on("error", reject);
    });
}

function getNonce() {
	let text = '';
	const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
	for (let i = 0; i < 32; i++) {
		text += possible.charAt(Math.floor(Math.random() * possible.length));
	}
	return text;
}

export function getHtmlForWebview(context: ExtensionContext, webview: Webview): string {
	// Local path to script and css for the webview
	const scriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'dist', 'skevu.js'));

	const styleMainUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'style', 'main.css'));

	/*const styleResetUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'media', 'reset.css'));

	const styleVSCodeUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'media', 'vscode.css'));*/

	const styleSkevuUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'style', 'skevu.css'));

	const styleFontUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/cmu-font/fonts/cmuserif_roman', 'stylesheet.css'));

	const jqueryScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/jquery', 'jquery-3.7.1.min.js'));

	const jqueryScrollToScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/jquery-scrollto', 'jquery.scrollTo.min.js'));

	const cytoscapeScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/cytoscape', 'cytoscape.min.js'));

	const dagreScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/dagre', 'dagre.min.js'));

	const cytoscapeDagreScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/cytoscape-dagre', 'cytoscape-dagre.js'));

	const cytoscapeElkScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/cytoscape.js-elk-2.3.0/dist', 'elk.bundled.js'));

	const hljsStyleUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/highlight/styles', 'default.css'));

	const hljsScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/highlight', 'highlight.min.js'));

	const hljsLineNumbersScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/static/highlight-line-numbers', 'highlightjs-line-numbers.min.js'));

	const dagreWorkerScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'dist', 'dagre-worker.js')).toString();

	const visualizerScriptUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'dist', 'visualizer.js')).toString();

	const skepuLibUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/skepu-lib')).toString();
	
	const skepuDNNUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'script/skepu-lib/dnn')).toString();

	const skevuLogoUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'SkeVU logo.png')).toString();

	const skevuLogoWhiteUri = webview.asWebviewUri(Uri.joinPath(
		context.extensionUri, 'SkeVU logo-white.png')).toString();

	// Use a nonce to whitelist which scripts can be run
	const nonce = getNonce();

	return /* html */`
		<!DOCTYPE html>
		<html lang="en">
		<head>
			<meta charset="UTF-8">

			<!--
			Use a content security policy to only allow loading images from https or from our extension directory,
			and only allow scripts that have a specific nonce.
			-->
			<meta http-equiv="Content-Security-Policy" content="default-src 'none'; font-src ${webview.cspSource}; connect-src ${webview.cspSource}; worker-src blob:; img-src ${webview.cspSource}; style-src ${webview.cspSource} 'nonce-${nonce}'; script-src 'nonce-${nonce}';">

			<meta name="viewport" content="width=device-width, initial-scale=1.0">

			
			<link href="${styleMainUri}" rel="stylesheet" />
			<link href="${styleFontUri}" rel="stylesheet" />
			<link href="${styleSkevuUri}" rel="stylesheet" />

			<title>Cat Scratch</title>

			<!-- jQuery -->
			<script nonce="${nonce}" src="${jqueryScriptUri}"></script>
			<script nonce="${nonce}" src="${jqueryScrollToScriptUri}"></script>


			<!-- Cytoscape -->
			<script nonce="${nonce}" src="${cytoscapeScriptUri}"></script>
			<script nonce="${nonce}" src="${dagreScriptUri}"></script>
			<script nonce="${nonce}" src="${cytoscapeDagreScriptUri}"></script>
			<script nonce="${nonce}" src="${cytoscapeElkScriptUri}"></script>


			<!-- Highlight.js -->
			<link id="hljs-theme" rel="stylesheet" href="${hljsStyleUri}">
			<script nonce="${nonce}" src="${hljsScriptUri}"></script>
			<script nonce="${nonce}" src="${hljsLineNumbersScriptUri}"></script>

			<script nonce="${nonce}">
				window.skevu = {
					dagreWorker: ${JSON.stringify(dagreWorkerScriptUri)},
					elkWorker: ${JSON.stringify(cytoscapeElkScriptUri)},
					visualizerFile: ${JSON.stringify(visualizerScriptUri)},
					skepuLib: ${JSON.stringify(skepuLibUri)},
					skepuDNN: ${JSON.stringify(skepuDNNUri)},
					traceLineRanges: []
				};
			</script>

		</head>
		<body class="welcome-mode">

			<div class="outer_container">
			<div id="left_pane">
				<nav id="menubar">

				<div class="menu"><a class="menu-anfang" href=".">
					<img src="${skevuLogoUri}" alt="SkePU logo" class="logo logo-small">
					<img src="${skevuLogoWhiteUri}" alt="SkePU logo" class="logo logo-small logo-dark">
				</a></div>

				
				<!-- View menu -->
				<div class="menu">
					<button class="menu-title">View</button>
					<div class="dropdown">
						<div class="submenu">
							<div class="submenu-title">Graph Mode<span class="menu-arrow">▸</span></div>
							<div class="dropdown">
								<button class="menu-radio" data-ctrl="graph-mode" data-value="dependence-dag"><span class="menu-icon">✓</span>Dependence DAG</button>
								<button class="menu-radio" data-ctrl="graph-mode" data-value="program-tree"><span class="menu-icon"></span>Program Tree</button>
							</div>
						</div>
					<div class="menu-sep"></div>
					<button id="menu-action-apply" class="menu-action">Apply Settings<span class="menu-key">Space</span></button>
					<button id="menu-action-restore-defaults" class="menu-action">Restore Defaults</button>
					<button id="menu-action-reset-view" class="menu-action">Reset View</button>
					<button id="menu-action-save-image" class="menu-action">Save Image</button>
					<div class="menu-sep"></div>
					<button id="menu-check-legend"   class="menu-check"><span class="menu-icon"></span>Legend</button>
					<button id="menu-check-timeline" class="menu-check"><span class="menu-icon"></span>Timeline<span class="menu-key">G</span></button>
					<button id="menu-check-source"     class="menu-check"><span class="menu-icon"></span>Source Panel<span class="menu-key">S</span></button>
					<button id="menu-check-code-paths" class="menu-check"><span class="menu-icon"></span>Code Paths</button>
					<div class="menu-sep"></div>
					<div id="gantt-timeline-scale-div" class="menu-form-row">
						<label id="gantt-timeline-scale-label">Timeline Scale: <span id="gantt-scale-label">1×</span></label>
						<input type="range" id="gantt-scale" min="1" max="20" step="1" value="1"
						oninput="ganttScaleChanged(this.value)">
					</div>
					<div class="menu-sep"></div>
					<div class="submenu">
						<button class="submenu-title menu-action">Source Code Theme<span class="menu-arrow">▶</span></button>
						<div class="dropdown">
						<button class="hljs-theme-radio" data-theme="default">          <span class="menu-icon">✓</span>Default (light)</button>
						<button class="hljs-theme-radio" data-theme="github">           <span class="menu-icon"></span>GitHub (light)</button>
						<button class="hljs-theme-radio" data-theme="intellij-light">   <span class="menu-icon"></span>IntelliJ (light)</button>
						<button class="hljs-theme-radio" data-theme="vs">               <span class="menu-icon"></span>Visual Studio (light)</button>
						<div class="menu-sep"></div>
						<button class="hljs-theme-radio" data-theme="atom-one-dark">    <span class="menu-icon"></span>Atom One Dark</button>
						<button class="hljs-theme-radio" data-theme="github-dark">      <span class="menu-icon"></span>GitHub Dark</button>
						<button class="hljs-theme-radio" data-theme="monokai">          <span class="menu-icon"></span>Monokai</button>
						<button class="hljs-theme-radio" data-theme="nord">             <span class="menu-icon"></span>Nord</button>
						<button class="hljs-theme-radio" data-theme="tokyo-night-dark"> <span class="menu-icon"></span>Tokyo Night</button>
						<button class="hljs-theme-radio" data-theme="vs2015">           <span class="menu-icon"></span>VS 2015 (dark)</button>
						</div>
					</div>
					<button id="menu-check-dark" class="menu-check"><span class="menu-icon"></span>Dark mode<span class="menu-key">M</span></button>
					</div>
				</div>

				<!-- Nodes menu -->
				<div class="menu">
					<button class="menu-title">Nodes</button>
					<div class="dropdown">
					<button class="menu-check menu-toggle-check" data-ctrl="container-updates"><span class="menu-icon"></span>Updates<span class="menu-key">U</span></button>
					<button class="menu-check menu-toggle-check" data-ctrl="container-allocations"><span class="menu-icon"></span>Allocations<span class="menu-key">A</span></button>
					<button class="menu-check menu-toggle-check" data-ctrl="container-deallocations"><span class="menu-icon"></span>Deallocations<span class="menu-key">D</span></button>
					<button class="menu-check menu-toggle-check" data-ctrl="container-transfers"><span class="menu-icon"></span>Transfers<span class="menu-key">T</span></button>
					<button class="menu-check menu-toggle-check" data-ctrl="element-accesses"><span class="menu-icon"></span>Element Accesses</button>
					<div class="menu-sep"></div>
					<button class="menu-check menu-toggle-check" data-ctrl="show_regions"><span class="menu-icon"></span>Regions<span class="menu-key">R</span></button>
					<div class="menu-sep"></div>
					<div class="submenu">
						<div class="submenu-title">Color<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button class="menu-radio" data-ctrl="node-color-call" data-value="duration-total"><span class="menu-icon">✓</span>Total Time</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="elements"><span class="menu-icon"></span>Elements</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="duration-per-element"><span class="menu-icon"></span>Time/element</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="iterations"><span class="menu-icon"></span>Iterations</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="duration-cv"><span class="menu-icon"></span>Duration Variance</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="dag-depth"><span class="menu-icon"></span>DAG Depth</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="nesting-level"><span class="menu-icon"></span>Nesting Level</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="fan-in"><span class="menu-icon"></span>Fan-In</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="fan-out"><span class="menu-icon"></span>Fan-Out</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="total-order"><span class="menu-icon"></span>Program Sequence</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="pattern"><span class="menu-icon"></span>Pattern</button>
						<button class="menu-radio" data-ctrl="node-color-call" data-value="backend"><span class="menu-icon"></span>Backend</button>
						</div>
					</div>
					<div class="submenu">
						<div class="submenu-title">Size<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button class="menu-radio" data-ctrl="node-size-call" data-value="fixed"><span class="menu-icon">✓</span>Fixed</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="duration-total"><span class="menu-icon"></span>Total Time</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="elements"><span class="menu-icon"></span>Elements</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="duration-per-element"><span class="menu-icon"></span>Time/element</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="iterations"><span class="menu-icon"></span>Iterations</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="duration-cv"><span class="menu-icon"></span>Duration Variance</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="nesting-level"><span class="menu-icon"></span>Nesting Level</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="fan-in"><span class="menu-icon"></span>Fan-In</button>
						<button class="menu-radio" data-ctrl="node-size-call" data-value="fan-out"><span class="menu-icon"></span>Fan-Out</button>
						</div>
					</div>
					</div>
				</div>

				<!-- Edges menu -->
				<div class="menu">
					<button class="menu-title">Edges</button>
					<div class="dropdown">
					<button class="menu-check menu-toggle-check" data-ctrl="anti-deps"><span class="menu-icon"></span>Anti-Dependences<span class="menu-key">N</span></button>
					<button class="menu-check menu-toggle-check" data-ctrl="virtual-alias-edges"><span class="menu-icon"></span>Virtual Datasets</button>
					<button class="menu-check menu-toggle-check" data-ctrl="edge-labels"><span class="menu-icon"></span>Labels<span class="menu-key">L</span></button>
					<div class="menu-sep"></div>
					<div class="submenu">
						<div class="submenu-title">Direction<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button class="menu-radio" data-ctrl="direction" data-value="vertical"><span class="menu-icon">✓</span>Vertical</button>
						<button class="menu-radio" data-ctrl="direction" data-value="horizontal"><span class="menu-icon"></span>Horizontal</button>
						</div>
					</div>
					<div class="submenu">
						<div class="submenu-title">Width<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button class="menu-radio" data-ctrl="edge-width" data-value="fixed"><span class="menu-icon">✓</span>Fixed</button>
						<button class="menu-radio" data-ctrl="edge-width" data-value="elements"><span class="menu-icon"></span>Elements</button>
						<button class="menu-radio" data-ctrl="edge-width" data-value="iterations"><span class="menu-icon"></span>Iterations</button>
						<button class="menu-radio" data-ctrl="edge-width" data-value="order-span"><span class="menu-icon"></span>Program Order Span</button>
						<button class="menu-radio" data-ctrl="edge-width" data-value="critical-path"><span class="menu-icon"></span>Critical Path</button>
						<button class="menu-radio" data-ctrl="edge-width" data-value="cross-backend"><span class="menu-icon"></span>Cross-Backend</button>
						</div>
					</div>
					<div class="submenu">
						<div class="submenu-title">Opacity<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button class="menu-radio" data-ctrl="edge-opacity" data-value="fixed"><span class="menu-icon">✓</span>Fixed</button>
						<button class="menu-radio" data-ctrl="edge-opacity" data-value="elements"><span class="menu-icon"></span>Elements</button>
						<button class="menu-radio" data-ctrl="edge-opacity" data-value="iterations"><span class="menu-icon"></span>Iterations</button>
						<button class="menu-radio" data-ctrl="edge-opacity" data-value="order-span"><span class="menu-icon"></span>Program Order Span</button>
						<button class="menu-radio" data-ctrl="edge-opacity" data-value="critical-path"><span class="menu-icon"></span>Critical Path</button>
						<button class="menu-radio" data-ctrl="edge-opacity" data-value="cross-backend"><span class="menu-icon"></span>Cross-Backend</button>
						</div>
					</div>
					</div>
				</div>

				<!-- Actions menu -->
				<div class="menu">
					<button class="menu-title">Actions</button>
					<div class="dropdown">
					<div class="submenu">
						<div class="submenu-title">Simulate<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button id="sim-btn-playpause" class="menu-action">Play<span class="menu-key">P</span></button>
						<button id="sim-btn-stop" class="menu-action">Stop</button>
						</div>
					</div>
					<div class="menu-sep"></div>
					<button id="gantt-btn-zoom" class="menu-action">Zoom to Measured Interval<span class="menu-key">Z</span></button>
					<button id="btn-next-source-file" class="menu-action">Next Source Tab<span class="menu-key">Tab</span></button>
					<div class="menu-sep"></div>
					<div class="submenu">
						<div class="submenu-title">Selected Node<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button id="btn-collapse-recursive" class="menu-action">Collapse Recursive<span class="menu-key">C</span></button>
						<button id="btn-expand-recursive" class="menu-action">Expand Recursive<span class="menu-key">E</span></button>
						<button id="btn-collapse-once" class="menu-action">Collapse Once<span class="menu-key">-</span></button>
						<button id="btn-expand-once" class="menu-action">Expand Once<span class="menu-key">+</span></button>
						</div>
					</div>
					<div class="submenu">
						<div class="submenu-title">All Regions<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button id="btn-collapse-all-regions" class="menu-action">Collapse</button>
						<button id="btn-expand-all-regions" class="menu-action">Expand</button>
						</div>
					</div>
					<div class="submenu">
						<div class="submenu-title">All Fusions<span class="menu-arrow">▸</span></div>
						<div class="dropdown">
						<button id="btn-collapse-all-fusions" class="menu-action">Collapse</button>
						<button id="btn-expand-all-fusions" class="menu-action">Expand</button>
						</div>
					</div>
					</div>
				</div>

				<!-- Color Spectrum menu -->
				<div class="menu">
					<button class="menu-title">Colors</button>
					<div class="dropdown dropdown-spectrum">
					<button class="spectrum-radio" data-spectrum="default">  <span class="menu-icon">✓</span>Red–Green <span class="spectrum-swatch" id="default-spectrum-background"></span></button>
					<button class="spectrum-radio" data-spectrum="viridis">  <span class="menu-icon"></span>Viridis   <span class="spectrum-swatch" id="viridis-spectrum-background"></span></button>
					<button class="spectrum-radio" data-spectrum="plasma">   <span class="menu-icon"></span>Plasma    <span class="spectrum-swatch" id="plasma-spectrum-background"></span></button>
					<button class="spectrum-radio" data-spectrum="inferno">  <span class="menu-icon"></span>Inferno   <span class="spectrum-swatch" id="inferno-spectrum-background"></span></button>
					<button class="spectrum-radio" data-spectrum="cividis">  <span class="menu-icon"></span>Cividis   <span class="spectrum-swatch" id="cividis-spectrum-background"></span></button>
					<button class="spectrum-radio" data-spectrum="cool-warm"><span class="menu-icon"></span>Cool–Warm <span class="spectrum-swatch" id="cool-warm-spectrum-background"></span></button>
					<button class="spectrum-radio" data-spectrum="grayscale"><span class="menu-icon"></span>Grayscale <span class="spectrum-swatch" id="grayscale-spectrum-background"></span></button>
					</div>
				</div>

				<!-- Advanced menu -->
				<div class="menu">
					<button class="menu-title">Advanced</button>
					<div class="dropdown">
					<button class="menu-check menu-toggle-check" data-ctrl="collapse_iteration"><span class="menu-icon"></span>Aggregate iteration<span class="menu-key">I</span></button>
					<button class="menu-check menu-toggle-check" data-ctrl="fusion_analysis"><span class="menu-icon"></span>Fusion analysis<span class="menu-key">F</span></button>
					</div>
				</div>

				<!-- Render indicator -->
				<div id="render-indicator" class="menu">
					<button class="menu-title"
							id="render-indicator-button">
					<span class="render-spinner"></span>Rendering…
					</button>
					<div class="dropdown dropdown-info" id="render-indicator-dropdown-div">
					<button id="render-indicator-btn-stop-rendering" class="menu-action">Stop rendering</button>
					</div>
				</div>

				<!-- Info button -->
				<div class="menu menu-info-btn">
					<button class="menu-title menu-info-title" title="Trace statistics">ⓘ</button>
					<div class="dropdown dropdown-info">
					<p><strong>Trace events:</strong> <span id="event-count">—</span></p>
					<p><strong>Nodes:</strong> <span id="node-count">—</span></p>
					<p><strong>Edges:</strong> <span id="edge-count">—</span></p>
					<p><strong>Gantt bars:</strong> <span id="gantt-count">—</span></p>
					<p><strong>Snapshots:</strong> <span id="snapshot-count">—</span></p>
					<p><strong>Analysis time:</strong> <span id="modeling-time">—</span> ms</p>
					<p><strong>Rendering time:</strong> <span id="rendering-time">—</span> ms</p>
					<p><strong>Total response time:</strong> <span id="response-time">—</span> ms</p>
					<hr class="menu-sep">
					<p><a href="https://skepu.github.io" target="_blank">SkePU website</a></p>
					</div>
				</div>

				</nav>
				<div id="alert-bar"></div>

				<script nonce=${nonce}>
				// ── Menu helpers ──────────────────────────────────────────────────────
				

				
				

				// switchTab / setHljsTheme / _applyHljsTheme / badgeClick are
				// window globals wired by SourceCodeView in graph.js.

				

				// menuRestoreDefaults needs to call into settings.js — that module
				// exports it via window in ui.js, but in worker mode we also need it.
				// It is wired up to window in settings.js → ui.js, so it will be
				// available after the modules initialise.
				</script>

				<div id="visualization">
				<!-- Welcome panel — visible before the first trace is loaded.
					Hidden by worker-loader.js after a successful load. -->
				<div id="welcome-panel">
					<div class="welcome-card">
					<img src="${skevuLogoUri}" alt="SkePU logo" class="logo welcome-logo">
					<img src="${skevuLogoWhiteUri}" alt="SkePU logo" class="logo welcome-logo logo-dark">
					<h1>SkeVU — SkePU Visualizer</h1>
					<p class="welcome-subtitle">Performance visualizer for <a href="https://skepu.github.io" target="_blank">SkePU</a> programs</p>

					<h3>Load your own trace</h3>
					<ol>
						<li>Build your program with the precompiler and <code>-DSKEPU_TRACING</code>.</li>
						<li>A <code>trace.json</code> is written to the working directory after execution.</li>
						<li>Use <strong>Files › Load Trace</strong> to upload the JSON and any source files.</li>
					</ol>

					<div id="welcome-examples-section">
						<h3>Load a built-in example</h3>
						<div id="welcome-examples-list"></div>
					</div>
					</div>
				</div>

				<div id="cy"></div>
				<div id="sim-progress">
					<div id="sim-ticks"></div>
					<div id="sim-progress-fill"></div>
					<div id="sim-progress-handle"></div>
				</div>
				<div id="cy-layout-overlay">
					<span class="cy-layout-spinner"></span>
					<button id="cy-stop-btn">&#9632; Stop</button>
				</div>
				</div>
				<div id="legend"></div>
			</div>

			<div id="split-gutter"></div>

			<!-- Source / info pane -->
			<div id="right_pane">
				<div id="cpp_container">
				<!-- Tab bar and tab panes are empty on first load;
					worker-loader.js populates them after each trace load. -->
				<div id="cpp_tab_bar"></div>
				<div id="cpp_tab_panes">
					<div class="cpp-tab-pane active">
					<pre id="cpp-tab-pre">Load a trace file or select an example from the Files menu to get started.</pre>
					</div>
				</div>
				</div>
				<div id="info_window">
				<div id="info-resizer"></div>
				<h3 id="info-heading"></h3>
				<div id="info-content">
					<div id="info"></div>
					<div id="heatmap-container"></div>
					<div id="plot"></div>
				</div>
				</div>
			</div>
			</div>

			<div id="gantt-pane">
			<div id="gantt-svg-container"></div>
			</div>

			<script nonce="${nonce}" src="${scriptUri}"></script>
		</body>
		</html>`;
}