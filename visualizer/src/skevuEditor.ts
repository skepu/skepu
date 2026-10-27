import * as vscode from 'vscode';
import { basename } from 'node:path';
import { getHtmlForWebview, hashContent, hashFile } from './util.js';
import { callsOfInstance, CallData } from './traceFacts.js';
import { linesOfNode, nodesWithBadges, pathEndsWith, traceSuffix, lensesOfBadges, lensTitle, nextNodeIndex, NodeData } from './nodeLines.js';


/** What the extension did with the source files of a trace (View > Output > SkeVU): which were found, which were not, and the badges made. */
const output = vscode.window.createOutputChannel('SkeVU');

interface TextBadge {
	nodeID: string,
	line: number,
	count: number,
	type: string
}

class TraceCodeLensProvider implements vscode.CodeLensProvider {

	badges: Array<TextBadge> = [];
	constructor(badges: Array<TextBadge>, private readonly sessionId: number) {
		this.badges = badges;
	}

    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
		const lenses: Array<vscode.CodeLens> = [];

		for (const lens of lensesOfBadges(this.badges)) {
			lenses.push(new vscode.CodeLens(new vscode.Range(lens.line, 0, lens.line, 0), {
				title: lensTitle(lens),
				command: 'skevu.highlightNode',
				arguments: [document.fileName, lens.line, lens.nodeIDs, this.sessionId]
			}));
		}

		return lenses;
    }
}

interface SkePUTextDocument {
	highlightRange: Array<vscode.Range>;
	/** The places of the selected call (instance, user functions, the call), lightly shaded. Lines with the strong highlight are not shaded again. */
	placeRange: Array<vscode.Range>;
	badges?: vscode.Disposable;
}

interface SkePUTextDocumentData {
	highlightableRanges: Map<string, vscode.Range>;
	currentHighlight: Array<vscode.Range>;
	badges?: Array<TextBadge>;
	hash?: string;
}

interface SkePUStuff {
	data: SkePUTextDocumentData;
	instance?: SkePUTextDocument;
}

/** One open editor: a trace, the source files that it names, and what is highlighted and badged in them. Editors do not share any of it. */
class SkevuSession {


	textDocuments: Map<string, SkePUTextDocument> = new Map();
	relevantTextDocuments: Map<string, SkePUTextDocumentData> = new Map();
	cppFileContent: string = "";
	/** The file of the trace that the editor shows: it has the lines of the events, which the nodes highlight. */
	traceFileName: string = "";
	traceLineRanges: Array<Array<number>> = [];
	/** The graph that the viewer drew last (what the badges are made of). */
	private lastGraph: any = undefined;
	/** Whether the viewer shows the regions: they have badges (lenses) only if it does, as in the viewer. */
	private showRegions = false;

	/** The call nodes of the instance with this name, if the file is a source file of this trace (found, with its hash). */
	public callsOfInstance(fileName: string, name: string): CallData[] {
		if (!this.lastGraph || !this.textDocuments.has(fileName)) return [];
		return callsOfInstance((this.lastGraph.nodes ?? []).map((node: { data: CallData }) => node.data), name);
	}
	private readonly highlightDecoration = vscode.window.createTextEditorDecorationType({
		backgroundColor: new vscode.ThemeColor("editor.selectionHighlightBackground"), // See https://code.visualstudio.com/api/references/theme-color for options
		isWholeLine: true
	});

	private readonly placeDecoration = vscode.window.createTextEditorDecorationType({
		backgroundColor: new vscode.ThemeColor("editor.wordHighlightBackground"),
		isWholeLine: true
	});

	private panel?: vscode.WebviewPanel;

	constructor(
		private readonly context: vscode.ExtensionContext,
		public readonly id: number
	) { }

	/** Show the trace that is the text of the document, in the panel. */
	public async open(
		document: vscode.TextDocument,
		webviewPanel: vscode.WebviewPanel
	): Promise<void> {
		this.panel = webviewPanel;
		// Setup initial content for the webview
		webviewPanel.webview.options = {
			enableScripts: true,
		};
		webviewPanel.webview.html = getHtmlForWebview(this.context, webviewPanel.webview);
		this.traceFileName = document.fileName;

		function updateWebview() {
			webviewPanel.webview.postMessage({
				type: 'update',
				text: document.getText(),
			});
		}
		

		// Hook up event handlers so that we can synchronize the webview with the text document.
		//
		// The text document acts as our model, so we have to sync change in the document to our
		// editor and sync changes in the editor back to the document.
		// 
		// Remember that a single text document can also be shared between multiple custom
		// editors (this happens for example when you split a custom editor)

		const changeDocumentSubscription = vscode.workspace.onDidChangeTextDocument(event => {
			if (event.document.uri.toString() === document.uri.toString()) {
				updateWebview();
			} else if (this.relevantTextDocuments.has(event.document.fileName) && event.contentChanges.length > 0) {
				this.validateFile(event.document);
			}
		});

		let previousEditors = vscode.window.visibleTextEditors;

		const changeActiveTextEditorSubscription = vscode.window.onDidChangeVisibleTextEditors(editors => {
			const added = editors.filter(
				editor => !previousEditors.includes(editor)
			);

			previousEditors = editors;

			for (const editor of added) {
				this.highlightRelevantEditor(editor);
			}
		})

		//vscode.window.onDidChangeVisibleTextEditors


		// Make sure we get rid of the listener when our editor is closed.
		webviewPanel.onDidDispose(() => {
			changeDocumentSubscription.dispose();
			this.clearHighlights();
			changeActiveTextEditorSubscription.dispose();
			this.highlightDecoration.dispose();
			this.placeDecoration.dispose();
			this.textDocuments.forEach(document => {
				document.badges?.dispose();
			})
		});

		// Receive message from the webview.
		webviewPanel.webview.onDidReceiveMessage(async e => {
			switch (e.type) {
				case 'status_message':
					return;

				case 'source_file':
					await this.registerSourceFile(e.file_name, e.file_content_hash);
					return;

				case 'trace_line_ranges':
					this.traceLineRanges = e.text;
					return;

				case 'highlight_trace_file':
					this.highlightTraceFile(e.node_id);
					return;

				case 'highlight_source_file':
					this.highlightSourceFile(e.source_file, e.node_id);
					return;

				case 'reveal_source':
					await this.revealSource(e.source_file, e.line, e.end_line);
					return;

				case 'show_definitions':
					this.showPlaces(e.marks);
					return;

				case 'clear_highlight':
					this.clearHighlights();
					return;

				case 'build_documents':
					this.lastGraph = e.rawData;
					this.showRegions = !!e.showRegions;
					this.buildDocuments(e.rawData);
					return;

				case 'focus_skevu_editor':
					webviewPanel.reveal(webviewPanel.viewColumn);
					return;
			}
		});

		updateWebview();

		this.addTextDocumentToHighlight(document.fileName);
	}

	/** The files of the trace (as it names them) that were found, for the links of the info pane. */
	private readonly sourceFiles = new Set<string>();

	/** The place of each lens in the nodes that it selects (the click goes on to the next of them). */
	private readonly lensNode = new Map<string, number>();

	/** A lens was clicked in a source file: the viewer selects the next node of the line, and the other files that have it highlight it. */
	public highlightNode(fileName: string, line: number, nodeIDs: string[]) {
		const place = `${fileName}:${line}`;
		const index = nextNodeIndex(this.lensNode.get(place), nodeIDs.length);
		this.lensNode.set(place, index);
		const nodeID = nodeIDs[index];
		this.panel?.webview.postMessage({
			type: 'highlight_node',
			node_id: nodeID
		});
		this.highlightOtherEditors(fileName, nodeID);
	}

	/** Whether the file at the path is the source file of the trace: there, and (for a trace that has the hash) with the contents of that hash. */
	private async isSourceFile(path: string, hash?: string): Promise<boolean> {
		try {
			if (hash) return await hashFile(path) === hash;
			await vscode.workspace.fs.stat(vscode.Uri.file(path));
			return true;
		} catch {
			return false; // (not there)
		}
	}

	/**
	 * The source file of the trace that has this hash. The path in the trace is where the program was compiled: it is the file if it is there with
	 * that hash. Only if it is not (the trace is from another machine, or the file moved) is the workspace searched, by the name of the file: a big
	 * workspace takes seconds to search, which a file at its path does not have to wait for. A trace without a hash takes the file at the path.
	 * Nothing is highlighted in a file that has changed since the trace was made.
	 */
	private async registerSourceFile(fileName: string, hash?: string) {
		const found: string[] = [];
		if (await this.isSourceFile(fileName, hash)) {
			found.push(fileName);
		} else {
			// (the files of the workspace that are the file: with the hash, and the ones that have the path of the trace at their end come first and
			// are all used, as a header can be in the workspace twice, as a copy, and the nodes are of the file that has the path)
			const suffix = traceSuffix(fileName);
			const same: string[] = [], others: string[] = [];
			for (const uri of await vscode.workspace.findFiles(`**/${basename(fileName)}`, '**/node_modules/**', 20))
				if (await this.isSourceFile(uri.fsPath, hash))
					(pathEndsWith(uri.fsPath, suffix) ? same : others).push(uri.fsPath);
			found.push(...(same.length ? same : others.slice(0, 1)));
		}
		if (!found.length) {
			output.appendLine(`source file ${fileName}: no file with ${hash ? 'the hash ' + hash.slice(0, 12) : 'that path'} (not at its path, and not in the workspace under that name): no badges`);
			return;
		}
		this.sourceFiles.add(fileName);
		this.panel?.webview.postMessage({ type: 'source_files', files: [...this.sourceFiles] });
		for (const path of found) {
			output.appendLine(`source file ${fileName}: ${path === fileName ? 'at its path' : 'found at ' + path}${hash ? ' (hash ' + hash.slice(0, 12) + ' matches)' : ''}`);
			this.addTextDocumentToHighlight(path).hash = hash;
		}
		// (the badges of the graph that is shown were made without this file, if the graph came first)
		if (this.lastGraph) this.buildDocuments(this.lastGraph);
	}

	private async validateFile(document: vscode.TextDocument) {
		if (document.fileName.endsWith(".cpp")
			|| document.fileName.endsWith(".hpp")
			|| document.fileName.endsWith(".cc")
			|| document.fileName.endsWith(".h")) {
			let fileHash = hashContent(document.getText());
			const originalHash = this.relevantTextDocuments.get(document.fileName)?.hash;
			if (!originalHash) return;
			if (fileHash != originalHash) {
				this.removeTextDocumentToHighlight(document.fileName);
			} else {
				this.addTextDocumentToHighlight(document.fileName);
			}
		}
	}

	private addTextDocumentToHighlight(fileName: string): SkePUTextDocumentData {
		if (!this.textDocuments.has(fileName)) {
			const relevantTextDocument = this.relevantTextDocuments.get(fileName);
			const textDocument: SkePUTextDocument =
			{
				highlightRange: [],
				placeRange: []
			}
			if (relevantTextDocument) {
				textDocument.highlightRange = relevantTextDocument.currentHighlight;
				if (relevantTextDocument.badges) {
					textDocument.badges = vscode.languages.registerCodeLensProvider(
						{pattern: fileName},
						new TraceCodeLensProvider(relevantTextDocument.badges, this.id)
					);
				}
			} else {
				this.relevantTextDocuments.set(fileName,
				{
					highlightableRanges: new Map(),
					currentHighlight: []
				});
			}
			this.textDocuments.set(fileName, textDocument);
			this.highlightAllRelevantEditors();
			return this.relevantTextDocuments.get(fileName)!;
		}
		return this.relevantTextDocuments.get(fileName)!;
	}

	private removeTextDocumentToHighlight(textDocument: string) {
		if (this.textDocuments.has(textDocument)) {
			this.textDocuments.get(textDocument)?.badges?.dispose();
			this.clearHighlightsOf(textDocument);
			this.textDocuments.delete(textDocument);
			return true;
		}
		return false;
	}

	private highlightTraceFile(nodeID: string) {
		const traceFile = this.traceFileName;
		if (!this.textDocuments.has(traceFile)) return;

		const editors = vscode.window.visibleTextEditors.filter(editor => traceFile === editor.document.fileName);
		if (!editors) return;

		const document = this.textDocuments.get(traceFile);
		if (!document) return;

		const relevantTextDocument = this.relevantTextDocuments.get(traceFile);
		if (!relevantTextDocument) return;

		const highlightRange = relevantTextDocument.highlightableRanges.get(nodeID);
		if (!highlightRange) return;

		document.highlightRange = [highlightRange];
		relevantTextDocument.currentHighlight = [highlightRange];

		this.highlightRelevantEditors(editors);
	}

	/** A link to a definition was followed in the viewer: the lines are shown in the editor of the file (beside the viewer if it is not open). */
	private async revealSource(sourceFile: string, line: number, endLine?: number) {
		const suffix = traceSuffix(sourceFile);
		const fileName = [...this.textDocuments.keys()].find(file => pathEndsWith(file, suffix));
		if (!fileName || !(line >= 1)) return;
		const last = Math.max(line, endLine ?? line);
		const range = new vscode.Range(line - 1, 0, last - 1, 0);
		const open = vscode.window.visibleTextEditors.find(editor => editor.document.fileName === fileName);
		await vscode.window.showTextDocument(vscode.Uri.file(fileName), {
			viewColumn: open?.viewColumn ?? vscode.ViewColumn.Beside,
			selection: new vscode.Range(range.start, range.start),
			preserveFocus: true,
		}).then(editor => editor.revealRange(range, vscode.TextEditorRevealType.InCenterIfOutsideViewport));
		// The revealed lines get the strong highlight, in place of the call's.
		this.textDocuments.forEach((document, name) => {
			document.highlightRange = [];
			const data = this.relevantTextDocuments.get(name);
			if (data) data.currentHighlight = [];
		});
		const document = this.textDocuments.get(fileName);
		const data = this.relevantTextDocuments.get(fileName);
		if (!document || !data) return;
		document.highlightRange = [range];
		data.currentHighlight = [range];
		this.highlightAllRelevantEditors();
	}

	private highlightSourceFile(sourceFileEnd: string, nodeID: string) {
		const sourceFile = [...this.textDocuments.keys()].find(file => file.endsWith(sourceFileEnd));
		if (!sourceFile) return;

		const editors = vscode.window.visibleTextEditors.filter(editor => sourceFile === editor.document.fileName);
		if (!editors) return;

		const document = this.textDocuments.get(sourceFile);
		if (!document) return;

		const relevantTextDocument = this.relevantTextDocuments.get(sourceFile);
		if (!relevantTextDocument) return;

		const highlightRange = relevantTextDocument.highlightableRanges.get(nodeID);
		if (!highlightRange) return;

		document.highlightRange = [highlightRange];
		relevantTextDocument.currentHighlight = [highlightRange];

		this.highlightRelevantEditors(editors);
	}

	private highlightOtherEditors(sourceFileName: string, nodeID: string) {
		this.textDocuments.forEach((document, fileName) => {
			if (fileName != sourceFileName && this.relevantTextDocuments.get(fileName)?.highlightableRanges.has(nodeID)) {
				this.highlightSourceFile(fileName, nodeID);
			}
		})
	}

	private highlightAllRelevantEditors() {
		for (const editor of vscode.window.visibleTextEditors) 
			this.highlightRelevantEditor(editor);
	}

	private highlightRelevantEditors(editors: vscode.TextEditor[]) {
		for (const editor of editors) 
			this.highlightRelevantEditor(editor);
	}

	private highlightRelevantEditor(editor: vscode.TextEditor) {
		const document = this.textDocuments.get(editor.document.fileName);
		if(!document) return;

		editor.setDecorations(this.highlightDecoration, document.highlightRange);
		editor.setDecorations(this.placeDecoration, document.placeRange.filter(place => !document.highlightRange.some(strong => strong.isEqual(place))));

		if (document.highlightRange.length > 0) {
			editor.revealRange(document.highlightRange[0], vscode.TextEditorRevealType.InCenterIfOutsideViewport);
		}
	}

	/** Lightly shades the places of the selected call (`marks`: file and lines from 1) in the files that were found. No marks removes the shading. */
	private showPlaces(marks: Array<{ file: string; from: number; to: number }>) {
		this.textDocuments.forEach(document => { document.placeRange = []; });
		for (const mark of marks ?? []) {
			const suffix = traceSuffix(mark.file);
			const fileName = [...this.textDocuments.keys()].find(file => pathEndsWith(file, suffix));
			if (!fileName || !(mark.from >= 1)) continue;
			this.textDocuments.get(fileName)!.placeRange.push(new vscode.Range(mark.from - 1, 0, Math.max(mark.from, mark.to) - 1, 0));
		}
		this.highlightAllRelevantEditors();
	}

	private clearHighlights() {
		this.textDocuments.forEach((document, fileName) => {
			document.highlightRange = [];
			document.placeRange = [];
			const relevantTextDocument = this.relevantTextDocuments.get(fileName);
			if (relevantTextDocument) {
				relevantTextDocument.currentHighlight = [];
			}
		});
		this.highlightAllRelevantEditors();
	}

	private clearHighlightsOf(fileName: string) {
		const editors = vscode.window.visibleTextEditors.filter(
			editor => editor.document.fileName == fileName
		)

		const document = this.textDocuments.get(fileName);
		if (!document) return;
		document.highlightRange = [];
		document.placeRange = [];

		for (const editor of editors) {
			this.highlightRelevantEditor(editor);
		} 
	}

	private buildDocuments(rawData: any) {
		this.textDocuments.forEach((document, fileName) => {
			const documentData = this.relevantTextDocuments.get(fileName);
			if (!documentData) return;
			// Calls to this function will likely contain different IDs for the same nodes.
			documentData.highlightableRanges.clear();
			const badges: Array<TextBadge> = [];
			for (const node of nodesWithBadges<{ data: NodeData }>(rawData.nodes ?? [], this.showRegions)) {
				const data: NodeData = node.data;
				const lines = linesOfNode(data, fileName, fileName === this.traceFileName, this.traceLineRanges);
				if (!lines) continue;
				documentData.highlightableRanges.set(data.id, new vscode.Range(lines[0], 0, lines[1], 0));
				badges.push({
					nodeID: data.id,
					line: lines[0],
					count: data.iteration_count ?? 0,
					type: data.type
				});
			}
			output.appendLine(`badges in ${fileName}: ${badges.length} (from ${rawData.nodes?.length ?? 0} nodes)`);
			document.badges?.dispose();
			documentData.badges = badges;
			document.badges = vscode.languages.registerCodeLensProvider(
				{pattern: fileName},
				new TraceCodeLensProvider(badges, this.id)
			);
		});
	}
}

/** The editors of SkeVU: a session for each trace that is open (any number of them), and the command of the badges, which is for all of them. */
export class SkevuEditorProvider implements vscode.CustomTextEditorProvider {

	private readonly sessions = new Map<number, SkevuSession>();
	/** The provider of this window (the hover of a skeleton asks it for what the open traces say about an instance). */
	public static current?: SkevuEditorProvider;

	/** The calls of the instance with this name, in each open trace that has the file as its source. */
	public callsOfInstance(fileName: string, name: string): { sessionId: number; calls: CallData[] }[] {
		return [...this.sessions.values()]
			.map(session => ({ sessionId: session.id, calls: session.callsOfInstance(fileName, name) }))
			.filter(found => found.calls.length > 0);
	}
	private nextId = 1;

	/** The editor is opened by default for the files that the tracer names (skepu_trace*.json) and *.skevu, and by Open With for any other JSON file. */
	public static register(context: vscode.ExtensionContext): vscode.Disposable[] {
		const provider = new SkevuEditorProvider(context);
		SkevuEditorProvider.current = provider;
		return [
			vscode.window.registerCustomEditorProvider(SkevuEditorProvider.viewType, provider),
			// (a badge in a source file has the session that made it: a command is registered once, not for each editor)
			vscode.commands.registerCommand('skevu.highlightNode', (fileName: string, line: number, nodeIDs: string[], sessionId: number) =>
				provider.sessions.get(sessionId)?.highlightNode(fileName, line, nodeIDs)),
		];
	}

	private static readonly viewType = 'skepu.skevu';

	constructor(private readonly context: vscode.ExtensionContext) { }

	public async resolveCustomTextEditor(
		document: vscode.TextDocument,
		webviewPanel: vscode.WebviewPanel,
		_token: vscode.CancellationToken
	): Promise<void> {
		const id = this.nextId++;
		const session = new SkevuSession(this.context, id);
		this.sessions.set(id, session);
		webviewPanel.onDidDispose(() => this.sessions.delete(id));
		await session.open(document, webviewPanel);
	}
}
