import * as vscode from 'vscode';
import { getHtmlForWebview, hashContent } from './util.js';


interface TextBadge {
	nodeID: string,
	line: number,
	count: number,
	type: string
}

class TraceCodeLensProvider implements vscode.CodeLensProvider {

	badges: Array<TextBadge> = [];
	constructor(badges: Array<TextBadge>) {
		this.badges = badges;
	}

    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
		const lenses: Array<vscode.CodeLens> = [];

		for (const badge of this.badges) {
			const range = new vscode.Range(badge.line, 0, badge.line, 0);

			lenses.push(new vscode.CodeLens(range, {
				title: `${badge.count} | Highlight Node`,
				command: 'skevu.highlightNode',
				arguments: [badge.nodeID, document.fileName]
			}))
		}

		return lenses;
    }
}

interface SkePUTextDocument {
	highlightRange: Array<vscode.Range>;
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

export class SkevuEditorProvider implements vscode.CustomTextEditorProvider {

	textDocuments: Map<string, SkePUTextDocument> = new Map();
	relevantTextDocuments: Map<string, SkePUTextDocumentData> = new Map();
	cppFileContent: string = "";
	traceLineRanges: Array<Array<number>> = [];
	private readonly highlightDecoration = vscode.window.createTextEditorDecorationType({
		backgroundColor: new vscode.ThemeColor("editor.selectionHighlightBackground"), // See https://code.visualstudio.com/api/references/theme-color for options
		isWholeLine: true
	});

	public static register(context: vscode.ExtensionContext): vscode.Disposable {
		const provider = new SkevuEditorProvider(context);
		const providerRegistration = vscode.window.registerCustomEditorProvider(SkevuEditorProvider.viewType, provider);
		return providerRegistration;
	}

	private static readonly viewType = 'skepu.skevu';

	constructor(
		private readonly context: vscode.ExtensionContext
	) { }

	/**
	 * Called when our custom editor is opened.
	 * 
	 * 
	 */
	public async resolveCustomTextEditor(
		document: vscode.TextDocument,
		webviewPanel: vscode.WebviewPanel,
		_token: vscode.CancellationToken
	): Promise<void> {
		// Setup initial content for the webview
		webviewPanel.webview.options = {
			enableScripts: true,
		};
		webviewPanel.webview.html = getHtmlForWebview(this.context, webviewPanel.webview);

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

		this.context.subscriptions.push(this.highlightDecoration);


		const skepuShowTraceCommand = 
			vscode.commands.registerCommand(
				'skevu.highlightNode',
				(nodeID, fileName) => {
					console.log(`Clicked ${nodeID} from ${fileName}`);
					webviewPanel.webview.postMessage({
						type: 'highlight_node',
						node_id: nodeID
					});
					this.highlightOtherEditors(fileName, nodeID);
				}
			);


		// Make sure we get rid of the listener when our editor is closed.
		webviewPanel.onDidDispose(() => {
			changeDocumentSubscription.dispose();
			this.clearHighlights();
			changeActiveTextEditorSubscription.dispose();
			skepuShowTraceCommand.dispose();
			this.textDocuments.forEach(document => {
				document.badges?.dispose();
			})
		});

		// Receive message from the webview.
		webviewPanel.webview.onDidReceiveMessage(async e => {
			switch (e.type) {
				case 'status_message':
					console.log(e.text);
					return;

				case 'source_file':
					const textDocument = this.addTextDocumentToHighlight(e.file_name);
					textDocument.hash = e.file_content_hash;
					const uri = vscode.Uri.file(e.file_name);
					const cppMtime = (await vscode.workspace.fs.stat(uri)).mtime;
					const traceCtime = (await vscode.workspace.fs.stat(document.uri)).ctime;
					if (cppMtime > traceCtime) {
						for (const editor of vscode.window.visibleTextEditors) {
							if (editor.document.fileName == e.file_name) {
								this.validateFile(editor.document);
							}
						}
					}
					for (let [key, value] of this.relevantTextDocuments.entries()) {
						console.log(`${key} : ${value}`);
					}
					return;
				
				case 'trace_line_ranges':
					this.traceLineRanges = e.text;
					console.log(this.traceLineRanges);
					return;

				case 'highlight_trace_file':
					this.highlightTraceFile(e.node_id);
					return;

				case 'highlight_source_file':
					this.highlightSourceFile(e.source_file, e.node_id);
					return;

				case 'clear_highlight':
					this.clearHighlights();
					return;

				case 'build_documents':
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

	private async validateFile(document: vscode.TextDocument) {
		if (document.fileName.endsWith(".cpp")
			|| document.fileName.endsWith(".hpp")
			|| document.fileName.endsWith(".cc")
			|| document.fileName.endsWith(".h")) {
			let fileHash = hashContent(document.getText());
			console.log(`${document.fileName} hash: ${fileHash}`);
			const originalHash = this.relevantTextDocuments.get(document.fileName)?.hash;
			if (!originalHash) return;
			if (fileHash != originalHash) {
				console.log(`${fileHash} != ${originalHash}`);
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
				highlightRange: []
			}
			if (relevantTextDocument) {
				textDocument.highlightRange = relevantTextDocument.currentHighlight;
				if (relevantTextDocument.badges) {
					textDocument.badges = vscode.languages.registerCodeLensProvider(
						{pattern: fileName},
						new TraceCodeLensProvider(relevantTextDocument.badges)
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
		const traceFile = [...this.textDocuments.keys()].find(file => file.endsWith(".skevu"));
		if (!traceFile) return;

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

	private highlightSourceFile(sourceFileEnd: string, nodeID: string) {
		console.log(`Requested to highlight ${sourceFileEnd} at ${nodeID}`);
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

		if (document.highlightRange.length > 0) {
			editor.revealRange(document.highlightRange[0], vscode.TextEditorRevealType.InCenterIfOutsideViewport);
		}
	}

	private clearHighlights() {
		this.textDocuments.forEach((document, fileName) => {
			document.highlightRange = [];
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
			for (const node of rawData.nodes) {
				const data = node.data;
				
				if (fileName.endsWith(".skevu")) {
					documentData.highlightableRanges.set(data.id, new vscode.Range(
							this.traceLineRanges[data.trace_index][0] - 1, 0,
							this.traceLineRanges[data.trace_index][1] - 1, 0));
					badges.push({
						nodeID: data.id,
						line: this.traceLineRanges[data.trace_index][0] - 1,
						count: data.iteration_count,
						type: data.type
					});
				}
				else if ((data.type == "skeleton_call" || data.type == "external") && fileName.endsWith(data.file)) {
					documentData.highlightableRanges.set(data.id, new vscode.Range(
						data.line - 1, 0,
						data.line - 1, 0
					));
					badges.push({
						nodeID: data.id,
						line: data.line - 1,
						count: data.iteration_count,
						type: data.type
					})
				}
			}
			document.badges?.dispose();
			documentData.badges = badges;
			document.badges = vscode.languages.registerCodeLensProvider(
				{pattern: fileName},
				new TraceCodeLensProvider(badges)
			);
		});
	}

	/**
	 * Get the static html used for the editor webviews.
	 */
	
}
