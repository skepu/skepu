import * as vscode from 'vscode';
import { buildGuideHtml, GUIDE_ANCHORS, guideAnchor, guideTarget, resolveGuideLink, splitTarget } from './guideLinks.js';

/** The user guide in a panel of the editor: the book that is in the extension (guide/, the simple edition of user-guide/). One panel is used again for every link. */
export class GuidePanel {
	private static panel?: vscode.WebviewPanel;
	/** What the pages of the book report (a script error, a blocked file, a page that was loaded): View > Output > SkeVU Guide. */
	private static readonly log = vscode.window.createOutputChannel('SkeVU Guide');
	/** The page of the book that the panel shows (src/lsp.html). */
	private static page = '';

	/** Shows the guide at the anchor (the first page if none): in a panel beside the editor, or moved to a window of its own. */
	static async show(context: vscode.ExtensionContext, anchor?: string, ownWindow = false) {
		const guideDir = vscode.Uri.joinPath(context.extensionUri, 'guide');
		const { page, fragment } = splitTarget(guideTarget(guideAnchor(anchor)));
		const created = !GuidePanel.panel;
		if (!GuidePanel.panel) {
			try {
				await vscode.workspace.fs.stat(vscode.Uri.joinPath(guideDir, 'index.html'));
			} catch {
				vscode.window.showInformationMessage('The user guide is not a part of this build of the SkeVU extension (it is built with the extension: make vsix, with Quarto installed).');
				return;
			}
			const panel = vscode.window.createWebviewPanel('skepu.guide', 'SkePU User Guide', vscode.ViewColumn.Beside,
				{ enableScripts: true, localResourceRoots: [guideDir], retainContextWhenHidden: true });   // (a page is not loaded again when the tab is shown again)
			panel.onDidDispose(() => { GuidePanel.panel = undefined; GuidePanel.page = ''; });
			// (a link in a page of the book: another page, or the web)
			panel.webview.onDidReceiveMessage(message => {
				if (message?.type === 'report') {
					GuidePanel.log.appendLine(`${new Date().toISOString().slice(11, 23)} ${message.what}: ${String(message.detail).slice(0, 500)}`);
					return;
				}
				if (message?.type !== 'link' || typeof message.page !== 'string' || typeof message.href !== 'string') return;
				const link = resolveGuideLink(message.page, message.href);
				if (link.kind === 'page') GuidePanel.load(context, link.page, link.fragment);
				else if (link.kind === 'external') vscode.env.openExternal(vscode.Uri.parse(link.url));
			});
			GuidePanel.panel = panel;
		} else {
			GuidePanel.panel.reveal(GuidePanel.panel.viewColumn ?? vscode.ViewColumn.Beside);
		}
		await GuidePanel.load(context, page, fragment);
		if (ownWindow && created) await vscode.commands.executeCommand('workbench.action.moveEditorToNewWindow');
	}

	/** Shows the page of the book (the same page is only scrolled). */
	private static async load(context: vscode.ExtensionContext, page: string, fragment: string) {
		const panel = GuidePanel.panel;
		if (!panel) return;
		if (page === GuidePanel.page) {
			panel.webview.postMessage({ type: 'scroll', fragment });
			return;
		}
		const guideDir = vscode.Uri.joinPath(context.extensionUri, 'guide');
		const parts = page.split('/');
		try {
			const bytes = await vscode.workspace.fs.readFile(vscode.Uri.joinPath(guideDir, ...parts));
			panel.webview.html = buildGuideHtml(new TextDecoder().decode(bytes), {
				cspSource: panel.webview.cspSource,
				baseUri: panel.webview.asWebviewUri(vscode.Uri.joinPath(guideDir, ...parts.slice(0, -1))).toString(),
				rootUri: panel.webview.asWebviewUri(guideDir).toString(),
				page,
				fragment,
			});
			GuidePanel.page = page;
		} catch {
			vscode.window.showWarningMessage(`The user guide has no page ${page}.`);
		}
	}

	/** The command: the anchor is the argument, or is chosen from the list. */
	static async command(context: vscode.ExtensionContext, anchor?: string, ownWindow = false) {
		anchor ??= await vscode.window.showQuickPick(GUIDE_ANCHORS, { placeHolder: 'Section of the guide' });
		if (anchor) await GuidePanel.show(context, anchor, ownWindow);
	}
}
