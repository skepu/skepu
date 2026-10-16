import * as vscode from 'vscode';
import { afterSkepuNamespace, GUIDE_COMMANDS, guideHoverMarkdown, skeletonAnchor, skeletonOfDetail } from './guideLinks.js';
import { callsMarkdown, summarizeCalls } from './traceFacts.js';
import { SkevuEditorProvider } from './skevuEditor.js';

/** The name of the skeleton of an instance that is declared at the position: the outline of the language server has it ("Skeletons", with the kind in the detail). */
async function skeletonOfInstance(document: vscode.TextDocument, name: string, line: number): Promise<string | undefined> {
	const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[] | undefined>('vscode.executeDocumentSymbolProvider', document.uri);
	const group = symbols?.find(symbol => symbol.name === 'Skeletons');
	const instance = group?.children?.find(symbol => symbol.name === name && symbol.selectionRange.start.line === line);
	return skeletonOfDetail(instance?.detail);
}

function trusted(text: string): vscode.MarkdownString {
	const markdown = new vscode.MarkdownString(text);
	markdown.isTrusted = { enabledCommands: [...GUIDE_COMMANDS, 'skevu.highlightNode'] };
	return markdown;
}

/**
 * What the extension adds to the hover of the editor for SkePU. A language server cannot make links that work (the hover follows only the commands of a trusted
 * extension), and it does not know the traces that are open. So, next to the hover of the language server:
 *  - a skeleton (skepu::MapReduce) and an instance at its declaration have a link to the section of the user guide about it;
 *  - an instance that the open traces have calls of has what they say (how many, how long, where), and a link that shows the calls in the viewer.
 */
export function registerSkeletonHover(): vscode.Disposable {
	return vscode.languages.registerHoverProvider([{ language: 'cpp' }, { language: 'cuda-cpp' }], {
		async provideHover(document, position) {
			const range = document.getWordRangeAtPosition(position);
			if (!range) return;
			const word = document.getText(range);
			const before = document.lineAt(range.start.line).text.slice(0, range.start.character);
			const contents: vscode.MarkdownString[] = [];

			const skeleton = afterSkepuNamespace(before) ? (skeletonAnchor(word) ? word : undefined) : await skeletonOfInstance(document, word, range.start.line);
			if (skeleton) contents.push(trusted(guideHoverMarkdown(skeletonAnchor(skeleton)!, skeleton)));

			// (the calls on this line if there are, else all the calls of the instance: at its declaration, say)
			for (const { sessionId, calls } of SkevuEditorProvider.current?.callsOfInstance(document.fileName, word) ?? []) {
				const here = calls.filter(call => call.line === range.start.line + 1);
				const shown = here.length ? here : calls;
				const facts = summarizeCalls(shown);
				if (facts) contents.push(trusted(callsMarkdown(word, facts,
					{ fileName: document.fileName, line: range.start.line, nodeIDs: shown.map(call => call.id), sessionId })));
			}
			return contents.length ? new vscode.Hover(contents, range) : undefined;
		},
	});
}
