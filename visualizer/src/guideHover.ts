import * as vscode from 'vscode';
import { afterSkepuNamespace, GUIDE_COMMANDS, guideHoverMarkdown, skeletonAnchor, skeletonOfDetail } from './guideLinks.js';

/** The name of the skeleton of an instance that is declared at the position: the outline of the language server has it ("Skeletons", with the kind in the detail). */
async function skeletonOfInstance(document: vscode.TextDocument, name: string, line: number): Promise<string | undefined> {
	const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[] | undefined>('vscode.executeDocumentSymbolProvider', document.uri);
	const group = symbols?.find(symbol => symbol.name === 'Skeletons');
	const instance = group?.children?.find(symbol => symbol.name === name && symbol.selectionRange.start.line === line);
	return skeletonOfDetail(instance?.detail);
}

/**
 * The hover of a skeleton (skepu::MapReduce) or of an instance of one (at its declaration) has a link to the section of the user guide about it. A language
 * server cannot make a link that works: the hover of the editor does not follow the links of a server, only the commands of a trusted extension.
 */
export function registerGuideHover(): vscode.Disposable {
	return vscode.languages.registerHoverProvider([{ language: 'cpp' }, { language: 'cuda-cpp' }], {
		async provideHover(document, position) {
			const range = document.getWordRangeAtPosition(position);
			if (!range) return;
			const word = document.getText(range);
			const before = document.lineAt(range.start.line).text.slice(0, range.start.character);
			const skeleton = afterSkepuNamespace(before) ? (skeletonAnchor(word) ? word : undefined) : await skeletonOfInstance(document, word, range.start.line);
			if (!skeleton) return;
			const markdown = new vscode.MarkdownString(guideHoverMarkdown(skeletonAnchor(skeleton)!, skeleton));
			markdown.isTrusted = { enabledCommands: GUIDE_COMMANDS };
			return new vscode.Hover(markdown, range);
		},
	});
}
