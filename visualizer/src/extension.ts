import * as vscode from 'vscode';
import { SkevuEditorProvider } from './skevuEditor.js';
import { GuidePanel } from './guide.js';
import { parseGuideUri } from './guideLinks.js';
import { registerSkeletonHover } from './skeletonHover.js';

export function activate(context: vscode.ExtensionContext) {
	// Register our custom editor providers
	context.subscriptions.push(...SkevuEditorProvider.register(context));

	// The user guide: by commands, and by links such as vscode://skepu.skepu-visualizer/guide?anchor=map, and from the hover of a skeleton (skeletonHover.ts)
	context.subscriptions.push(
		registerSkeletonHover(),
		vscode.commands.registerCommand('skevu.openGuide', (anchor?: string) => GuidePanel.command(context, anchor)),
		vscode.commands.registerCommand('skevu.openGuideInWindow', (anchor?: string) => GuidePanel.command(context, anchor, true)),
		vscode.window.registerUriHandler({
			handleUri(uri: vscode.Uri) {
				const link = parseGuideUri(uri.path, uri.query);
				if (link) return GuidePanel.show(context, link.anchor);
			},
		}),
	);
}
