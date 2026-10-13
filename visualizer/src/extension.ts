import * as vscode from 'vscode';
import { SkevuEditorProvider } from './skevuEditor.js';

export function activate(context: vscode.ExtensionContext) {
	// Register our custom editor providers
	context.subscriptions.push(SkevuEditorProvider.register(context));
}