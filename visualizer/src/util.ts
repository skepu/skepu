import { ExtensionContext, Webview, Uri } from 'vscode';

import { createHash, randomBytes } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { buildWebviewHtml } from "./webviewHtml.js";

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
	return randomBytes(24).toString('base64url');
}

/** The page of the webview: index.html of the extension, made for the webview (see webviewHtml.ts). */
export function getHtmlForWebview(context: ExtensionContext, webview: Webview): string {
	const uri = (...path: string[]) => webview.asWebviewUri(Uri.joinPath(context.extensionUri, ...path)).toString();
	const indexHtml = readFileSync(Uri.joinPath(context.extensionUri, 'index.html').fsPath, 'utf8');

	return buildWebviewHtml(indexHtml, {
		baseUri: uri(),
		cspSource: webview.cspSource,
		nonce: getNonce(),
		entryScript: 'dist/skevu.js',
		skevu: {
			dagreWorker: uri('dist', 'dagre-worker.js'),
			elkWorker: uri('static', 'cytoscape.js-elk-2.3.0', 'dist', 'elk.bundled.js'),
			visualizerFile: uri('dist', 'visualizer.js'),
			skepuLib: uri('static', 'skepu-lib') + '/',
			skepuDNN: uri('static', 'skepu-lib', 'dnn') + '/',
		},
	});
}
