// The webview page: index.html (the browser's page, the single source of the viewer's markup) adapted for an editor.
// Pure text to text, testable without an editor (tests/unit/test_webview_html.js).

export interface WebviewPage {
	/** The extension folder's URL as the webview sees it; the page's relative URLs resolve against it. */
	baseUri: string;
	/** Where the webview can load from (webview.cspSource). */
	cspSource: string;
	/** Allows the scripts of the page (and only them) to run. */
	nonce: string;
	/** Extension values for the viewer scripts: a URL each (window.skevu). */
	skevu: { [key: string]: string };
	entryScript: string;
}

/** The webview policy: scripts only with the nonce, nothing from outside the extension. Inline styles are allowed: the viewer sets colours and sizes in style attributes and the graph library creates a style element; with scripts locked and nothing loadable from outside, a style cannot do harm. */
export function contentSecurityPolicy(cspSource: string, nonce: string): string {
	return [
		`default-src 'none'`,
		`font-src ${cspSource}`,
		`connect-src ${cspSource}`,
		`worker-src blob:`,
		`img-src ${cspSource} data: blob:`,
		`style-src ${cspSource} 'unsafe-inline'`,
		`script-src 'nonce-${nonce}'`,
	].join('; ') + ';';
}

/** Drops what is for a browser (data-variant="web") and the links that would navigate the webview away from the viewer. */
function withoutBrowserParts(html: string): string {
	return html
		// scripts for the browser, with a source or inline
		.replace(/[ \t]*<script\b[^>]*\bdata-variant="web"[^>]*>[\s\S]*?<\/script>[ \t]*\n?/g, '')
		// the logo links to the page itself
		.replace(/<a class="menu-anfang" href="\.">([\s\S]*?)<\/a>/, '<span class="menu-anfang">$1</span>');
}

export function buildWebviewHtml(indexHtml: string, page: WebviewPage): string {
	const { baseUri, cspSource, nonce } = page;
	let html = withoutBrowserParts(indexHtml).replace(/<html\b/, '<html class="editor"');

	// the remaining scripts (libraries and early.js) are the extension's classic scripts; the nonce allows them
	html = html.replace(/<script\b(?![^>]*\bnonce=)/g, `<script nonce="${nonce}"`);

	const config = Object.assign({ traceLineRanges: [] }, page.skevu);
	const head = [
		`<meta http-equiv="Content-Security-Policy" content="${contentSecurityPolicy(cspSource, nonce)}">`,
		`<base href="${baseUri.replace(/\/?$/, '/')}">`,
	].join('\n    ');
	// the policy and the base are before everything that has a URL; the settings of the viewer are before the scripts that read them
	html = html.replace(/<head>/, `<head>\n    ${head}`);
	html = html.replace(/<\/head>/, `    <script nonce="${nonce}">window.skevu = ${JSON.stringify(config).replace(/</g, '\\u003c')};</script>\n</head>`);
	html = html.replace(/<\/body>/, `    <script nonce="${nonce}" src="${page.entryScript}"></script>\n</body>`);
	return html;
}
