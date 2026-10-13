// The page of the webview: index.html (the page of the browser, the one source of the viewer's markup) with what is different in an editor.
// Pure text to text, so that it can be tested without an editor (tests/unit/test_webview_html.js).

export interface WebviewPage {
	/** The URL of the folder of the extension, as the webview has it: the relative URLs of the page are relative to it. */
	baseUri: string;
	/** Where the webview can load from (webview.cspSource). */
	cspSource: string;
	/** Allows the scripts of the page (and only them) to run. */
	nonce: string;
	/** What the scripts of the viewer need to know of the extension: a URL for each (window.skevu). */
	skevu: { [key: string]: string };
	/** The script of the viewer, relative to the folder of the extension. */
	entryScript: string;
}

/** The policy of the webview: scripts only with the nonce, nothing from outside the extension. Inline styles are allowed: the viewer sets colours and sizes in style attributes, and the graph library makes a style element (with scripts locked and nothing loadable from outside, a style cannot do harm). */
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

/** The part of the markup that is for a browser (marked data-variant="web"), and the links that would take the webview away from the viewer, are not in the page of an editor. */
function withoutBrowserParts(html: string): string {
	return html
		// scripts for the browser, with a source or inline
		.replace(/[ \t]*<script\b[^>]*\bdata-variant="web"[^>]*>[\s\S]*?<\/script>[ \t]*\n?/g, '')
		// the logo links to the page itself; the designer is another page of the browser
		.replace(/<a class="menu-anfang" href="\.">([\s\S]*?)<\/a>/, '<span class="menu-anfang">$1</span>')
		.replace(/[ \t]*<div id="welcome-designer-section">[\s\S]*?<\/div>\s*<\/div>/, '</div>');
}

export function buildWebviewHtml(indexHtml: string, page: WebviewPage): string {
	const { baseUri, cspSource, nonce } = page;
	let html = withoutBrowserParts(indexHtml);

	// the scripts that are left are the libraries and early.js: classic scripts of the extension, which the nonce lets run
	html = html.replace(/<script\b(?![^>]*\bnonce=)/g, `<script nonce="${nonce}"`);

	const config = Object.assign({ traceLineRanges: [] }, page.skevu);
	const head = [
		`<meta http-equiv="Content-Security-Policy" content="${contentSecurityPolicy(cspSource, nonce)}">`,
		`<base href="${baseUri.replace(/\/?$/, '/')}">`,
	].join('\n    ');
	// (the policy and the base are before everything that has a URL; the settings of the viewer are before the scripts that read them)
	html = html.replace(/<head>/, `<head>\n    ${head}`);
	html = html.replace(/<\/head>/, `    <script nonce="${nonce}">window.skevu = ${JSON.stringify(config).replace(/</g, '\\u003c')};</script>\n</head>`);
	html = html.replace(/<\/body>/, `    <script nonce="${nonce}" src="${page.entryScript}"></script>\n</body>`);
	return html;
}
