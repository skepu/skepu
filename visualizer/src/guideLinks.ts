// The links to the user guide: what the guide has an anchor for, the address that a link has, and the page that is shown. No editor in it, so that
// it can be tested (tests/unit/test_guide.js).

/** The skeletons that the guide has a section for (a link from the hover of a skeleton names one of them). */
export const SKELETON_ANCHORS = ['map', 'reduce', 'mapreduce', 'scan', 'mapoverlap', 'mappairs', 'mappairsreduce', 'mappool'];

/**
 * Where each anchor is in the rendered book (user-guide/, in its simple edition): the page and the id of the section, as the sources have them
 * ({#sec-map} in src/programming.qmd, which the book makes src/programming.html#sec-map). tests/unit/test_guide.js checks them against the sources.
 */
export const GUIDE_SECTIONS: Record<string, string> = {
	map: 'src/programming.html#sec-map',
	reduce: 'src/programming.html#sec-reduce',
	mapreduce: 'src/programming.html#sec-mapreduce',
	scan: 'src/programming.html#sec-scan',
	mapoverlap: 'src/programming.html#sec-mapoverlap',
	mappairs: 'src/programming.html#sec-mappairs',
	mappairsreduce: 'src/programming.html#sec-mappairsreduce',
	mappool: 'src/programming.html#sec-mappool',
	programming: 'src/programming.html#sec-skeletons',
	skevu: 'src/skevu.html',
	lsp: 'src/lsp.html#sec-lsp',
};

/** All the anchors that a link can have. */
export const GUIDE_ANCHORS = Object.keys(GUIDE_SECTIONS);

/** The page of the book (and the section in it) for an anchor; the first page for none. */
export function guideTarget(anchor: string | undefined): string {
	return anchor !== undefined && Object.prototype.hasOwnProperty.call(GUIDE_SECTIONS, anchor) ? GUIDE_SECTIONS[anchor] : 'index.html';
}

/** The anchor of a link, if it is a name that the page can have (it goes into the page: only letters, digits and dashes). */
export function guideAnchor(value: string | null | undefined): string | undefined {
	return value && /^[a-z0-9-]{1,40}$/.test(value) ? value : undefined;
}

/** The anchor of a link of the form vscode://skepu.skepu-visualizer/guide?anchor=map (`path` and `query` of its URI); null if it is not a link to the guide. */
export function parseGuideUri(path: string, query: string): { anchor?: string } | null {
	if (path.replace(/\/+$/, '') !== '/guide') return null;
	return { anchor: guideAnchor(new URLSearchParams(query).get('anchor')) };
}

/** The anchor for a skeleton, from its name (MapOverlap2D: mapoverlap), if the guide has a section for it. */
export function skeletonAnchor(name: string | null | undefined): string | undefined {
	const anchor = (name ?? '').toLowerCase().replace(/[1-4]d$/, '');
	return SKELETON_ANCHORS.includes(anchor) ? anchor : undefined;
}

/** The name of the skeleton of an instance, from the detail that the outline of the language server gives it ("MapReduce(float, float)"), if the guide has a section for it. */
export function skeletonOfDetail(detail: string | null | undefined): string | undefined {
	const name = /^(\w+)\(/.exec(detail ?? '')?.[1];
	return skeletonAnchor(name) ? name : undefined;
}

/** Whether the text before a name ends with the namespace of SkePU (skepu::), which a skeleton is named with. */
export function afterSkepuNamespace(before: string): boolean {
	return /\bskepu::\s*$/.test(before);
}

/** The Markdown of the hover: the links that open the guide at the anchor, in the panel or in a window of its own (commands of the extension). */
export function guideHoverMarkdown(anchor: string, label: string): string {
	const link = (command: string) => `command:${command}?${encodeURIComponent(JSON.stringify([anchor]))}`;
	return `[SkePU guide: ${label}](${link('skevu.openGuide')}) · [in a window](${link('skevu.openGuideInWindow')})`;
}

export const GUIDE_COMMANDS = ['skevu.openGuide', 'skevu.openGuideInWindow'];

/** What a link in a page of the book does: stays in the page, goes to another page of the book, goes to the web, or does nothing. */
export type GuideLink =
	| { kind: 'page'; page: string; fragment: string }
	| { kind: 'external'; url: string }
	| { kind: 'none' };

/** What a click on a link in a page of the book is, told in the page itself (the function is put into the page: it must not use anything outside it). */
export type LinkClass = { kind: 'anchor'; id: string } | { kind: 'page'; rel: string } | { kind: 'web' };

/**
 * Where a link points, from the page's point of view. `href` is the attribute, `resolved` the address that the browser makes of it, `base` the base of the page
 * (its directory), `here` the address of the page and `root` the directory of the book. Quarto changes the links to a section of the page to addresses from the
 * base ("<directory>/#section"), and the addresses are compared by their paths, decoded, because the same address is not always written the same way.
 */
export function classifyLink(href: string, resolved: string, base: string, here: string, root: string): LinkClass {
	const decode = (text: string) => { try { return decodeURIComponent(text); } catch { return text; } };
	const url = new URL(resolved);
	const path = decode(url.pathname);
	if (href.charAt(0) === '#') return { kind: 'anchor', id: decode(href.slice(1)) };
	if (url.hash && (path === decode(new URL(base).pathname) || path === decode(new URL(here).pathname))) return { kind: 'anchor', id: decode(url.hash.slice(1)) };
	const all = decode(url.href), top = decode(root);
	// (a link to a directory, the book's own or one in it, is to the first page there: Quarto writes the link to index.html so)
	if (all.indexOf(top) === 0) return { kind: 'page', rel: all.slice(top.length) || 'index.html' };
	return { kind: 'web' };
}

/** The page of the book that has the link (`page`, such as src/lsp.html) and the link's address (`href`) make: another page of the book, with the fragment. */
export function resolveGuideLink(page: string, href: string): GuideLink {
	// (an address of the webview itself is never for the browser)
	if (/^https:\/\/[^/]*(vscode-cdn\.net|vscode-resource)/i.test(href)) return { kind: 'none' };
	if (/^https:\/\//i.test(href)) return { kind: 'external', url: href };
	if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('//') || href.startsWith('#') || !href) return { kind: 'none' };
	let url: URL;
	try { url = new URL(href, 'https://guide.invalid/' + page); } catch { return { kind: 'none' }; }
	let target = decodeURIComponent(url.pathname).replace(/^\/+/, '');
	if (target === '' || target.endsWith('/')) target += 'index.html';   // (a link to a directory is to its first page)
	// (only the pages of the book: no way out of it by .., which a URL does not leave at the top either, and nothing but .html)
	if (!/^[A-Za-z0-9_./-]+\.html$/.test(target) || target.split('/').includes('..')) return { kind: 'none' };
	return { kind: 'page', page: target, fragment: url.hash.replace(/^#/, '') };
}

/** The page and the fragment of a target of GUIDE_SECTIONS ("src/lsp.html#sec-lsp"). */
export function splitTarget(target: string): { page: string; fragment: string } {
	const [page, fragment = ''] = target.split('#');
	return { page, fragment };
}

export interface GuidePage {
	cspSource: string;
	/** The address of the directory of the page in the webview, which the links and files of the page are relative to. */
	baseUri: string;
	/** The address of the directory of the book in the webview: a link under it is a link to a page of the book. */
	rootUri: string;
	/** The page, from the root of the book (src/lsp.html): the links of the page are resolved from it. */
	page: string;
	/** The id of the section to scroll to. */
	fragment: string;
}

/**
 * A page of the book as the webview shows it: the page itself, with a policy, a base for its files, and a script that takes over the links (a link in a webview
 * does not go to another page: the extension is asked to, see resolveGuideLink) and goes to the section. The book is trusted content of the extension and has
 * scripts of its own (Quarto's, and MathJax from the web), so its inline scripts run; a policy with a nonce could not allow them.
 */
export function buildGuideHtml(pageHtml: string, page: GuidePage): string {
	// (what the page may load from the web: only MathJax, which does not hold the page up. The fonts of Google and the polyfill script that Quarto
	// puts first in a page are left out: a slow or missing network held the page back for many seconds, and they are not needed here)
	const web = 'https://cdn.jsdelivr.net';
	const csp = [
		`default-src 'none'`,
		`img-src ${page.cspSource} data: https:`,
		`font-src ${page.cspSource} data: ${web}`,
		`style-src ${page.cspSource} 'unsafe-inline' ${web}`,
		`script-src ${page.cspSource} 'unsafe-inline' 'unsafe-eval' ${web}`,
		`connect-src ${page.cspSource}`,
		`worker-src blob:`,
		`frame-src ${page.cspSource}`,   // (a figure that is a PDF is an <embed>)
		`object-src ${page.cspSource}`,
	].join('; ') + ';';
	const json = (text: string) => JSON.stringify(text).replace(/</g, '\\u003c');
	// (first in the page, before the scripts of the book: what goes wrong in the page is reported to the extension, which logs it: View > Output > SkeVU Guide)
	const early = `<script>
(function() {
  var vscode = window.guideVscode = acquireVsCodeApi();
  function report(what, detail) { try { vscode.postMessage({ type: 'report', what: what, detail: detail }); } catch (e) { /* (nobody to tell) */ } }
  window.addEventListener('error', function(e) { report('error', e.message + ' at ' + (e.filename || '?').split('/').slice(-2).join('/') + ':' + e.lineno + ':' + e.colno); });
  window.addEventListener('unhandledrejection', function(e) { report('rejection', String(e.reason)); });
  document.addEventListener('securitypolicyviolation', function(e) { report('policy', e.violatedDirective + ' blocked ' + e.blockedURI); });
  report('load', ${json(page.page)});
})();
</script>`;
	const head = `<meta http-equiv="Content-Security-Policy" content="${csp}">\n<base href="${page.baseUri.replace(/\/?$/, '/')}">\n${early}`;
	const script = `<script>
(function() {
  var vscode = window.guideVscode;
  var page = ${json(page.page)};
  function scrollTo(id) { var e = id && document.getElementById(id); if (e) e.scrollIntoView(); }
  // The links of the page: Quarto makes each of them an address from the base ("<directory of the page>/#section", "<directory>/other.html"), so they are
  // told apart by where they point. In the page: by scrolling. To another page of the book, or to the web: the extension is asked to.
  var root = ${json(page.rootUri.replace(/\/?$/, '/'))};
  var classify = ${classifyLink.toString()};
  document.addEventListener('click', function(event) {
    var a = event.target.closest && event.target.closest('a[href]');
    if (!a || event.defaultPrevented) return;
    if (a.hasAttribute('data-bs-toggle')) return;   // (a button of the page: it is not a link, and its script is to see it)
    var href = a.getAttribute('href');
    // The click is ours: the webview has a listener of its own that opens every link in the browser, which must not see it.
    event.preventDefault();
    event.stopPropagation();
    var c = classify(href, a.href, document.baseURI, location.href, root);
    if (c.kind === 'anchor') scrollTo(c.id);
    else if (c.kind === 'page') vscode.postMessage({ type: 'link', page: 'index.html', href: c.rel });
    else vscode.postMessage({ type: 'link', page: page, href: href });
  }, true);
  window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'scroll') scrollTo(event.data.fragment);
  });
  window.addEventListener('load', function() { scrollTo(${json(page.fragment)}); });
})();
</script>`;
	// (the polyfill is for old browsers, and a script of the web that is not deferred holds the page up until it is there)
	const html = pageHtml.replace(/<script\b[^>]*\bsrc="https:\/\/cdnjs\.cloudflare\.com\/polyfill[^>]*><\/script>\s*/gi, '');
	return html.replace(/<head>/i, `<head>\n${head}`).replace(/<\/body>/i, `${script}\n</body>`);
}
