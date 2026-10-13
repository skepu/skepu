// Which lines of a file a node of the graph marks. No editor in it, so that it can be tested (tests/unit/test_node_lines.js).

export interface NodeData {
	id: string;
	type: string;
	file?: string;
	line?: number;
	trace_index?: number;
	iteration_count?: number;
}

/**
 * What can be compared of the path of a file in a trace: the path as the compiler had it, which is relative to the directory that the program was
 * compiled in for a file that was found by a relative include (".//..//skepu-headers/src/skepu-lib/io.hpp"). Doubled and "." parts go, and so do
 * the ".." at the start of a relative path (they are where the program was compiled, which is not known here); an absolute path stays absolute.
 */
export function traceSuffix(path: string): string {
	const normal = path.replace(/\\/g, '/');
	const absolute = normal.startsWith('/') || /^[A-Za-z]:\//.test(normal);
	const parts: string[] = [];
	for (const part of normal.split('/')) {
		if (part === '' || part === '.') continue;
		if (part === '..') {
			if (parts.length && parts[parts.length - 1] !== '..') parts.pop();
			else if (!absolute) parts.push('..');
			continue;
		}
		parts.push(part);
	}
	while (!absolute && parts[0] === '..') parts.shift();
	return (normal.startsWith('/') ? '/' : '') + parts.join('/');
}

/** Whether the path of a file ends with the path of a source file of the trace, at the start of a name (o.cpp is not the end of foo.cpp). */
export function pathEndsWith(path: string, end: string): boolean {
	const p = path.replace(/\\/g, '/'), e = end.replace(/\\/g, '/');
	if (!e || !p.endsWith(e)) return false;
	return e.startsWith('/') || p.length === e.length || p[p.length - e.length - 1] === '/';
}

/**
 * The first and the last line (from 0) that a node marks in the file, or null if it marks none. In the file of the trace, the lines of the event
 * that the node is of. In a source file, the line of the call of a skeleton (or of an `external`) that is in that file: an event may have no file
 * or no line (an `external` made without C++20 has neither), and then it is in no file.
 */
export function linesOfNode(data: NodeData, fileName: string, isTraceFile: boolean, traceLineRanges: number[][]): [number, number] | null {
	if (isTraceFile) {
		const range = data.trace_index === undefined ? undefined : traceLineRanges[data.trace_index];
		return range ? [Math.max(0, range[0] - 1), Math.max(0, range[1] - 1)] : null;
	}
	if (data.type !== 'skeleton_call' && data.type !== 'external') return null;
	if (!data.file || !pathEndsWith(fileName, traceSuffix(data.file))) return null;
	if (!Number.isInteger(data.line) || (data.line as number) < 1) return null;
	return [(data.line as number) - 1, (data.line as number) - 1];
}
