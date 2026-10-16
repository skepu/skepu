// What the trace says about a skeleton instance, for its hover. No editor in it, so that it can be tested (tests/unit/test_trace_facts.js).

/** The data of a call node of the graph that the hover uses (durations are in microseconds). */
export interface CallData {
	id: string;
	type: string;
	label?: string;
	line?: number;
	duration?: number;
	durations?: number[];
	iteration_count?: number;
	backend?: string;
	elements?: number | string;
}

export interface CallFacts {
	calls: number;
	total: number;
	mean: number;
	max: number;
	backends: string[];
	elements?: string;
}

/** A time in microseconds for a person: 850 µs, 1.2 ms, 3.4 s. */
export function formatMicroseconds(us: number): string {
	if (us < 1000) return `${Math.round(us)} µs`;
	if (us < 1e6) return `${+(us / 1000).toPrecision(3)} ms`;
	return `${+(us / 1e6).toPrecision(3)} s`;
}

/** The call nodes of the graph of an instance (the label of a call is the name of its instance, unless the program named it otherwise). */
export function callsOfInstance(nodes: CallData[], name: string): CallData[] {
	return nodes.filter(node => node.type === 'skeleton_call' && node.label === name);
}

/** The times of the calls that a node stands for (a node of iterations that were coalesced has one for each). */
function timesOf(node: CallData): number[] {
	if (node.durations?.length) return node.durations;
	return node.duration !== undefined ? [node.duration] : [];
}

/** The calls of the nodes together: how many, how long, on which backends. Null if the nodes have no time (nothing to say). */
export function summarizeCalls(nodes: CallData[]): CallFacts | null {
	const times = nodes.flatMap(timesOf);
	if (!times.length) return null;
	const total = times.reduce((a, b) => a + b, 0);
	const elements = nodes.find(node => node.elements !== undefined)?.elements;
	return {
		calls: times.length,
		total,
		mean: total / times.length,
		max: Math.max(...times),
		backends: [...new Set(nodes.map(node => node.backend).filter((b): b is string => !!b))],
		elements: elements === undefined ? undefined : String(elements),
	};
}

/** The line of facts: "3 calls · total 1.2 ms · mean 400 µs · longest 800 µs · OpenMP · 2048 elements". */
export function describeCalls(facts: CallFacts): string {
	const parts = [`${facts.calls} ${facts.calls === 1 ? 'call' : 'calls'}`, `total ${formatMicroseconds(facts.total)}`];
	if (facts.calls > 1) parts.push(`mean ${formatMicroseconds(facts.mean)}`, `longest ${formatMicroseconds(facts.max)}`);
	if (facts.backends.length) parts.push(facts.backends.join(', '));
	if (facts.elements !== undefined) parts.push(`${facts.elements} elements`);
	return parts.join(' · ');
}

/** The Markdown of the hover for the calls: the facts, and a link that shows them in the viewer (the command of the badges, which goes on to the next node at each click). */
export function callsMarkdown(name: string, facts: CallFacts, command: { fileName: string; line: number; nodeIDs: string[]; sessionId: number }): string {
	const args = encodeURIComponent(JSON.stringify([command.fileName, command.line, command.nodeIDs, command.sessionId]));
	return `**${name}**: ${describeCalls(facts)} · [Show in SkeVU](command:skevu.highlightNode?${args})`;
}
