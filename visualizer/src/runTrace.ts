import * as vscode from 'vscode';
import * as path from 'node:path';
import * as fs from 'node:fs';
import { execFile } from 'node:child_process';
import * as os from 'node:os';
import { MakefileKind, MakefileValues, listed, makefileKind, planRun, PRINT_RULES, printTargets, RunOverrides, RunPlan, sourceProgram } from './runPlan.js';
import { runOptionsHtml } from './runOptionsHtml.js';

/** What the user has set, for the runs of this window (not saved): what is not set is the Makefile's. */
let overrides: RunOverrides = {};
/** How each program (by path) was built last, as far as tracing goes: tracing and snapshots are flags of the build, so a change of them builds again. */
const built = new Map<string, { trace: boolean; snapshots: boolean }>();

const shellQuote = (word: string) => /^[\w@%+=:,./-]+$/.test(word) ? word : `'${word.replace(/'/g, `'\\''`)}'`;

const makefileOf = (dir: string) => ['Makefile', 'makefile', 'GNUmakefile'].map(name => path.join(dir, name)).find(fs.existsSync);

/** The source to run: the argument of the command (the explorer), or the file of the active editor. */
function sourceOf(uri?: vscode.Uri): vscode.Uri | undefined {
	return uri ?? vscode.window.activeTextEditor?.document.uri;
}

/** Builds (and, but for `buildOnly`, runs) the program of a source with the rules of the Makefile next to it, then shows its trace. */
async function runTrace(uri?: vscode.Uri, buildOnly = false): Promise<void> {
	const source = sourceOf(uri);
	if (!source || source.scheme !== 'file') return void vscode.window.showErrorMessage('SkeVU: open the source of a SkePU program.');
	const dir = path.dirname(source.fsPath);
	const makefile = makefileOf(dir);
	if (!makefile) return void vscode.window.showErrorMessage(`SkeVU: no Makefile in ${dir}.`);
	const document = vscode.workspace.textDocuments.find(d => d.uri.fsPath === source.fsPath);
	if (document?.isDirty) await document.save();

	const config = vscode.workspace.getConfiguration('skevu.run');
	const trace = overrides.trace !== false, snapshots = trace && overrides.snapshots === true;
	const before = built.get(source.fsPath);
	const changed = !!before && (before.trace !== trace || before.snapshots !== snapshots);
	const plan = planRun({
		makefile: fs.readFileSync(makefile, 'utf8'), source: source.fsPath,
		overrides: { ...overrides, ...(changed ? { trace, snapshots } : {}), rebuild: overrides.rebuild || changed },
		make: config.get<string>('make', 'make'), buildOnly,
	}, dir);
	if ('error' in plan) return void vscode.window.showErrorMessage(`SkeVU: ${plan.error}`);
	await execute(plan);
	built.set(source.fsPath, { trace, snapshots });
}

async function execute(plan: RunPlan): Promise<void> {
	const line = plan.commands.map(command => command.map(shellQuote).join(' ')).join(' && ');
	const task = new vscode.Task(
		{ type: 'shell' }, vscode.TaskScope.Workspace, `SkeVU: ${plan.program}`, 'skevu',
		new vscode.ShellExecution(line, { cwd: plan.cwd, env: plan.env }), '$gcc');
	task.presentationOptions = { reveal: vscode.TaskRevealKind.Always, clear: true, panel: vscode.TaskPanelKind.Dedicated };
	const execution = await vscode.tasks.executeTask(task);
	const code = await new Promise<number | undefined>(resolve => {
		const done = vscode.tasks.onDidEndTaskProcess(e => {
			if (e.execution === execution) { done.dispose(); resolve(e.exitCode); }
		});
	});
	if (code !== 0) return void vscode.window.showErrorMessage(`SkeVU: ${plan.program} failed${code === undefined ? '' : ` (exit code ${code})`}.`);
	if (!plan.trace) return;      // run without tracing: there is nothing to show
	const trace = vscode.Uri.file(path.join(plan.cwd, plan.trace));
	if (!fs.existsSync(trace.fsPath)) return void vscode.window.showErrorMessage(`SkeVU: ${plan.program} wrote no trace (${plan.trace}); was it built with tracing?`);
	await vscode.commands.executeCommand('vscode.openWith', trace, 'skepu.skevu', vscode.ViewColumn.Beside);
}

/** What the Makefile of the examples runs a program with (the values make has, not the text of the assignments); nothing if make cannot say. */
async function makefileValues(makefile: string, program: string, kind: MakefileKind): Promise<MakefileValues> {
	const rules = path.join(os.tmpdir(), 'skevu_print.mk');
	fs.writeFileSync(rules, PRINT_RULES);
	const make = vscode.workspace.getConfiguration('skevu.run').get<string>('make', 'make');
	const print = (target: string) => new Promise<string | undefined>(resolve =>
		execFile(make, ['-s', '-f', makefile, '-f', rules, target], { cwd: path.dirname(makefile), timeout: 10000 }, (error, out) => resolve(error ? undefined : out.trim())));
	const values = await Promise.all(printTargets(program, kind).map(print));
	return kind === 'examples' ? { backend: values[0], arguments: values[1] } : { arguments: values[0] };
}

let panel: vscode.WebviewPanel | undefined;
/** The source of the form (the form is one panel, for the source that it was opened from last). */
let formSource: vscode.Uri | undefined;

/** The options of the runs, in a form beside the editor: each is what the Makefile has until it is set. */
async function runOptions(uri?: vscode.Uri): Promise<void> {
	const source = sourceOf(uri);
	if (!source || source.scheme !== 'file') return void vscode.window.showErrorMessage('SkeVU: open the source of a SkePU program.');
	const program = sourceProgram(source.fsPath);
	const makefile = makefileOf(path.dirname(source.fsPath));
	const text = makefile ? fs.readFileSync(makefile, 'utf8') : '';
	const kind = makefileKind(text) ?? 'package';
	formSource = source;
	if (!panel) {
		panel = vscode.window.createWebviewPanel('skevu.runOptions', 'SkeVU run', { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true }, { enableScripts: true });
		panel.onDidDispose(() => { panel = undefined; });
		panel.webview.onDidReceiveMessage((message: { type: string, overrides: RunOverrides }) => {
			overrides = message.overrides;
			if (message.type === 'run') void runTrace(formSource);
			if (message.type === 'build') void runTrace(formSource, true);
		});
	}
	panel.title = `Options for ${path.basename(source.fsPath)}`;
	const nonce = [...crypto.getRandomValues(new Uint8Array(16))].map(b => b.toString(16).padStart(2, '0')).join('');
	panel.webview.html = runOptionsHtml({ program, file: path.basename(source.fsPath), listed: listed(text, program), kind, overrides, makefile: makefile ? await makefileValues(makefile, program, kind) : {}, cspSource: panel.webview.cspSource, nonce });
	panel.reveal(undefined, true);
}

export function registerRunTrace(): vscode.Disposable {
	return vscode.Disposable.from(
		// (a menu passes more than the file; only the file is for the command)
		vscode.commands.registerCommand('skevu.runTrace', (uri?: vscode.Uri) => runTrace(uri)),
		vscode.commands.registerCommand('skevu.runOptions', (uri?: vscode.Uri) => runOptions(uri)),
	);
}
