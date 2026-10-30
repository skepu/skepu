import * as vscode from 'vscode';
import * as path from 'node:path';
import * as fs from 'node:fs';
import { execFile } from 'node:child_process';
import * as os from 'node:os';
import { MakefileValues, planRun, PRINT_RULES, printTargets, RunOverrides, RunPlan, sourceProgram, valuesOf } from './runPlan.js';
import { runOptionsHtml } from './runOptionsHtml.js';

/** User-set options for the runs of this window (not saved); unset fields use the Makefile's. */
let overrides: RunOverrides = {};
/** Tracing and snapshot flags of each program's last build (by path). They are build flags, so a change forces a rebuild. */
const built = new Map<string, { trace: boolean; snapshots: boolean }>();

const shellQuote = (word: string) => /^[\w@%+=:,./-]+$/.test(word) ? word : `'${word.replace(/'/g, `'\\''`)}'`;

const makefileOf = (dir: string) => ['Makefile', 'makefile', 'GNUmakefile'].map(name => path.join(dir, name)).find(fs.existsSync);

function sourceOf(uri?: vscode.Uri): vscode.Uri | undefined {
	return uri ?? vscode.window.activeTextEditor?.document.uri;
}

async function runTrace(uri?: vscode.Uri, buildOnly = false): Promise<void> {
	const source = sourceOf(uri);
	if (!source || source.scheme !== 'file') return void vscode.window.showErrorMessage('SkeVU: no source file is open.');
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
		makefile: fs.readFileSync(makefile, 'utf8'), source: source.fsPath, values: await makefileValues(makefile, sourceProgram(source.fsPath)),
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
		new vscode.ShellExecution(line, { cwd: plan.cwd }), '$gcc');
	task.presentationOptions = { reveal: vscode.TaskRevealKind.Always, clear: true, panel: vscode.TaskPanelKind.Dedicated };
	const execution = await vscode.tasks.executeTask(task);
	const code = await new Promise<number | undefined>(resolve => {
		const done = vscode.tasks.onDidEndTaskProcess(e => {
			if (e.execution === execution) { done.dispose(); resolve(e.exitCode); }
		});
	});
	if (code !== 0) return void vscode.window.showErrorMessage(`SkeVU: ${plan.program} failed${code === undefined ? '' : ` (exit code ${code})`}.`);
	if (!plan.trace) return;
	const trace = vscode.Uri.file(path.join(plan.cwd, plan.trace));
	if (!fs.existsSync(trace.fsPath)) return void vscode.window.showErrorMessage(`SkeVU: no trace at ${plan.trace}.`);
	await vscode.commands.executeCommand('vscode.openWith', trace, 'skepu.skevu', vscode.ViewColumn.Beside);
}

/** The Makefile's values for a program as make resolves them; missing if make fails. */
async function makefileValues(makefile: string, program: string): Promise<MakefileValues> {
	const rules = path.join(os.tmpdir(), 'skevu_print.mk');
	fs.writeFileSync(rules, PRINT_RULES);
	const make = vscode.workspace.getConfiguration('skevu.run').get<string>('make', 'make');
	const print = (target: string) => new Promise<string | undefined>(resolve =>
		execFile(make, ['-s', '-f', makefile, '-f', rules, target], { cwd: path.dirname(makefile), timeout: 10000 }, (error, out) => resolve(error ? undefined : out.trim())));
	return valuesOf(await Promise.all(printTargets(program).map(print)));
}

let panel: vscode.WebviewPanel | undefined;
/** The source the form is for (one panel, for the source it was last opened from). */
let formSource: vscode.Uri | undefined;

async function runOptions(uri?: vscode.Uri): Promise<void> {
	const source = sourceOf(uri);
	if (!source || source.scheme !== 'file') return void vscode.window.showErrorMessage('SkeVU: no source file is open.');
	const program = sourceProgram(source.fsPath);
	const makefile = makefileOf(path.dirname(source.fsPath));
	const text = makefile ? fs.readFileSync(makefile, 'utf8') : '';
	const values = makefile ? await makefileValues(makefile, program) : {};
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
	panel.webview.html = runOptionsHtml({ program, file: path.basename(source.fsPath), overrides, makefile: values, cspSource: panel.webview.cspSource, nonce });
	panel.reveal(undefined, true);
}

export function registerRunTrace(): vscode.Disposable {
	return vscode.Disposable.from(
		// menus pass extra arguments; only the file is used
		vscode.commands.registerCommand('skevu.runTrace', (uri?: vscode.Uri) => runTrace(uri)),
		vscode.commands.registerCommand('skevu.runOptions', (uri?: vscode.Uri) => runOptions(uri)),
	);
}
