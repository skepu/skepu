// The form of the options of a run (a panel of the editor): all of them at once, each with what the Makefile has until it is set.
// Pure text, so that it can be tested without an editor (tests/unit/test_run_plan.js).
import { BACKENDS, MakefileValues, RunOverrides } from './runPlan.js';

export interface RunForm {
	program: string;
	/** The source file, as the form names it. */
	file: string;
	/** The program is in PROGS of the Makefile. */
	listed: boolean;
	kind: 'examples' | 'package';
	overrides: RunOverrides;
	/** What the Makefile runs the program with (examples). */
	makefile: MakefileValues;
	cspSource: string;
	nonce: string;
}

const escape = (text: string) => text.replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);

export function runOptionsHtml(form: RunForm): string {
	const { overrides, makefile } = form;
	const backends = [['', `Makefile${makefile.backend ? ` (${makefile.backend})` : ''}`], ...BACKENDS.map(b => [b, b])]
		.map(([value, label]) => `<option value="${value}"${(overrides.backend ?? '') === value ? ' selected' : ''}>${escape(label)}</option>`).join('');
	const examples = form.kind === 'examples';
	const placeholder = makefile.arguments ?? '';
	return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'nonce-${form.nonce}'; script-src 'nonce-${form.nonce}';">
<style nonce="${form.nonce}">
body { font: var(--vscode-font-size) var(--vscode-font-family); color: var(--vscode-foreground); padding: 12px 20px; max-width: 560px; }
h1 { font-size: 1.3em; font-weight: 600; margin: 0 0 14px; }
label.field { display: block; margin: 12px 0 0; }
label.field span { display: block; margin-bottom: 3px; }
select, input[type=text] { width: 100%; box-sizing: border-box; padding: 4px 6px; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border, transparent); font: inherit; }
label.check { display: flex; align-items: center; gap: 6px; margin: 10px 0 0; }
small { display: block; color: var(--vscode-descriptionForeground); margin-top: 2px; }
.buttons { margin-top: 18px; display: flex; gap: 8px; }
button { padding: 5px 14px; border: 0; color: var(--vscode-button-foreground); background: var(--vscode-button-background); font: inherit; cursor: pointer; }
button:hover { background: var(--vscode-button-hoverBackground); }
button.secondary { color: var(--vscode-button-secondaryForeground); background: var(--vscode-button-secondaryBackground); }
</style></head><body>
<h1>Options for ${escape(form.file)}</h1>
${examples ? `<label class="field"><span>Default backend</span><select id="backend"${overrides.arguments ? ' disabled' : ''}>${backends}</select><small>Part of the arguments of the Makefile; not used when Arguments is set.</small></label>` : '<select id="backend" hidden></select>'}
<label class="field"><span>Arguments</span><input type="text" id="arguments" value="${escape(overrides.arguments ?? '')}" placeholder="${escape(placeholder)}" spellcheck="false"><small>${examples && !form.listed ? 'Required: not in PROGS of the Makefile.' : makefile.arguments ? 'Replace the arguments of the Makefile.' : 'Arguments of the program.'}</small></label>
<label class="check"><input type="checkbox" id="trace"${overrides.trace === false ? '' : ' checked'}>Enable tracing <small style="display:inline">TRACE=1</small></label>
<label class="check"><input type="checkbox" id="snapshots"${overrides.snapshots && overrides.trace !== false ? ' checked' : ''}${overrides.trace === false ? ' disabled' : ''}>Enable snapshots <small style="display:inline">TRACE_SNAPSHOTS=1</small></label>
<label class="check"><input type="checkbox" id="rebuild"${overrides.rebuild ? ' checked' : ''}>Force rebuild <small style="display:inline">make -B</small></label>
<div class="buttons"><button id="build">Build</button><button id="run">Build and Run</button><button id="reset" class="secondary">Reset</button></div>
<script nonce="${form.nonce}">
const vscode = acquireVsCodeApi(), $ = id => document.getElementById(id);
const read = () => ({ backend: $('backend').value || undefined, arguments: $('arguments').value || undefined, trace: $('trace').checked ? undefined : false, snapshots: $('snapshots').checked || undefined, rebuild: $('rebuild').checked || undefined });
const sync = () => { $('backend').disabled = !!$('arguments').value; if (!$('trace').checked) $('snapshots').checked = false; $('snapshots').disabled = !$('trace').checked; };
for (const id of ['backend', 'arguments', 'trace', 'snapshots', 'rebuild']) $(id).addEventListener('input', () => { sync(); vscode.postMessage({ type: 'set', overrides: read() }); });
$('build').addEventListener('click', () => vscode.postMessage({ type: 'build', overrides: read() }));
$('run').addEventListener('click', () => vscode.postMessage({ type: 'run', overrides: read() }));
$('reset').addEventListener('click', () => { $('backend').value = ''; $('arguments').value = ''; $('trace').checked = true; $('snapshots').checked = false; $('rebuild').checked = false; sync(); vscode.postMessage({ type: 'set', overrides: {} }); });
</script></body></html>
`;
}
