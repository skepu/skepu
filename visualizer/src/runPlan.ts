// Plans the build and run of a SkePU program through the Makefile next to its source, which includes make/skepu.mk (directly in a package, through
// examples/Makefile in the examples). The rules are the same in both: `make bin/<program>` builds; `make run/<program>` builds and runs from RUN_CWD with
// `lookup.<program>` as arguments; a program built with TRACE=1 writes RUN_CWD/trace/trace_<program>.json.
// The Makefile's values for a program (arguments, run directory) come from make (runTrace.ts) as MakefileValues.
// Pure text to text, so that it can be tested without an editor (tests/unit/test_run_plan.js).

export interface RunPlan {
	/** The Makefile's directory. */
	cwd: string;
	/** The source name without its extension. */
	program: string;
	/** Run in order, each if the previous one succeeded. */
	commands: string[][];
	/** Trace written by the run, relative to cwd; none without tracing or for a build only. */
	trace?: string;
}

/** User-set options; unset fields use the Makefile's. */
export interface RunOverrides {
	/** RUN_BACKEND, used by the Makefile's arguments when they contain it. */
	backend?: string;
	/** Program arguments; they replace the Makefile's, and with them the backend. */
	arguments?: string;
	/** TRACE; the Makefile has a default. Off: no trace and no snapshots. */
	trace?: boolean;
	/** Requires tracing. */
	snapshots?: boolean;
	/** make -B. */
	rebuild?: boolean;
}

/** The Makefile's values as make resolves them; missing if make could not say. */
export interface MakefileValues {
	/** RUN_BACKEND. */
	backend?: string;
	/** lookup.<program>. */
	arguments?: string;
	/** RUN_CWD and TRACE_DIR. */
	runCwd?: string;
	traceDir?: string;
}

export interface RunOptions {
	makefile: string;
	source: string;
	values: MakefileValues;
	overrides: RunOverrides;
	make: string;
	buildOnly?: boolean;
}

export const BACKENDS = ['CPU', 'OpenMP', 'OpenCL', 'CUDA'];

/** Whether the backend is among the Makefile's arguments for the program (the lookup of some programs has none). */
export const usesBackend = (values: MakefileValues) => !!values.backend && !!values.arguments?.split(/\s+/).includes(values.backend);

export const isSkepuMakefile = (text: string) => /^\s*-?include\s+\S*(skepu\.mk|\.\.\/Makefile)\s*$/m.test(text);

export function sourceProgram(source: string): string {
	const base = source.replace(/^.*[\\/]/, '');
	return base.replace(/\.[^.]*$/, '');
}

/** Run directory relative to the Makefile, when make cannot say: the directory of skepu.mk (package) or the current directory (examples). */
const runCwdOf = (makefile: string) => (makefile.match(/^\s*-?include\s+([\w./-]*)skepu\.mk\b/m)?.[1] ?? '').replace(/\/$/, '') || '.';

const joinPath = (...parts: string[]) => parts.filter(p => p !== '.' && p !== '').join('/');

export function planRun(options: RunOptions, cwd: string): RunPlan | { error: string } {
	if (!isSkepuMakefile(options.makefile)) return { error: 'The Makefile does not include skepu.mk or ../Makefile.' };
	const program = sourceProgram(options.source);
	if (/\s/.test(program)) return { error: `make cannot build "${program}": the name has a space. Rename the file.` };
	const { overrides, values } = options;
	const traced = overrides.trace !== false;
	const rebuild = overrides.rebuild ? ['-B'] : [];
	const build = [`TRACE=${traced ? 1 : 0}`, ...(traced && overrides.snapshots !== undefined ? [`TRACE_SNAPSHOTS=${overrides.snapshots ? 1 : 0}`] : [])];
	// command-line variables replace the Makefile's, including the program's arguments
	const given = [...(overrides.backend ? [`RUN_BACKEND=${overrides.backend}`] : []), ...(overrides.arguments !== undefined ? [`lookup.${program}=${overrides.arguments}`] : [])];
	return {
		cwd, program,
		commands: [options.buildOnly ? [options.make, ...rebuild, `bin/${program}`, ...build] : [options.make, ...rebuild, `run/${program}`, ...build, ...given]],
		trace: traced && !options.buildOnly ? joinPath(values.runCwd ?? runCwdOf(options.makefile), values.traceDir ?? 'trace', `trace_${program}.json`) : undefined,
	};
}

/** Makefile rule that prints a variable: `make print-NAME`. */
export const PRINT_RULES = 'print-%:\n\t@echo $($*)\n';
export const printTargets = (program: string) => ['print-RUN_BACKEND', `print-lookup.${program}`, 'print-RUN_CWD', 'print-TRACE_DIR'];

/** The values printed for printTargets, in the same order; missing if a line is empty or make failed. */
export function valuesOf(printed: (string | undefined)[]): MakefileValues {
	const [backend, args, runCwd, traceDir] = printed.map(v => v?.trim() || undefined);
	return { backend, arguments: args, runCwd, traceDir };
}
