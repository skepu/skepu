// How a SkePU program is built and run for a trace: the build rules are those of the Makefile next to its source, which is one of
//  - the Makefile of the examples (`make run/<program>` builds, runs with the arguments of the Makefile and writes trace/trace_<program>.json), or
//  - the Makefile of a package (`include skepu.mk`: `make run/<program>` builds, runs from the directory of skepu.mk with `lookup.<program>` as the arguments,
//    and writes trace_<program>.json there, with the program built with TRACE=1).
// Pure text to text, so that it can be tested without an editor (tests/unit/test_run_plan.js).

export type MakefileKind = 'examples' | 'package';

export interface RunPlan {
	kind: MakefileKind;
	/** Where the commands run: the directory of the Makefile. */
	cwd: string;
	/** The program, as the Makefile names it (the source without its extension). */
	program: string;
	/** The commands, in order; each is run when the one before succeeded. A command is a program and its arguments. */
	commands: string[][];
	/** The trace the run writes, relative to cwd; none without tracing. */
	trace?: string;
	/** Environment of the commands. */
	env: { [key: string]: string };
}

/** What the user has set for the runs; whatever is not set is the Makefile's. */
export interface RunOverrides {
	/** The backend of the examples (RUN_BACKEND); the arguments of the Makefile use it. */
	backend?: string;
	/** The arguments of the program; in the examples they replace those of the Makefile, and so the backend. */
	arguments?: string;
	/** Tracing (TRACE; on unless the Makefile says otherwise). Off, there is no trace, and no snapshots. */
	trace?: boolean;
	/** Snapshots of containers (TRACE_SNAPSHOTS=1); they need tracing. */
	snapshots?: boolean;
	/** Builds everything again (make -B). */
	rebuild?: boolean;
}

export interface RunOptions {
	/** The Makefile with its text, and the source it is for. */
	makefile: string;
	source: string;
	overrides: RunOverrides;
	make: string;
	/** Builds the program, and does not run it. */
	buildOnly?: boolean;
}

export const BACKENDS = ['CPU', 'OpenMP', 'OpenCL', 'CUDA'];

/** The kind of a Makefile, if it is one that builds SkePU programs. */
export function makefileKind(text: string): MakefileKind | undefined {
	if (/^\s*include\s+\S*skepu\.mk\b/m.test(text)) return 'package';
	if (/^\s*\$\(RUN_DIR\)\/%/m.test(text) && /^\s*PROGS\s*\+?=/m.test(text)) return 'examples';
	if (/^\s*include\s+\S*\.\.\/Makefile\s*$/m.test(text)) return 'examples';      // a program of its own directory, with the rules of the examples
	return undefined;
}

/** The programs that a Makefile lists (PROGS = a b, PROGS += c, with continued lines and comments). */
export function programsOf(text: string): string[] {
	const programs: string[] = [];
	const joined = text.replace(/\\\r?\n/g, ' ');
	for (const match of joined.matchAll(/^\s*PROGS\s*(\+?=|:=)([^\n]*)/gm)) {
		const value = match[2].replace(/#.*/, '');
		if (value.includes('$(')) continue;      // computed (a filter of the list): not a list of programs
		const words = value.split(/\s+/).filter(w => /^[\w.-]+$/.test(w));
		if (match[1] !== '+=') programs.length = 0;
		programs.push(...words);
	}
	return programs;
}

/** Splits arguments as a shell would for the simple cases: spaces, and single or double quotes. */
export function splitArguments(text: string): string[] {
	return [...text.matchAll(/"([^"]*)"|'([^']*)'|(\S+)/g)].map(m => m[1] ?? m[2] ?? m[3]);
}

export function sourceProgram(source: string): string {
	const base = source.replace(/^.*[\\/]/, '');
	return base.replace(/\.[^.]*$/, '');
}

/** Whether the Makefile lists a program in PROGS. */
export const listed = (makefile: string, program: string) => programsOf(makefile).includes(program);

/** The plan to build and run `options.source`, or the reason that there is none. */
export function planRun(options: RunOptions, cwd: string): RunPlan | { error: string } {
	const kind = makefileKind(options.makefile);
	if (!kind) return { error: 'The Makefile next to the source is not one of the examples or a package (it does not include skepu.mk).' };
	const program = sourceProgram(options.source);
	if (/\s/.test(program)) return { error: `make cannot build "${program}": the name has a space. Rename the file.` };
	const { overrides } = options;
	// a source that is not in PROGS is built by the same rules; the examples do not know its arguments, so a run has to be given them
	if (kind === 'examples' && !options.buildOnly && !listed(options.makefile, program) && !overrides.arguments) return { error: `${program} is not in PROGS of the Makefile: give its arguments to run it.` };
	const rebuild = overrides.rebuild ? ['-B'] : [];
	const traced = overrides.trace !== false;
	const snapshots = traced && overrides.snapshots !== undefined ? [`TRACE_SNAPSHOTS=${overrides.snapshots ? 1 : 0}`] : [];
	if (kind === 'examples') {
		// (variables on the command line replace those of the Makefile, the arguments of the program among them)
		const given = [...(overrides.backend ? [`RUN_BACKEND=${overrides.backend}`] : []), ...(overrides.arguments !== undefined ? [`lookup.${program}=${overrides.arguments}`] : [])];
		const flags = [...(traced ? [] : ['TRACE=0']), ...snapshots];
		return {
			kind, cwd, program,
			commands: [options.buildOnly ? [options.make, ...rebuild, `parallel/${program}`, ...flags] : [options.make, ...rebuild, `run/${program}`, ...given, ...flags]],
			trace: traced && !options.buildOnly ? `trace/trace_${program}.json` : undefined,
			env: {},
		};
	}
	// (skepu.mk is where `include` says; the programs are run from its directory, where their trace is)
	const root = (options.makefile.match(/^\s*include\s+(\S*)skepu\.mk\b/m)?.[1] ?? '').replace(/\/$/, '');
	const trace = `${root ? root + '/' : ''}trace_${program}.json`;
	const lookup = overrides.arguments !== undefined ? [`lookup.${program}=${overrides.arguments}`] : [];
	const build = [`TRACE=${traced ? 1 : 0}`, ...(traced && overrides.snapshots ? ['TRACE_SNAPSHOTS=1'] : [])];
	return {
		kind, cwd, program,
		commands: [options.buildOnly ? [options.make, ...rebuild, `bin/${program}`, ...build] : [options.make, ...rebuild, `run/${program}`, ...build, ...lookup]],
		trace: traced && !options.buildOnly ? trace : undefined,
		env: {},
	};
}

/** The make program's own words for what to ask of a Makefile: a rule that prints a variable (`make print-NAME`). */
export const PRINT_RULES = 'print-%:\n\t@echo $($*)\n';

/** What the Makefile of the examples runs a program with, as make has it: the backend and the arguments, the variables replaced by their values. */
export interface MakefileValues { backend?: string; arguments?: string }
export const printTargets = (program: string, kind: MakefileKind = 'examples') => kind === 'examples' ? ['print-RUN_BACKEND', `print-lookup.${program}`] : [`print-lookup.${program}`];
