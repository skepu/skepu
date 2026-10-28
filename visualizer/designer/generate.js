#!/usr/bin/env node
// generate.js — prints the program of a design:
//   node designer/generate.js [design.json | --sample] [cpp|python] [--lambdas] [--anonymous] [--first-use] [--skeletons-first-use] [--whitespace=readable|balanced|compact]
// Problems go to stderr (exit 1 on errors).
import { readFileSync } from 'node:fs';
import { lower, sampleDesign } from './model.js';
import { EMITTERS } from './codegen.js';

const args = process.argv.slice(2).filter(a => !a.startsWith('--') || a === '--sample');
const [file, language = 'cpp'] = args;
const options = { userFunctions: process.argv.includes('--lambdas') ? 'lambdas' : 'functions', skeletonInstancing: process.argv.includes('--anonymous') ? 'anonymous' : 'named', containerPlacement: process.argv.includes('--first-use') ? 'first-use' : 'grouped',
                  skeletonPlacement: process.argv.includes('--skeletons-first-use') ? 'first-use' : 'grouped',
                  whitespace: (process.argv.find(a => a.startsWith('--whitespace=')) ?? '').slice(13) || 'balanced' };
const design = !file || file === '--sample' ? sampleDesign() : JSON.parse(readFileSync(file, 'utf8'));
const result = lower(design);
for (const e of result.errors) console.error(`error: ${e.msg}`);
for (const w of result.warnings) console.error(`warning: ${w.msg}`);
if (!result.ok) process.exit(1);
process.stdout.write(EMITTERS[language === 'py' ? 'python' : language](result.program, options).text);
