import * as esbuild from 'esbuild'
import { readFileSync } from 'node:fs'

//const esbuild = require('esbuild');

async function build() {
	await Promise.all([

		// esm to handle top-level awaits

		esbuild.build({
			entryPoints: ['script/workers/dagre-worker.js'],
			bundle: true,
			format: 'iife',
			outfile: 'dist/dagre-worker.js',
			banner: {
				js: `${readFileSync('script/static/dagre/dagre.min.js', 'utf8')}`
			}
		}),

		esbuild.build({
			entryPoints: ['script/backend/visualizer.js'],
			bundle: true,
			format: 'iife',
			outfile: 'dist/visualizer.js'
		}),

		esbuild.build({
			entryPoints: ['skevu.js'],
			bundle: true,
			format: 'iife',
			outfile: 'dist/skevu.js'
		})
	])
}

build().catch(() => process.exit(1));