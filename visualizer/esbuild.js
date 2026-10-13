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
				js: `${readFileSync('static/dagre/dagre.min.js', 'utf8')}`
			}
		}),

		esbuild.build({
			entryPoints: ['script/backend/visualizer.js'],
			bundle: true,
			format: 'iife',
			outfile: 'dist/visualizer.js',
			// (the worker is given the URLs it needs: the ones from import.meta are the defaults of the page)
			logOverride: { 'empty-import-meta': 'silent' }
		}),

		esbuild.build({
			entryPoints: ['skevu.js'],
			bundle: true,
			format: 'iife',
			outfile: 'dist/skevu.js',
			// (the URLs that use import.meta are the defaults of the page: the extension gives its own, in window.skevu)
			logOverride: { 'empty-import-meta': 'silent' }
		})
	])
}

build().catch(() => process.exit(1));