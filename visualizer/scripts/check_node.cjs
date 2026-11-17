// Checks that this Node is one that @vscode/vsce runs on (the engines.node of the installed vsce, so that it follows the version of vsce).
// usage: node scripts/check_node.cjs     exits 1 with a message if it is not; used by `make vsix`.
'use strict';
const path = require('path');
const vscePkg = path.join(__dirname, '..', 'node_modules', '@vscode', 'vsce', 'package.json');
let engines;
try { engines = require(vscePkg).engines?.node; } catch {
    console.error('@vscode/vsce is not installed in node_modules: run npm install in ' + path.join(__dirname, '..') + ' (it is a dev dependency; npm install --omit=dev leaves it out)');
    process.exit(1);
}
if (!engines) process.exit(0);
const semver = require(require.resolve('semver', { paths: [path.dirname(vscePkg)] }));
if (!semver.satisfies(process.version, engines)) {
    console.error(`the extension is packaged by @vscode/vsce, which needs Node ${engines}; this is Node ${process.version}.`);
    console.error('Use a newer Node (nodejs.org, nvm, brew install node), or build the extension on another computer and pass the file: packaging/make-package.sh --vsix FILE');
    process.exit(1);
}
