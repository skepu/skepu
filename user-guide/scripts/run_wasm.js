// Runs a built listing as the page does and prints its output: node run_wasm.js listings/wasm/<hash>.js [args]
const path = require("path");

const factory = require(path.resolve(process.argv[2]));
const lines = [];
factory({ print: l => lines.push(l), printErr: l => process.stderr.write(l + "\n") })
  .then(mod => {
    try { mod.callMain(process.argv.slice(3)); }
    catch (e) { if (!String(e).includes("ExitStatus")) throw e; }   // main() returning is an exit
    process.stdout.write(lines.join("\n") + "\n");
  })
  .catch(e => { process.stderr.write(String(e) + "\n"); process.exit(1); });
