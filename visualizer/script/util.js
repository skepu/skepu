// util.js — what differs between the places that the viewer runs in (a browser page, the webview of an editor). The defaults are those of a
// page; util_vscode.js sets the others.
export class Util {
    /** A worker of the script at the URL (the layout of the graph, off the main thread). */
    async fetchWorker(url) {
        return new Worker(url);
    }

    /** The options for ELK (a layout engine, which makes its worker from the URL of its script if it is given one). */
    elkOptions(workerUrl) {
        return { workerUrl };
    }
}

export const globalUtil = new Util();
