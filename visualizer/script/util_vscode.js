import { globalUtil } from './util.js';

// A webview cannot make a worker of a script of the extension (another origin than the page): it makes it of a copy of the script.
globalUtil.fetchWorker = async function (url) {
    const response = await fetch(url);
    const blob = await response.blob();
    return new Worker(URL.createObjectURL(blob));
};

// ELK has its layout in the script that the page has loaded, and runs it in the page when it is not given a worker
globalUtil.elkOptions = function () {
    return {};
};
