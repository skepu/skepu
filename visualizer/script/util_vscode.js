import { globalUtil } from './util.js';

globalUtil.fetchWorker = async function (url) {
    const response = await fetch(url);
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    
    return new Worker(blobUrl);
};