import { globalUtil } from './util.js';

globalUtil.fetchWorker = async function (url) {
    return new Worker(url);
};