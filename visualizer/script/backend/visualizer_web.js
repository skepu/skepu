// The entry of the worker in the web page: the visualizer worker, with the library headers where the page has them.
import { setLibRootUrl, setLibDNNUrl } from './visualizer.js'

const _BASE_URL = new URL('..', import.meta.url).href;

setLibRootUrl(_BASE_URL + '../static/skepu-lib/');
setLibDNNUrl(_BASE_URL + '../static/skepu-lib/dnn/');
