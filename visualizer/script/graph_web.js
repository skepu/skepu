import { State } from './state.js';
import { setInfoViewOnHighlightNode } from './graph.js';

setInfoViewOnHighlightNode(function(data) {
    if (State.sourceView) State.sourceView.highlightNode(data, undefined, State._selectedRank);
});
