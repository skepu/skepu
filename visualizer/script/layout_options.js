// layout_options.js — the spacing of the dagre layouts, shared by SkeVU and the designer.

export const LAYOUT_NODE_SEP    = 15;
export const LAYOUT_RANK_SEP    = 20;

/** Dagre adds edges for the nesting of regions, which stretch the ranks: with regions in the graph the distances are scaled down to make up for it. */
export const COMPOUND_SEP_SCALE = 0.15;
