// edge.js — Data-flow edge types.  Browser/Worker compatible ES module.
// Ported from backend/edge.py.
'use strict';

// ── Base ──────────────────────────────────────────────────────────────────────

export class Edge {
    constructor(graph, sourceId, target) {
        this.id             = crypto.randomUUID();
        this.label          = '';
        this.target         = target;
        this.source         = (typeof sourceId === 'number')
                                ? graph.producerForID(sourceId, target.backend)
                                : sourceId;   // already a Node
        this.isCriticalPath = false;
        this.iterationCount = 1;

        if (this.source && this.target) {
            graph.addDependenceEdge(this);
        }
    }

    toCytoscape() {
        const srcBackend = this.source.backend ?? null;
        const tgtBackend = this.target.backend ?? null;
        return {
            id:               this.id,
            label:            this.label,
            source:           this.source.id,
            target:           this.target.id,
            is_critical_path: this.isCriticalPath,
            iteration_count:  this.iterationCount,
            order_span:       Math.max(0, this.target.totalOrder - this.source.totalOrder),
            cross_backend:    !!(srcBackend && tgtBackend && srcBackend !== tgtBackend),
        };
    }
}

export class TreeEdge {
  constructor(graph, sourceId, target) {
      this.id             = crypto.randomUUID();
      this.label          = '';
      this.target         = target;
      this.source         = (typeof sourceId === 'number')
                              ? graph.producerForID(sourceId, target.backend)
                              : sourceId;   // already a Node
      this.isCriticalPath = false;
      this.iterationCount = 1;

      if (this.source && this.target) {
          graph.addTreeEdge(this);
      }
  }

  toCytoscape() {
      const srcBackend = this.source.backend ?? null;
      const tgtBackend = this.target.backend ?? null;
      return {
          type:             'tree',
          id:               this.id,
          label:            this.label,
          source:           this.source.id,
          target:           this.target.id,
          is_critical_path: this.isCriticalPath,
          iteration_count:  this.iterationCount,
          order_span:       Math.max(0, this.target.totalOrder - this.source.totalOrder),
          cross_backend:    !!(srcBackend && tgtBackend && srcBackend !== tgtBackend),
      };
  }
}

// ── Concrete edge types ───────────────────────────────────────────────────────

export class ElwiseEdge extends Edge {
    constructor(graph, label, sourceId, target) {
        super(graph, sourceId, target);
        this.label = label;
    }

    toCytoscape() {
        return { ...super.toCytoscape(), access_mode: 'elwise', type: 'forward-dep' };
    }
}

export class ProxyEdge extends Edge {
    constructor(graph, label, sourceId, target) {
        super(graph, sourceId, target);
        this.label = label;
    }

    toCytoscape() {
        return { ...super.toCytoscape(), access_mode: 'proxy', type: 'forward-dep' };
    }
}

export class ScalarEdge extends Edge {
    constructor(graph, label, sourceId, target) {
        super(graph, sourceId, target);
        this.label = label;
    }

    toCytoscape() {
        return {
            ...super.toCytoscape(),
            access_mode:   'scalar',
            type:          'forward-dep',
            cross_backend: false,   // scalar edges carry a single value; no backend cost
        };
    }
}

export class AntiDepEdge extends Edge {
    constructor(graph, label, sourceId, target) {
        super(graph, sourceId, target);
        this.label = label;
    }

    toCytoscape() {
        return { ...super.toCytoscape(), access_mode: 'proxy', type: 'anti-dep' };
    }
}

export class PrngEdge extends Edge {
    constructor(graph, label, sourceId, target) {
        super(graph, sourceId, target);
        this.label = label;
    }

    toCytoscape() {
        return { ...super.toCytoscape(), access_mode: 'proxy', type: 'prng-dep' };
    }
}

// ── Decorative (not part of the dependency graph) ─────────────────────────────

/**
 * Decorative-only edge connecting UpdateNodes that belong to the same virtual
 * container group (e.g. a double-buffered pair).
 *
 * Does NOT touch node incoming/outgoing lists, so it has no effect on layout
 * or critical-path analysis.  Stored in graph.decorativeEdges.
 */
export class VirtualAliasEdge {
    constructor(graph, sourceNode, targetNode, virtualLabel) {
        this.id           = crypto.randomUUID();
        this.label        = virtualLabel;
        this.source       = sourceNode;
        this.target       = targetNode;
        this.virtualLabel = virtualLabel;
        this.isCriticalPath = false;
        this.iterationCount = 1;
        graph.addDecorativeEdge(this);
    }

    toCytoscape() {
        return {
            id:               this.id,
            label:            this.virtualLabel,
            source:           this.source.id,
            target:           this.target.id,
            type:             'virtual-alias',
            is_critical_path: false,
            iteration_count:  1,
            virtual_label:    this.virtualLabel,
        };
    }
}
