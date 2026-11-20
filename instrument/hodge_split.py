"""Split edge data on a graph into its integrable (gradient) and cyclic parts.

For a graph with oriented edges (i, j), i < j, and edge values y, the integrable part is the
least-squares projection of y onto differences of node values, d0 z. The cyclic part is the
remainder y - d0 z; it is orthogonal to the integrable part, and it is the only part any
closure (loop-sum) statistic can see. Independent implementation; it is checked in
verification/ to agree with the solver used in the study.
"""
from itertools import combinations
import numpy as np


def complete_graph_edges(n):
    return list(combinations(range(n), 2))


def incidence(n, edges):
    d0 = np.zeros((len(edges), n))
    for k, (i, j) in enumerate(edges):
        d0[k, i] = -1.0; d0[k, j] = 1.0
    return d0


def split(y, n, edges=None):
    """y: (n_edges,) or (n_edges, d). Returns (integrable, cyclic), same shape as y."""
    edges = complete_graph_edges(n) if edges is None else edges
    d0 = incidence(n, edges)
    z, *_ = np.linalg.lstsq(d0, y, rcond=None)
    g = d0 @ z
    return g, y - g


def energies(err, n, edges=None):
    """Integrable and cyclic energy of a 3-D error field (n_edges, 3)."""
    g, c = split(np.asarray(err, float), n, edges)
    return float((g ** 2).sum()), float((c ** 2).sum())


def chance_invisible_share(n):
    """Expected integrable share for independent isotropic edge errors on K_n: (n-1)/C(n,2) = 2/n."""
    return 2.0 / n
