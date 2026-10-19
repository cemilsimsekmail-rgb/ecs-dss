#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examination Center Selection - Decision Support System (ECS-DSS) v2.0

Companion software for:
    C. Simsek, "A Decision Support System for Risk-Constrained Examination
    Center Selection Under a Coverage Requirement," IEEE Access, 2026.

Solves the proposed integer program on the institutional data, runs every
analysis reported in the article and regenerates all tables and figures.

Model (equation numbers follow the article):

    max  Z = sum_i P_i y_ii                                          (1)
    s.t. sum_{j in N_i(M)} y_ij = 1            for all i             (2)
         y_ij <= x_j                           for all i, j          (3)
         y_ij = 0                              if d_ij > M           (4)
         x_j  = 1                              for all j in I_H      (5)
         sum_j x_j = K                                               (6)
         x, y binary                                                 (7)

    q(M, I_H) = min cover of I at radius M with I_H open             (9)
    K = max{N, V, q(M, I_H)}                                        (10)

Proposition 1: q >= V always, hence K = max{N, q}; V never binds alone.

Lexicographic travel stage (15): with Z* the optimum of (1)-(7),
    min sum_ij P_i d_ij y_ij  s.t. (2)-(7) and sum_i P_i y_ii = Z*.

Extensions added in v2.1 (Section III-G of the article):
    capacity (16):   sum_i P_i y_ij <= C_j x_j          for all j
    elastic (17)-(19): penalty variables u_i >= sum_j (d_ij - M)^+ y_ij
                     admit assignments beyond M at a lexicographically
                     dominant penalty, giving a best-effort plan when no
                     plan with K centers satisfies the travel limit.

Requirements:  pip install -r requirements.txt   (PuLP 2.x, NumPy, Matplotlib)
               tkinter (python3-tk on Debian/Ubuntu) for the GUI only
Data:          data.json next to this file
Usage:         python ecs_dss.py

Author:  Cemil Simsek
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import numpy as np

try:
    import pulp
except ImportError:
    sys.exit("PuLP is required:  pip install -r requirements.txt")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# The graphical interface is optional: --cli runs without Tk.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _TkBase = tk.Tk
except ImportError:          # headless installation
    tk = None
    _TkBase = object

APP_TITLE = "Examination Center Selection - Decision Support System v2.1"


def _binaries(name, keys):
    """Dictionary of binary variables. Uses LpVariable.dicts where PuLP
    provides it (2.x, the version used for the published results) and an
    equivalent explicit construction otherwise."""
    if hasattr(pulp.LpVariable, "dicts"):
        return pulp.LpVariable.dicts(name, keys, cat="Binary")
    out = {}
    for k in keys:
        tag = "_".join(map(str, k)) if isinstance(k, tuple) else str(k)
        out[k] = pulp.LpVariable(f"{name}_{tag}", cat="Binary")
    return out


def _solve(model):
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[model.status] == "Optimal"
HERE = os.path.dirname(os.path.abspath(__file__))

INK, ACC, ALERT, GOOD, MUTED = "#1f2933", "#1f4e79", "#c00000", "#2e7d32", "#7a8a99"
PAPER, PANEL = "#ffffff", "#f4f6f8"
PLOT_RC = {"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .8,
           "figure.dpi": 110, "savefig.dpi": 600}

# Provincial capitals, keyed by official vehicle registration code.
COORD = {
    1: (37.00, 35.32), 2: (37.76, 38.28), 3: (38.76, 30.54), 4: (39.72, 43.05),
    5: (40.65, 35.83), 6: (39.93, 32.86), 7: (36.90, 30.70), 8: (41.18, 41.82),
    9: (37.85, 27.84), 10: (39.65, 27.89), 11: (40.15, 29.98), 12: (38.88, 40.50),
    13: (38.40, 42.11), 14: (40.74, 31.61), 15: (37.72, 30.29), 16: (40.19, 29.06),
    17: (40.15, 26.41), 18: (40.60, 33.62), 19: (40.55, 34.95), 20: (37.78, 29.09),
    21: (37.91, 40.24), 22: (41.68, 26.56), 23: (38.68, 39.22), 24: (39.75, 39.49),
    25: (39.90, 41.27), 26: (39.78, 30.52), 27: (37.07, 37.38), 28: (40.91, 38.39),
    29: (40.46, 39.48), 30: (37.57, 43.74), 31: (36.20, 36.16), 32: (37.76, 30.55),
    33: (36.81, 34.63), 34: (41.01, 28.98), 35: (38.42, 27.14), 36: (40.60, 43.09),
    37: (41.38, 33.78), 38: (38.73, 35.49), 39: (41.74, 27.22), 40: (39.15, 34.16),
    41: (40.77, 29.94), 42: (37.87, 32.48), 43: (39.42, 29.98), 44: (38.35, 38.31),
    45: (38.61, 27.43), 46: (37.58, 36.93), 47: (37.31, 40.74), 48: (37.22, 28.36),
    49: (38.73, 41.49), 50: (38.62, 34.71), 51: (37.97, 34.68), 52: (40.98, 37.88),
    53: (41.02, 40.52), 54: (40.76, 30.38), 55: (41.29, 36.33), 56: (37.93, 41.94),
    57: (42.03, 35.15), 58: (39.75, 37.02), 59: (40.98, 27.51), 60: (40.31, 36.55),
    61: (41.00, 39.72), 62: (39.11, 39.55), 63: (37.16, 38.79), 64: (38.68, 29.41),
    65: (38.49, 43.38), 66: (39.82, 34.81), 67: (41.46, 31.79), 68: (38.37, 34.03),
    69: (40.26, 40.23), 70: (37.18, 33.22), 71: (39.85, 33.51), 72: (37.89, 41.13),
    73: (37.52, 42.46), 74: (41.64, 32.34), 75: (41.11, 42.70), 76: (39.92, 44.04),
    77: (40.66, 29.28), 78: (41.20, 32.63), 79: (36.72, 37.12), 80: (37.07, 36.25),
    81: (40.84, 31.16),
}

# Simplified national outline (lon, lat) for schematic mapping.
OUTLINE = [
    (26.05, 40.62), (26.35, 41.30), (26.62, 41.35), (27.20, 42.09),
    (28.02, 41.98), (28.98, 41.98), (29.90, 41.22), (31.40, 41.22),
    (32.30, 41.73), (33.35, 42.02), (34.80, 41.95), (35.15, 42.10),
    (36.00, 41.35), (36.66, 41.38), (38.40, 40.92), (39.43, 41.10),
    (40.25, 41.02), (41.55, 41.52), (41.82, 41.43), (42.85, 41.58),
    (43.45, 41.12), (43.66, 40.55), (43.72, 40.13), (44.30, 40.05),
    (44.80, 39.65), (44.42, 38.35), (44.30, 37.85), (44.79, 37.30),
    (44.22, 37.28), (42.78, 37.38), (42.35, 37.11), (41.51, 37.09),
    (40.71, 37.10), (39.36, 36.68), (38.20, 36.90), (37.10, 36.66),
    (36.66, 36.83), (36.15, 35.82), (35.55, 36.58), (34.90, 36.75),
    (34.03, 36.26), (33.70, 36.18), (32.80, 36.02), (32.00, 36.55),
    (30.55, 36.20), (30.10, 36.30), (29.10, 36.68), (28.25, 36.65),
    (27.42, 37.05), (27.28, 37.72), (26.78, 38.15), (26.90, 38.40),
    (26.35, 38.65), (26.70, 39.00), (26.19, 39.47), (26.70, 39.60),
    (26.20, 40.05), (26.75, 40.40), (26.05, 40.62),
]

# The 17 centers the agency currently operates.
CURRENT_PLAN = (1, 6, 7, 20, 21, 25, 30, 34, 35, 42, 55, 56, 58, 61, 62, 65, 73)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ProblemData:
    """Personnel counts and the inter-provincial highway distance matrix."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.P = {int(k): int(v) for k, v in raw["personnel"].items()}
        self.D = {}
        for key, val in raw["distance"].items():
            a, b = key.split("_")
            self.D[(int(a), int(b))] = int(val)
        self.I = sorted(self.P)
        self.n = len(self.I)
        self.total = sum(self.P.values())
        self._check()
        self._nb = {}

    def _check(self):
        missing = [(a, b) for a in self.I for b in self.I
                   if (a, b) not in self.D]
        if missing:
            raise ValueError(f"{len(missing)} distance pairs missing")
        if any(self.D[(a, a)] for a in self.I):
            raise ValueError("distance matrix has a non-zero diagonal")
        asym = [(a, b) for a in self.I for b in self.I
                if self.D[(a, b)] != self.D[(b, a)]]
        if asym:
            raise ValueError(f"{len(asym)} asymmetric distance pairs")

    def neighbours(self, i, M):
        """N_i(M)."""
        band = self._nb.setdefault(M, {})
        if i not in band:
            band[i] = [j for j in self.I if self.D[(i, j)] <= M]
        return band[i]

    def pair_count(self, M):
        return sum(len(self.neighbours(i, M)) for i in self.I)


# ---------------------------------------------------------------------------
# Risk scores
# ---------------------------------------------------------------------------
class RiskModel:
    """TOPSIS risk scores under the institutional policy weights.

    Criterion direction is stated with respect to security adequacy: a cost
    criterion is one whose increase lowers adequacy and therefore raises risk.
    """

    CRITERIA = [
        ("C1", "Annual operational incidents", "cost"),
        ("C2", "Seized weapons and IED interventions", "cost"),
        ("C3", "Fortified strategic station count", "cost"),
        ("C4", "Security personnel per capita", "benefit"),
        ("C5", "Other public-order incidents", "cost"),
    ]
    WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    LOW_CUT, HIGH_CUT = 0.33, 0.75

    # Calibration of the two landscapes reported in the article.
    SEED = 199851
    RHOS = [0.5350, 0.2592, 0.2698, 0.2583, 0.4498]
    ESCALATE = {30: 0.65, 62: 0.45}

    def __init__(self, data):
        self.data = data
        self.landscapes = {}

    def build_matrix(self, escalate=None):
        """Synthetic 81x5 decision matrix.

        The true indicators are confidential. Operational intensity is
        correlated with the size of the provincial deployment; each criterion
        also carries a large idiosyncratic component so that no province is
        extreme on all five dimensions at once. Replace this method with the
        real indicators when they are available.
        """
        rng = np.random.default_rng(self.SEED)
        p = np.array([self.data.P[i] for i in self.data.I], float)
        ranks = p.argsort().argsort() / (len(p) - 1)
        base = 0.55 * ranks + 0.45 * rng.uniform(0, 1, 81)
        base = (base - base.min()) / (base.max() - base.min())

        def mix(rho):
            u = rng.uniform(0, 1, 81)
            z = rho * base + (1 - rho) * u
            return (z - z.min()) / (z.max() - z.min())

        m = [mix(r) for r in self.RHOS]
        X = np.zeros((81, 5))
        X[:, 0] = 2 + m[0] * 8.5
        X[:, 1] = 12 + m[1] * 40
        X[:, 2] = 5 + m[2] * 16
        X[:, 3] = 0.0115 - m[3] * 0.0070
        X[:, 4] = 180 + m[4] * 480
        for prov, f in (escalate or {}).items():
            k = prov - 1
            X[k, 0] = min(15, X[k, 0] + f * (15 - X[k, 0]))
            X[k, 1] = min(70, X[k, 1] + f * (70 - X[k, 1]))
            X[k, 2] = min(30, X[k, 2] + f * (30 - X[k, 2]))
            X[k, 3] = max(0.0005, X[k, 3] * (1 - 0.85 * f))
            X[k, 4] = min(1000, X[k, 4] + f * (1000 - X[k, 4]))
        return X

    @classmethod
    def topsis(cls, X, w=None):
        """Equations (12)-(14)."""
        w = cls.WEIGHTS if w is None else w
        R = X / np.sqrt((X ** 2).sum(axis=0))
        V = R * w
        pis, nis = np.zeros(5), np.zeros(5)
        for j, (_, _, kind) in enumerate(cls.CRITERIA):
            hi, lo = V[:, j].max(), V[:, j].min()
            pis[j], nis[j] = (hi, lo) if kind == "benefit" else (lo, hi)
        dp = np.sqrt(((V - pis) ** 2).sum(axis=1))
        dn = np.sqrt(((V - nis) ** 2).sum(axis=1))
        C = dn / (dp + dn)
        return dict(X=X, R=R, V=V, pis=pis, nis=nis, dp=dp, dn=dn, C=C, S=1 - C)

    def build(self):
        self.landscapes["A"] = self.topsis(self.build_matrix())
        self.landscapes["B"] = self.topsis(self.build_matrix(self.ESCALATE))
        return self.landscapes

    @classmethod
    def classify(cls, s):
        if s <= cls.LOW_CUT:
            return "Low"
        return "Medium" if s <= cls.HIGH_CUT else "High"

    def mandatory(self, landscape, R):
        S = self.landscapes[landscape]["S"]
        return tuple(sorted(i for i in self.data.I if S[i - 1] > R))


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
class Optimizer:
    def __init__(self, data):
        self.d = data
        self._cov = {}

    # -- helpers ------------------------------------------------------------
    def _cover_sets(self, M):
        """For each province i, the set of candidate centres within M km."""
        if M not in self._cov:
            self._cov[M] = {i: frozenset(self.d.neighbours(i, M))
                            for i in self.d.I}
        return self._cov[M]

    def covers(self, centres, M=250):
        cov, c = self._cover_sets(M), set(centres)
        return all(cov[i] & c for i in self.d.I)

    @staticmethod
    def _cap(cap, j):
        if cap is None:
            return None
        return cap[j] if isinstance(cap, dict) else cap

    def nearest_assignment(self, centres):
        """Uncapacitated travel-minimal assignment of a fixed centre set."""
        d = self.d
        return {i: min(centres, key=lambda j: (d.D[(i, j)], j)) for i in d.I}

    def metrics(self, centres, assign, forced=(), M=250):
        """Common performance measures of a plan."""
        d = self.d
        local = sum(d.P[i] for i in d.I if assign[i] == i)
        over = [i for i in d.I if d.D[(i, assign[i])] > M]
        return dict(
            centres=sorted(centres), assign=assign, Z=local,
            rate=local / d.total,
            pkm=sum(d.P[i] * d.D[(i, assign[i])] for i in d.I),
            load={j: sum(d.P[i] for i in d.I if assign[i] == j)
                  for j in centres},
            maxdist=max(d.D[(i, assign[i])] for i in d.I),
            violations=over, violating_personnel=sum(d.P[i] for i in over),
            # personnel of mandatory provinces who must leave them
            hr_travel=sum(d.P[i] for i in forced if assign[i] != i))

    # -- covering minimum ---------------------------------------------------
    def covering_minimum(self, M=250, forced=(), cap=None):
        """Equation (9). With a capacity vector the minimum is taken over
        centre sets that admit a capacity-feasible assignment (q_C)."""
        d = self.d
        m = pulp.LpProblem("cover", pulp.LpMinimize)
        x = _binaries("x", d.I)
        m += pulp.lpSum(x[j] for j in d.I)
        if cap is None:
            for i in d.I:
                m += pulp.lpSum(x[j] for j in d.neighbours(i, M)) >= 1
        else:
            pairs = [(i, j) for i in d.I for j in d.neighbours(i, M)]
            y = _binaries("y", pairs)
            for i in d.I:
                m += pulp.lpSum(y[(i, j)] for j in d.neighbours(i, M)) == 1
            for i, j in pairs:
                m += y[(i, j)] <= x[j]
            for j in d.I:
                m += pulp.lpSum(d.P[i] * y[(i, j)] for i in d.I
                                if (i, j) in y) <= self._cap(cap, j) * x[j]
        for j in forced:
            m += x[j] == 1
        if not _solve(m):
            return None
        return int(round(pulp.value(m.objective)))

    # -- main model -----------------------------------------------------------
    def solve(self, K, M=250, forced=(), lexicographic=True, cap=None):
        """Model (1)-(7), optionally with capacity (16). The second stage (15)
        minimises total travel among the optimal solutions so that reported
        distances are well defined."""
        d = self.d
        t0 = time.time()
        m = pulp.LpProblem("ECS", pulp.LpMaximize)
        x = _binaries("x", d.I)
        pairs = [(i, j) for i in d.I for j in d.neighbours(i, M)]
        y = _binaries("y", pairs)

        m += pulp.lpSum(d.P[i] * y[(i, i)] for i in d.I)
        for i in d.I:
            m += pulp.lpSum(y[(i, j)] for j in d.neighbours(i, M)) == 1
            for j in d.neighbours(i, M):
                m += y[(i, j)] <= x[j]
        for j in forced:
            m += x[j] == 1
        m += pulp.lpSum(x[j] for j in d.I) == K
        if cap is not None:
            for j in d.I:
                m += pulp.lpSum(d.P[i] * y[(i, j)] for i in d.I
                                if (i, j) in y) <= self._cap(cap, j) * x[j]

        if not _solve(m):
            return None
        Z = int(round(pulp.value(m.objective)))
        t1 = time.time() - t0

        if lexicographic:
            m += pulp.lpSum(d.P[i] * y[(i, i)] for i in d.I) == Z
            m.sense = pulp.LpMinimize
            m.setObjective(pulp.lpSum(d.P[i] * d.D[(i, j)] * y[(i, j)]
                                      for i, j in pairs))
            _solve(m)

        centres = sorted(j for j in d.I if pulp.value(x[j]) > 0.5)
        assign = {i: j for i, j in pairs if pulp.value(y[(i, j)]) > 0.5}
        return dict(K=K, M=M, forced=list(forced), centres=centres,
                    assign=assign, Z=Z, rate=Z / d.total,
                    pkm=sum(d.P[i] * d.D[(i, assign[i])] for i in d.I),
                    load={j: sum(d.P[i] for i in d.I if assign[i] == j)
                          for j in centres},
                    maxdist=max(d.D[(i, assign[i])] for i in d.I),
                    hr_travel=sum(d.P[i] for i in forced if assign[i] != i),
                    nvar=len(d.I) + len(pairs), cpu=time.time() - t0,
                    cpu_stage1=t1)

    def solve_policy(self, N=17, M=250, forced=(), cap=None):
        """Cardinality rule (10); with capacity, q is replaced by q_C."""
        V, q = len(forced), self.covering_minimum(M, forced, cap)
        if q is None:
            return None
        K = max(N, V, q)
        r = self.solve(K, M, forced, cap=cap)
        if r is not None:
            driver = ("Budget = coverage" if K == N == q
                      else "Budget" if K == N else "Coverage")
            r.update(N=N, V=V, q=q, driver=driver)
        return r

    # -- elastic (best-effort) model ------------------------------------------
    def elastic(self, K, M=250, forced=(), cap=None):
        """Best-effort variant (17)-(19) for a hard budget K.

        Every pair (i, j) is admissible; the penalty variable u_i records by
        how many kilometres the assignment of province i exceeds M:
            u_i >= sum_j (d_ij - M)^+ y_ij,   u_i >= 0.
        The penalised objective  max sum_i P_i y_ii - lam * sum_i P_i u_i
        with lam > sum_i P_i is equivalent to the lexicographic order
        (minimise excess personnel-km first, then maximise Z), which is how it
        is solved here to avoid large coefficients. A third stage minimises
        total travel. When a plan satisfying the limit exists, the penalty is
        zero and the result coincides with model (1)-(7)."""
        d = self.d
        t0 = time.time()
        pairs = [(i, j) for i in d.I for j in d.I]
        exc = {(i, j): max(0, d.D[(i, j)] - M) for i, j in pairs}
        m = pulp.LpProblem("ECS_elastic", pulp.LpMinimize)
        x = _binaries("x", d.I)
        y = _binaries("y", pairs)
        u = {i: pulp.LpVariable(f"u_{i}", lowBound=0) for i in d.I}
        for i in d.I:
            m += pulp.lpSum(y[(i, j)] for j in d.I) == 1
            m += u[i] >= pulp.lpSum(exc[(i, j)] * y[(i, j)] for j in d.I
                                    if exc[(i, j)] > 0)
            for j in d.I:
                m += y[(i, j)] <= x[j]
        for j in forced:
            m += x[j] == 1
        m += pulp.lpSum(x[j] for j in d.I) == K
        if cap is not None:
            for j in d.I:
                m += pulp.lpSum(d.P[i] * y[(i, j)] for i in d.I) \
                    <= self._cap(cap, j) * x[j]
        penalty = pulp.lpSum(d.P[i] * u[i] for i in d.I)
        local = pulp.lpSum(d.P[i] * y[(i, i)] for i in d.I)
        travel = pulp.lpSum(d.P[i] * d.D[(i, j)] * y[(i, j)] for i, j in pairs)

        m.setObjective(penalty)                       # stage 1
        if not _solve(m):
            return None
        E = int(round(pulp.value(penalty)))
        m += penalty <= E
        m.sense = pulp.LpMaximize                     # stage 2
        m.setObjective(local)
        _solve(m)
        Z = int(round(pulp.value(local)))
        m += local >= Z
        m.sense = pulp.LpMinimize                     # stage 3
        m.setObjective(travel)
        _solve(m)

        centres = sorted(j for j in d.I if pulp.value(x[j]) > 0.5)
        assign = {i: j for i, j in pairs if pulp.value(y[(i, j)]) > 0.5}
        r = self.metrics(centres, assign, forced, M)
        r.update(K=K, M=M, excess_pkm=E, cpu=time.time() - t0)
        return r

    def lp_bound(self, K, M=250, forced=()):
        """Optimal value of the linear relaxation of (1)-(7)."""
        d = self.d
        m = pulp.LpProblem("ECS_LP", pulp.LpMaximize)
        x = {j: pulp.LpVariable(f"x_{j}", 0, 1) for j in d.I}
        pairs = [(i, j) for i in d.I for j in d.neighbours(i, M)]
        y = {p: pulp.LpVariable(f"y_{p[0]}_{p[1]}", 0, 1) for p in pairs}
        m += pulp.lpSum(d.P[i] * y[(i, i)] for i in d.I)
        for i in d.I:
            m += pulp.lpSum(y[(i, j)] for j in d.neighbours(i, M)) == 1
            for j in d.neighbours(i, M):
                m += y[(i, j)] <= x[j]
        for j in forced:
            m += x[j] == 1
        m += pulp.lpSum(x.values()) == K
        return pulp.value(m.objective) if _solve(m) else None

    # -- diagnostics ----------------------------------------------------------
    def criticality(self, centres, M=250):
        """Which provinces are stranded when each centre is removed."""
        d = self.d
        out = {}
        for j in centres:
            rest = set(centres) - {j}
            unc = [i for i in d.I if not any(d.D[(i, k)] <= M for k in rest)]
            out[j] = dict(home=d.P[j], uncovered=unc,
                          stranded=sum(d.P[i] for i in unc),
                          status="Redundant" if not unc else "Critical")
        return out

    def evaluate_plan(self, centres, M=250):
        """Audit an externally supplied plan. Assignment is to the nearest
        open centre without imposing the M-km bound, so that plans which
        cannot satisfy it remain measurable."""
        d = self.d
        a = {i: min(centres, key=lambda j: d.D[(i, j)]) for i in d.I}
        local = sum(d.P[i] for i in d.I if a[i] == i)
        over = [i for i in d.I if d.D[(i, a[i])] > M]
        return dict(centres=sorted(centres), assign=a, Z=local,
                    rate=local / d.total,
                    pkm=sum(d.P[i] * d.D[(i, a[i])] for i in d.I),
                    load={j: sum(d.P[i] for i in d.I if a[i] == j)
                          for j in centres},
                    violations=sorted(over, key=lambda i: -d.D[(i, a[i])]),
                    violating_personnel=sum(d.P[i] for i in over),
                    maxdist=max(d.D[(i, a[i])] for i in d.I))

    # -- benchmarks -----------------------------------------------------------
    # Every benchmark receives the same K, M and mandatory set as the
    # proposed model. Plans are evaluated with the nearest-open-centre
    # assignment, which is the travel-minimal assignment of a fixed set.
    def pmedian(self, K=17, M=250, forced=()):
        d = self.d
        t0 = time.time()
        m = pulp.LpProblem("pmed", pulp.LpMinimize)
        x = _binaries("x", d.I)
        pairs = [(i, j) for i in d.I for j in d.neighbours(i, M)]
        y = _binaries("y", pairs)
        m += pulp.lpSum(d.P[i] * d.D[(i, j)] * y[(i, j)] for i, j in pairs)
        for i in d.I:
            m += pulp.lpSum(y[(i, j)] for j in d.neighbours(i, M)) == 1
            for j in d.neighbours(i, M):
                m += y[(i, j)] <= x[j]
        for j in forced:
            m += x[j] == 1
        m += pulp.lpSum(x[j] for j in d.I) == K
        if not _solve(m):
            return None
        cpu = time.time() - t0
        a = {i: j for i, j in pairs if pulp.value(y[(i, j)]) > 0.5}
        c = sorted(j for j in d.I if pulp.value(x[j]) > 0.5)
        r = self.metrics(c, a, forced, M)
        r.update(cpu=cpu, feasible=True)
        return r

    def mclp(self, K=17, radius=250, forced=()):
        """Maximal covering location problem. At this radius the covering
        objective is flat over a very large family of centre sets; the set
        reported is the one returned by CBC."""
        d = self.d
        t0 = time.time()
        m = pulp.LpProblem("mclp", pulp.LpMaximize)
        x = _binaries("x", d.I)
        z = _binaries("z", d.I)
        m += pulp.lpSum(d.P[i] * z[i] for i in d.I)
        for i in d.I:
            m += pulp.lpSum(x[j] for j in d.neighbours(i, radius)) >= z[i]
        for j in forced:
            m += x[j] == 1
        m += pulp.lpSum(x[j] for j in d.I) == K
        if not _solve(m):
            return None
        cpu = time.time() - t0
        c = sorted(j for j in d.I if pulp.value(x[j]) > 0.5)
        r = self.metrics(c, self.nearest_assignment(c), forced, radius)
        r.update(cpu=cpu, feasible=True,
                 covered=int(round(pulp.value(m.objective))))
        return r

    def greedy(self, K=17, M=250, forced=()):
        """Greedy coverage repair followed by one-swap local search.

        Construction: starting from the mandatory set, repeatedly open the
        centre covering the most uncovered personnel until every province is
        within M; if this produces more than K centres, close non-mandatory
        centres (smallest home cohort first) whose removal keeps coverage;
        then fill up to K by home cohort. Improvement: first-improvement
        one-swap search on the home cohort subject to coverage. Returns
        feasible=False when no K-centre cover is reached."""
        d = self.d
        t0 = time.time()
        cov = self._cover_sets(M)
        fset = set(forced)

        checks = [0]

        def feasible(c):
            checks[0] += 1
            return all(cov[i] & c for i in d.I)

        cur = set(forced)
        while not feasible(cur):
            unc = [i for i in d.I if not cov[i] & cur]
            cur.add(max((j for j in d.I if j not in cur),
                        key=lambda j: sum(d.P[i] for i in unc
                                          if d.D[(i, j)] <= M)))
        # Cardinality repair (added in v2.1): the greedy cover may exceed K.
        dropped = True
        while len(cur) > K and dropped:
            dropped = False
            for j in sorted(cur - fset, key=lambda j: (d.P[j], j)):
                if feasible(cur - {j}):
                    cur.discard(j)
                    dropped = True
                    break
        if len(cur) > K:
            return dict(feasible=False, size=len(cur), checks=checks[0],
                        cpu=time.time() - t0)
        while len(cur) < K:
            cur.add(max((j for j in d.I if j not in cur), key=lambda j: d.P[j]))
        improved = True
        while improved:
            improved = False
            for out in list(cur - fset):
                for inn in (j for j in d.I if j not in cur):
                    trial = (cur - {out}) | {inn}
                    if feasible(trial) and \
                            sum(d.P[j] for j in trial) > sum(d.P[j] for j in cur):
                        cur, improved = trial, True
                        break
                if improved:
                    break
        cpu = time.time() - t0
        c = sorted(cur)
        r = self.metrics(c, self.nearest_assignment(c), forced, M)
        r.update(cpu=cpu, feasible=True, checks=checks[0])
        return r

    def sequential(self, K=17, M=250, forced=(), base=None):
        """Risk-blind optimisation followed by post-hoc insertion of the
        mandatory provinces, the way a risk requirement is typically
        accommodated when it is not part of the model: each mandatory
        province (highest risk first, i.e. in the order given) replaces the
        non-mandatory centre whose removal keeps coverage and loses the
        smallest home cohort. feasible=False if some insertion admits no
        coverage-preserving swap."""
        d = self.d
        t0 = time.time()
        if base is None:
            base = self.solve(K, M)["centres"]
        cur, fixed = set(base), set()
        for h in forced:
            fixed.add(h)
            if h in cur:
                continue
            best = None
            for out in sorted(cur - fixed, key=lambda j: (d.P[j], j)):
                if self.covers((cur - {out}) | {h}, M):
                    best = out
                    break
            if best is None:
                return dict(feasible=False, cpu=time.time() - t0)
            cur = (cur - {best}) | {h}
        cpu = time.time() - t0
        c = sorted(cur)
        r = self.metrics(c, self.nearest_assignment(c), forced, M)
        r.update(cpu=cpu, feasible=True)
        return r

    def risk_blind(self, K=17, M=250, forced=()):
        """Model (1)-(7) without constraint (5); the mandatory set is used
        only to measure how many high-risk personnel would have to leave."""
        r = self.solve(K, M)
        r["hr_travel"] = sum(self.d.P[i] for i in forced
                             if r["assign"][i] != i)
        return r


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------
class Study:
    # Numbering follows the article (Table 1 is the literature comparison,
    # which is not computed).
    TABLES = ["Table 2", "Table 3", "Table 4", "Table 5", "Table 6",
              "Table 7", "Table 8", "Table 9", "Table 10", "Table 11",
              "Table 12", "Table 13", "Full risk table"]
    CAPACITIES = (None, 6000, 5000, 4500, 4000, 3500)
    BENCH_V = (2, 5, 8, 10)
    ELASTIC_V = (0, 8, 10, 12, 15)
    FIGURES = ["Figure 1 - System architecture",
               "Figure 2 - Selected centers (map)",
               "Figure 3 - V > N regime",
               "Figure 4 - Center count sweep",
               "Figure 5 - Distance and risk thresholds",
               "Figure 6 - Two risk landscapes",
               "Figure 7 - Weight robustness"]

    def __init__(self, data):
        self.d = data
        self.risk = RiskModel(data)
        self.opt = Optimizer(data)
        self.res = {}

    # -- weight robustness --------------------------------------------------
    def weight_robustness(self, R, N, M, XA, XB, n_draws=500,
                          sigmas=(0.10, 0.20, 0.30), say=None):
        """Perturb the policy weights and record how often the mandatory set
        and the optimal centre sets survive."""
        d, opt = self.d, self.opt
        cache = {}

        def solve_cached(forced):
            key = tuple(sorted(forced))
            if key not in cache:
                cache[key] = opt.solve_policy(N, M, key)
            return cache[key]

        def evaluate(w):
            SA = RiskModel.topsis(XA, w)["S"]
            SB = RiskModel.topsis(XB, w)["S"]
            HA = tuple(sorted(i for i in d.I if SA[i - 1] > R))
            HB = tuple(sorted(i for i in d.I if SB[i - 1] > R))
            r1, r2 = solve_cached(HA), solve_cached(HB)
            return HB, tuple(r1["centres"]), r1["Z"], tuple(r2["centres"]), r2["Z"]

        base = evaluate(RiskModel.WEIGHTS)
        rows = []
        for sigma in sigmas:
            if say:
                say(f"Weight robustness, sigma = {sigma:.2f} ...")
            rng = np.random.default_rng(20260827)
            hits = dict(hb=0, c1=0, c2=0, p30=0, p62=0)
            z2 = []
            for _ in range(n_draws):
                w = RiskModel.WEIGHTS * np.exp(rng.normal(0, sigma, 5))
                w = w / w.sum()
                HB, C1, _, C2, Z2 = evaluate(w)
                hits["hb"] += HB == base[0]
                hits["c1"] += C1 == base[1]
                hits["c2"] += C2 == base[3]
                hits["p30"] += 30 in HB
                hits["p62"] += 62 in HB
                z2.append(Z2)
            z2 = np.array(z2)
            rows.append(dict(sigma=sigma, n=n_draws,
                             **{k: v / n_draws for k, v in hits.items()},
                             z2_min=int(z2.min()), z2_max=int(z2.max())))

        oat = []
        for c in range(5):
            for f in (0.5, 0.75, 1.25, 1.5):
                w = RiskModel.WEIGHTS.copy()
                w[c] *= f
                w = w / w.sum()
                HB, C1, Z1, C2, Z2 = evaluate(w)
                oat.append(dict(crit=f"C{c+1}", factor=f, HB=HB, Z1=Z1, Z2=Z2,
                                same1=C1 == base[1], same2=C2 == base[3]))
        return dict(base_HB=base[0], base_Z1=base[2], global_=rows, oat=oat)

    # -- full run -----------------------------------------------------------
    def run(self, N=17, M=250, R=0.75, current_plan=CURRENT_PLAN,
            progress=None):
        say = progress or (lambda *_: None)
        d, opt = self.d, self.opt

        say("Evaluating risk landscapes ...")
        self.risk.build()
        SA = self.risk.landscapes["A"]["S"]
        SB = self.risk.landscapes["B"]["S"]
        HA, HB = self.risk.mandatory("A", R), self.risk.mandatory("B", R)

        say("Solving the two scenarios ...")
        s1, s2 = opt.solve_policy(N, M, HA), opt.solve_policy(N, M, HB)

        say("Computing coverage criticality ...")
        crit = opt.criticality(s1["centres"], M)

        say("Auditing the current institutional plan ...")
        cur = opt.evaluate_plan(current_plan, M)
        keep = opt.solve_policy(N, M, tuple(current_plan))

        say("Running the V > N regime ...")
        order = sorted(d.I, key=lambda i: -SB[i - 1])
        vreg = []
        for V in (0, 2, 5, 8, 10, 12, 15, 18, 20, 25):
            forced = tuple(order[:V])
            r = opt.solve_policy(N, M, forced)
            old = opt.solve(max(N, V), M, forced, lexicographic=False)
            vreg.append(dict(V=V, q=r["q"], K=r["K"], driver=r["driver"],
                             Z=r["Z"], rate=r["rate"],
                             old_feasible=old is not None))

        say("Sweeping the center count ...")
        nsweep = [dict(N=k, **{key: opt.solve(k, M)[key]
                               for key in ("Z", "rate", "pkm")})
                  for k in range(11, 31)]

        say("Sweeping the distance threshold ...")
        msweep = []
        for mm in (150, 175, 200, 225, 250, 275, 300, 350, 400):
            r = opt.solve(N, mm)
            msweep.append(dict(M=mm, q=opt.covering_minimum(mm),
                               Z=r["Z"] if r else None,
                               rate=r["rate"] if r else None,
                               pkm=r["pkm"] if r else None))

        say("Sweeping the risk threshold ...")
        rsweep = []
        for rt in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            f = tuple(sorted(i for i in d.I if SB[i - 1] > rt))
            r = opt.solve_policy(N, M, f)
            rsweep.append(dict(R=rt, V=len(f), q=r["q"], K=r["K"],
                               Z=r["Z"], rate=r["rate"]))

        say("Running benchmarks ...")
        bench = dict(proposed=s1, pmedian=opt.pmedian(N, M),
                     mclp=opt.mclp(N, M), greedy=opt.greedy(N, M),
                     lp=opt.lp_bound(s1["K"], M))

        say("Running benchmarks under mandatory inclusion ...")
        bench_v = []
        for V in self.BENCH_V:
            forced = tuple(order[:V])
            prop = opt.solve_policy(N, M, forced)
            K = prop["K"]
            bench_v.append(dict(
                V=V, K=K, forced=forced, hr=sum(d.P[i] for i in forced),
                methods=[("Proposed model", prop),
                         ("p-median", opt.pmedian(K, M, forced)),
                         ("Maximal covering", opt.mclp(K, M, forced)),
                         ("Greedy + 1-swap", opt.greedy(K, M, forced)),
                         ("Sequential insertion",
                          opt.sequential(K, M, forced)),
                         ("Risk-blind optimum",
                          opt.risk_blind(K, M, forced))]))

        say("Capacity sensitivity ...")
        capacity = []
        for C in self.CAPACITIES:
            row = dict(C=C)
            for tag, forced in (("s1", HA), ("s2", HB)):
                t0 = time.time()
                res_c = opt.solve_policy(N, M, forced, cap=C)
                res_c["cpu_total"] = time.time() - t0   # incl. q_C
                row[tag] = res_c
            capacity.append(row)

        say("Best-effort plans under a hard budget ...")
        elastic = []
        for V in self.ELASTIC_V:
            forced = tuple(order[:V])
            pol = opt.solve_policy(N, M, forced)
            el = opt.elastic(N, M, forced)
            elastic.append(dict(V=V, q=pol["q"], K=pol["K"], Zpol=pol["Z"],
                                ratepol=pol["rate"], el=el))

        rob = self.weight_robustness(R, N, M,
                                     self.risk.landscapes["A"]["X"],
                                     self.risk.landscapes["B"]["X"], say=say)

        self.res = dict(N=N, M=M, R=R, SA=SA, SB=SB, HA=HA, HB=HB, s1=s1,
                        s2=s2, crit=crit, cur=cur, keep=keep, vreg=vreg,
                        nsweep=nsweep, msweep=msweep, rsweep=rsweep,
                        bench=bench, bench_v=bench_v, capacity=capacity,
                        elastic=elastic, rob=rob,
                        current_plan=tuple(current_plan))
        say("Done.")
        return self.res

    # -- tables -------------------------------------------------------------
    def table(self, name):
        r, d = self.res, self.d
        if not r:
            return [], []

        if name == "Table 2":
            rng = ["0-15", "0-70", "1-30", "0.0005-0.015", "10-1000"]
            return (["Code", "Criterion", "Range", "Direction", "Effect"],
                    [[c, n, g, "Benefit" if k == "benefit" else "Cost",
                      "Decreasing" if k == "benefit" else "Increasing"]
                     for (c, n, k), g in zip(RiskModel.CRITERIA, rng)])

        if name == "Table 3":
            s1, s2 = r["s1"], r["s2"]
            codes = sorted(set(s1["centres"]) | set(s2["centres"]),
                           key=lambda j: -d.P[j])
            rows = [[j, f"{d.P[j]:,}",
                     f"{s1['load'][j]:,}" if j in s1["centres"] else "-",
                     f"{r['SA'][j-1]:.4f}" if j in s1["centres"] else "-",
                     f"{s2['load'][j]:,}" if j in s2["centres"] else "-",
                     f"{r['SB'][j-1]:.4f}" if j in s2["centres"] else "-"]
                    for j in codes]
            rows.append(["Total", "", f"{d.total:,}", "", f"{d.total:,}", ""])
            return (["Code", "Home", "S1 load", "S (A)", "S2 load", "S (B)"],
                    rows)

        if name == "Table 4":
            crit = r["crit"]
            return (["Code", "Home", "Stranded provinces", "Personnel",
                     "Status"],
                    [[j, f"{crit[j]['home']:,}",
                      ", ".join(map(str, crit[j]["uncovered"])) or "none",
                      f"{crit[j]['stranded']:,}", crit[j]["status"]]
                     for j in sorted(crit, key=lambda j: -crit[j]["stranded"])])

        if name == "Table 5":
            cur, s1, s2 = r["cur"], r["s1"], r["s2"]
            base = cur["rate"]
            return (["Metric", "Current plan", "Scenario 1", "Scenario 2"], [
                ["Number of centers", len(cur["centres"]),
                 len(s1["centres"]), len(s2["centres"])],
                ["Examined locally", f"{cur['Z']:,}", f"{s1['Z']:,}",
                 f"{s2['Z']:,}"],
                ["Local rate", f"{base:.2%}", f"{s1['rate']:.2%}",
                 f"{s2['rate']:.2%}"],
                ["Gain", "-", f"{100*(s1['rate']-base):+.2f} pp",
                 f"{100*(s2['rate']-base):+.2f} pp"],
                ["Travellers", f"{d.total-cur['Z']:,}", f"{d.total-s1['Z']:,}",
                 f"{d.total-s2['Z']:,}"],
                ["Personnel-km", f"{cur['pkm']:,}", f"{s1['pkm']:,}",
                 f"{s2['pkm']:,}"],
                ["Longest journey", f"{cur['maxdist']} km",
                 f"{s1['maxdist']} km", f"{s2['maxdist']} km"],
                ["Beyond 250 km", f"{cur['violating_personnel']:,}", "0", "0"],
            ])

        if name == "Table 6":
            return (["V", "q", "K", "Binding driver", "Examined locally",
                     "Rate", "max(N,V) rule"],
                    [[v["V"], v["q"], v["K"], v["driver"], f"{v['Z']:,}",
                      f"{v['rate']:.2%}",
                      "Feasible" if v["old_feasible"] else "Infeasible"]
                     for v in r["vreg"]])

        if name == "Table 7":
            return (["N", "Examined locally", "Rate", "Personnel-km"],
                    [[x["N"], f"{x['Z']:,}", f"{x['rate']:.2%}",
                      f"{x['pkm']:,}"] for x in r["nsweep"]])

        if name == "Table 8":
            return (["sigma", "I_H unchanged", "S1 set unchanged",
                     "S2 set unchanged", "Z range (S2)"],
                    [[f"{g['sigma']:.2f}", f"{g['hb']:.1%}", f"{g['c1']:.1%}",
                      f"{g['c2']:.1%}", f"{g['z2_min']:,}-{g['z2_max']:,}"]
                     for g in r["rob"]["global_"]])

        if name == "Table 9":
            b = r["bench"]
            rows = []
            for label, x in (("Proposed model", r["s1"]),
                             ("p-median", b["pmedian"]),
                             ("Maximal covering", b["mclp"]),
                             ("Greedy + 1-swap", b["greedy"])):
                rows.append([label, f"{x['Z']:,}", f"{x['rate']:.2%}",
                             f"{x['pkm']:,}", f"{x['maxdist']} km",
                             f"{x['cpu']:.2f}"])
            return (["Model", "Examined locally", "Rate", "Personnel-km",
                     "Longest", "CPU (s)"], rows)

        if name == "Table 10":
            rows = []
            for blk in r["bench_v"]:
                for label, x in blk["methods"]:
                    head = f"V={blk['V']}, K={blk['K']}"
                    if x is None or not x.get("feasible", True):
                        rows.append([head, label, "no K-center cover found",
                                     "", "", "", "", ""])
                        continue
                    rows.append([head, label, f"{x['Z']:,}",
                                 f"{x['rate']:.2%}", f"{x['pkm']:,}",
                                 f"{x['maxdist']} km", f"{x['hr_travel']:,}",
                                 f"{x['cpu']:.2f}"])
            return (["Instance", "Model", "Examined locally", "Rate",
                     "Personnel-km", "Longest", "High-risk travelling",
                     "CPU (s)"], rows)

        if name == "Table 11":
            rows = []
            for c in r["capacity"]:
                cells = ["none" if c["C"] is None else f"{c['C']:,}"]
                for tag in ("s1", "s2"):
                    x = c[tag]
                    cells += [x["q"], f"{x['Z']:,}", f"{x['pkm']:,}",
                              f"{max(x['load'].values()):,}"]
                rows.append(cells)
            return (["Capacity C", "q_C (S1)", "Local (S1)", "Pers.-km (S1)",
                     "Max load (S1)", "q_C (S2)", "Local (S2)",
                     "Pers.-km (S2)", "Max load (S2)"], rows)

        if name == "Table 12":
            rows = []
            for e in r["elastic"]:
                el = e["el"]
                rows.append([e["V"], e["q"], e["K"], f"{e['Zpol']:,}",
                             f"{el['Z']:,}", f"{el['violating_personnel']:,}",
                             len(el["violations"]),
                             f"{el['maxdist']} km", f"{el['excess_pkm']:,}"])
            return (["V", "q", "K by (10)", "Local at K",
                     "Local at N (best effort)", "Personnel beyond M",
                     "Provinces beyond M", "Longest", "Excess personnel-km"],
                    rows)

        if name == "Table 13":
            A = self.risk.landscapes["A"]
            rows = [[p] + [f"{A['V'][p-1, c]:.5f}" for c in range(5)] +
                    [f"{A['dp'][p-1]:.5f}", f"{A['dn'][p-1]:.5f}",
                     f"{A['C'][p-1]:.4f}", f"{A['S'][p-1]:.4f}"]
                    for p in (1, 6, 21, 30, 62, 73)]
            rows.append(["Ideal"] + [f"{A['pis'][c]:.5f}" for c in range(5)] +
                        [""] * 4)
            rows.append(["Anti-ideal"] +
                        [f"{A['nis'][c]:.5f}" for c in range(5)] + [""] * 4)
            return (["Province", "v1", "v2", "v3", "v4", "v5", "D+", "D-",
                     "C*", "S"], rows)

        if name == "Full risk table":
            XA = self.risk.landscapes["A"]["X"]
            return (["Province", "C1", "C2", "C3", "C4", "C5", "S (A)",
                     "Class (A)", "S (B)", "Class (B)"],
                    [[i, f"{XA[i-1,0]:.2f}", f"{XA[i-1,1]:.1f}",
                      f"{XA[i-1,2]:.2f}", f"{XA[i-1,3]*1000:.2f}",
                      f"{XA[i-1,4]:.0f}", f"{r['SA'][i-1]:.4f}",
                      RiskModel.classify(r["SA"][i-1]), f"{r['SB'][i-1]:.4f}",
                      RiskModel.classify(r["SB"][i-1])] for i in d.I])
        return [], []

    # -- figures ------------------------------------------------------------
    def figure(self, name, fig=None):
        with plt.rc_context(PLOT_RC):
            fig = fig or Figure()
            fig.clear()
            k = name.split()[1]
            {"1": self._f_arch, "2": self._f_map, "3": self._f_vreg,
             "4": self._f_nsweep, "5": self._f_sens, "6": self._f_land,
             "7": self._f_rob}[k](fig)
        return fig

    def _f_arch(self, fig):
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        fig.set_size_inches(6.95, 2.25)
        ax = fig.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(-4, 34)
        ax.axis("off")
        spec = [
            (0.8, "DATA LAYER", ["Personnel counts $P_i$",
                                 "Distance matrix $d_{ij}$",
                                 "Security indicators", "Consistency checks"],
             "#eef2f6", ACC),
            (20.6, "RISK ENGINE", ["TOPSIS with policy", "weights $w$",
                                   "Risk scores $S_i$", "Mandatory set $I_H$"],
             "#fdeeee", ALERT),
            (40.4, "OPTIMIZATION CORE", ["Covering minimum $q$ (9)",
                                         "Cardinality rule (10)",
                                         "Model (1)\u2013(7), (15)",
                                         "Capacity, elastic (16)\u2013(19)"],
             "#eef6ef", GOOD),
            (60.2, "ANALYSIS", ["Coverage criticality", "Current-plan audit",
                                "Sweeps and robustness", "Benchmarks"],
             "#eef2f6", ACC),
            (80.0, "USER INTERFACE", ["Scenario control", "Tables and figures",
                                      "What-if solver", "Export and reporting"],
             "#f4f6f8", MUTED),
        ]
        for x, title, lines, fc, ec in spec:
            ax.add_patch(FancyBboxPatch(
                (x, 7), 17.6, 23,
                boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.0,
                facecolor=fc, edgecolor=ec))
            ax.text(x + 8.8, 28, title, ha="center", va="top", fontsize=7.4,
                    fontweight="bold", color=ec)
            for k, t in enumerate(lines):
                ax.text(x + 8.8, 22.5 - 4.1 * k, t, ha="center", va="top",
                        fontsize=6.3, color=INK)
        for a, b in zip(spec[:-1], spec[1:]):
            ax.add_patch(FancyArrowPatch((a[0] + 17.6, 18.5), (b[0], 18.5),
                                         arrowstyle="-|>", mutation_scale=9,
                                         lw=1.0, color=MUTED))
        # scenario feedback loop: interface back to the risk engine
        x_ui, x_risk = 80.0 + 8.8, 20.6 + 8.8
        ax.plot([x_ui, x_ui, x_risk], [6.6, 2.0, 2.0], ls="--", lw=1.0,
                color=MUTED)
        ax.add_patch(FancyArrowPatch((x_risk, 2.0), (x_risk, 6.6),
                                     arrowstyle="-|>", mutation_scale=9,
                                     lw=1.0, color=MUTED))
        ax.text((x_ui + x_risk) / 2, 0.6, "scenario feedback: revised "
                "weights, thresholds and mandatory sets", ha="center",
                va="top", fontsize=6.2, style="italic", color=MUTED)
        fig.tight_layout(pad=0.2)

    def _panel(self, ax, res, title, forced=()):
        d = self.d
        ax.add_patch(Polygon(OUTLINE, closed=True, facecolor=PANEL,
                             edgecolor="#b6c2cc", lw=.8, zorder=0))
        for i in d.I:
            j = res["assign"][i]
            if i != j:
                ax.plot([COORD[i][1], COORD[j][1]],
                        [COORD[i][0], COORD[j][0]],
                        color="#9fb6cc", lw=.5, zorder=1, alpha=.85)
        non = [i for i in d.I if i not in res["centres"]]
        ax.scatter([COORD[i][1] for i in non], [COORD[i][0] for i in non],
                   s=11, c="white", edgecolors=MUTED, linewidths=.5, zorder=2)
        cen = [j for j in res["centres"] if j not in forced]
        ax.scatter([COORD[j][1] for j in cen], [COORD[j][0] for j in cen],
                   s=[20 + d.P[j] / 30 for j in cen], c=ACC,
                   edgecolors="white", linewidths=.6, zorder=4)
        if forced:
            ax.scatter([COORD[j][1] for j in forced],
                       [COORD[j][0] for j in forced],
                       s=[20 + d.P[j] / 30 for j in forced], c=ALERT,
                       marker="s", edgecolors="white", linewidths=.6, zorder=5)
        for j in res["centres"]:
            ax.annotate(str(j), (COORD[j][1], COORD[j][0]),
                        textcoords="offset points", xytext=(0, 5.5),
                        ha="center", fontsize=6.2, fontweight="bold",
                        color=ALERT if j in forced else ACC, zorder=6)
        ax.set_xlim(25.4, 45.3)
        ax.set_ylim(35.4, 42.6)
        ax.set_aspect(1 / np.cos(np.radians(39)))
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")

    def _f_map(self, fig):
        r = self.res
        fig.set_size_inches(6.95, 5.6)
        a1, a2 = fig.subplots(2, 1)
        s1, s2 = r["s1"], r["s2"]
        self._panel(a1, s1, f"(a) Scenario 1 \u2014 $V={s1['V']}$, "
                            f"$K={s1['K']}$: {s1['Z']:,} examined locally "
                            f"({s1['rate']:.2%})")
        self._panel(a2, s2, f"(b) Scenario 2 \u2014 $V={s2['V']}$, "
                            f"$K={s2['K']}$: {s2['Z']:,} examined locally "
                            f"({s2['rate']:.2%})", forced=r["HB"])
        leg = [Line2D([], [], marker="o", ls="", mfc=ACC, mec="w", ms=6,
                      label="Selected examination center"),
               Line2D([], [], marker="s", ls="", mfc=ALERT, mec="w", ms=6,
                      label="Mandatory high-risk center"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=5,
                      label="Province assigned to a center"),
               Line2D([], [], color="#9fb6cc", lw=1,
                      label=f"Assignment (\u2264 {r['M']} km)")]
        fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False,
                   fontsize=7)
        fig.tight_layout(rect=[0, .045, 1, 1])

    def _f_vreg(self, fig):
        r = self.res
        fig.set_size_inches(6.95, 2.7)
        a1, a2 = fig.subplots(1, 2)
        vr = r["vreg"]
        V = [v["V"] for v in vr]
        K = [v["K"] for v in vr]
        a1.fill_between(V, [r["N"]] * len(V), K, color=ACC, alpha=.08, lw=0)
        a1.plot(V, [r["N"]] * len(V), "-", color=MUTED, lw=1.1,
                label=f"budget $N={r['N']}$")
        a1.plot(V, V, ":", color=ALERT, lw=1.3, label="risk $V$")
        a1.plot(V, [v["q"] for v in vr], "--", color=GOOD, lw=1.3,
                label="coverage $q(M, I_H)$")
        a1.plot(V, K, "o-", color=ACC, ms=4, lw=1.7,
                label="realized $K=\\max(N, V, q)$")
        bad = [v for v in vr if not v["old_feasible"]]
        a1.plot([v["V"] for v in bad], [v["K"] for v in bad], "x",
                color=ALERT, ms=6, mew=1.5, ls="",
                label="$K=\\max(N, V)$ infeasible")
        a1.set_xlabel("Mandatory high-risk provinces, $V$")
        a1.set_ylabel("Number of examination centers")
        a1.set_title("(a) Which force sets the center count", fontsize=8.5)
        a1.legend(fontsize=6.2, frameon=False, loc="upper left")
        a1.grid(alpha=.25, lw=.5)
        # shade the budget-driven and coverage-driven regimes
        last_b = max(v["V"] for v in vr if v["K"] == r["N"])
        first_c = min((v["V"] for v in vr if v["K"] > r["N"]),
                      default=V[-1])
        cut = (last_b + first_c) / 2
        a2.axvspan(V[0] - 0.5, cut, color=MUTED, alpha=.10, lw=0)
        a2.axvspan(cut, V[-1] + 0.5, color=GOOD, alpha=.10, lw=0)
        a2.axvline(cut, color=MUTED, lw=.6, alpha=.6)
        a2.plot(V, [100 * v["rate"] for v in vr], "o-", color=ACC,
                ms=4, lw=1.6)
        a2.set_xlim(V[0] - 0.5, V[-1] + 0.5)
        a2.set_ylim(30, 56)
        a2.text(cut / 2, 52.5, "budget-driven\n$K=N$", ha="center",
                va="center", fontsize=6.4, color=MUTED)
        a2.text((cut + V[-1]) / 2, 52.5, "coverage-driven\n$K=q>N$",
                ha="center", va="center", fontsize=6.4, color=GOOD)
        a2.set_xlabel("Mandatory high-risk provinces, $V$")
        a2.set_ylabel("Local examination rate (%)")
        a2.set_title("(b) Cost of binding security", fontsize=8.5)
        a2.grid(alpha=.25, lw=.5)
        fig.tight_layout()

    def _f_nsweep(self, fig):
        r, d = self.res, self.d
        fig.set_size_inches(6.95, 2.7)
        ax = fig.subplots()
        ns = r["nsweep"]
        N = [x["N"] for x in ns]
        rates = [100 * x["rate"] for x in ns]
        ax.plot(N, rates, "o-", color=ACC, ms=3.6, lw=1.4)
        ax.set_xlabel("Number of examination centers, $N$")
        ax.set_ylabel("Local examination rate (%)", color=ACC)
        ax.tick_params(axis="y", labelcolor=ACC)
        ax.set_xticks(N[::2])
        ax.axvline(r["N"], color=ALERT, ls="--", lw=.9)
        cur = 100 * r["cur"]["rate"]
        ax.axhline(cur, color=MUTED, ls=":", lw=.9)
        ax.text(N[0] + 1.6, cur + 0.6,
                f"current plan ({r['cur']['rate']:.2%}, infeasible at "
                f"{r['M']} km)", fontsize=6.2, color="#4a5a6a", va="bottom")
        q0 = N[0]
        ax.axvline(q0, color=GOOD, ls="-.", lw=.9)
        ax.annotate(f"$q={q0}$: minimum for\nfull {r['M']}-km coverage",
                    xy=(q0, rates[0] + 7), xytext=(q0 + 1.8, rates[0] + 4),
                    fontsize=6.2, color=GOOD,
                    arrowprops=dict(arrowstyle="->", color=GOOD, lw=.7))
        k = N.index(r["N"])
        ax.annotate(f"current configuration ($N={r['N']}$)",
                    xy=(r["N"], rates[0] + 13 - 13), xytext=(r["N"] + 3.5,
                                                             rates[0] + 1),
                    fontsize=6.2, color=ALERT,
                    arrowprops=dict(arrowstyle="->", color=ALERT, lw=.7))
        ax2 = ax.twinx()
        ax2.plot(N, [x["pkm"] / 1e6 for x in ns], "s--",
                 color="#e07b00", ms=3.2, lw=1.2)
        ax2.set_ylabel("Total travel (million personnel-km)", color="#e07b00")
        ax2.tick_params(axis="y", labelcolor="#e07b00")
        ax.grid(alpha=.25, lw=.5)
        fig.tight_layout()

    def _f_sens(self, fig):
        r = self.res
        fig.set_size_inches(6.95, 2.6)
        a1, a2 = fig.subplots(1, 2)
        ms = [x for x in r["msweep"] if x["rate"] is not None]
        a1.plot([x["M"] for x in ms], [100 * x["rate"] for x in ms], "o-",
                color=ACC, ms=3.8, lw=1.4)
        for x in ms:
            a1.annotate(f"$q={x['q']}$", (x["M"], 100 * x["rate"]),
                        textcoords="offset points", xytext=(0, -11),
                        fontsize=6.0, ha="center", color=GOOD)
        a1.set_xlabel("Maximum travel distance, $M$ (km)")
        a1.set_ylabel("Local examination rate (%)")
        a1.set_title(f"(a) Accessibility threshold, $N={r['N']}$",
                     fontsize=8.5)
        a1.grid(alpha=.25, lw=.5)
        R = [x["R"] for x in r["rsweep"]]
        a2.plot(R, [100 * x["rate"] for x in r["rsweep"]], "o-", color=ALERT,
                ms=3.8, lw=1.4)
        a2b = a2.twinx()
        a2b.step(R, [x["V"] for x in r["rsweep"]], where="mid", color=GOOD,
                 lw=1.2, ls="--")
        a2b.set_ylabel("Mandatory provinces $V$", color=GOOD)
        a2b.tick_params(axis="y", labelcolor=GOOD)
        a2.set_xlabel("Risk-classification threshold, $R$")
        a2.set_ylabel("Local examination rate (%)", color=ALERT)
        a2.tick_params(axis="y", labelcolor=ALERT)
        a2.set_title("(b) Risk threshold, Landscape B", fontsize=8.5)
        a2.grid(alpha=.25, lw=.5)
        fig.tight_layout()

    def _f_land(self, fig):
        r = self.res
        fig.set_size_inches(6.95, 2.5)
        ax = fig.subplots()
        SA, SB = r["SA"], r["SB"]
        order = np.argsort(-SA)
        ax.plot(range(len(SA)), SA[order], "o-", color=ACC, ms=2.6, lw=1.0,
                label="Landscape A (baseline)")
        ax.plot(range(len(SB)), SB[order], "s-", color=ALERT, ms=2.6, lw=1.0,
                alpha=.85, label="Landscape B (two provinces deteriorate)")
        ax.axhline(r["R"], color=GOOD, ls="--", lw=1.0)
        ax.text(1, r["R"] + 0.015, f"$R={r['R']}$ mandatory-inclusion "
                "threshold", fontsize=6.2, color=GOOD, va="bottom")
        ax.axhline(RiskModel.LOW_CUT, color=MUTED, ls=":", lw=.9)
        ax.text(1, RiskModel.LOW_CUT + 0.015, "Low / Medium boundary "
                f"({RiskModel.LOW_CUT})", fontsize=6.2, color="#4a5a6a",
                va="bottom")
        ax.set_xlabel("Provinces, ordered by Landscape A risk score")
        ax.set_ylabel("Risk score $S_i$")
        ax.legend(fontsize=6.8, frameon=False, loc="upper right")
        ax.grid(alpha=.25, lw=.5)
        fig.tight_layout()

    def _f_rob(self, fig):
        r = self.res
        fig.set_size_inches(6.95, 2.6)
        a1, a2 = fig.subplots(1, 2)
        g = r["rob"]["global_"]
        sig = [x["sigma"] for x in g]
        a1.plot(sig, [100 * x["hb"] for x in g], "o-", color=ALERT, ms=4,
                lw=1.5, label="mandatory set $I_H=\\{30, 62\\}$ unchanged")
        a1.plot(sig, [100 * x["c2"] for x in g], "s-", color=ACC, ms=4, lw=1.5,
                label="Scenario 2 center set unchanged")
        a1.plot(sig, [100 * x["c1"] for x in g], "^--", color=MUTED, ms=4,
                lw=1.3, label="Scenario 1 center set unchanged")
        a1.set_xlabel("Perturbation dispersion $\\sigma$")
        a1.set_ylabel(f"Share of {g[0]['n']} draws (%)")
        a1.set_ylim(60, 102)
        a1.set_xticks(sig)
        a1.legend(fontsize=6.2, frameon=False, loc="lower left")
        a1.set_title("(a) Stability under random weight perturbation",
                     fontsize=8.5)
        a1.grid(alpha=.25, lw=.5)
        crits = [f"C{k+1}" for k in range(5)]
        for k, f in enumerate((0.5, 0.75, 1.25, 1.5)):
            vals = [next(o["Z1"] for o in r["rob"]["oat"]
                         if o["crit"] == c and abs(o["factor"] - f) < 1e-9)
                    for c in crits]
            a2.bar(np.arange(5) + (k - 1.5) * 0.2, [v / 1000 for v in vals],
                   0.2, label=f"\u00d7{f}",
                   color=["#9db8d2", "#6f95bb", ACC, "#12314c"][k])
        a2.axhline(r["rob"]["base_Z1"] / 1000, color=ALERT, ls="--", lw=1.0)
        a2.text(4.45, r["rob"]["base_Z1"] / 1000 + 0.08,
                f"baseline {r['rob']['base_Z1']:,}", ha="right",
                va="bottom", fontsize=6.2, color=ALERT)
        a2.set_xticks(range(5))
        a2.set_xticklabels(crits)
        a2.set_ylim(17.4, 20.9)
        a2.set_ylabel("Examined locally (thousands)")
        a2.set_xlabel("Criterion whose weight is scaled")
        a2.legend(fontsize=6.2, frameon=False, ncol=4, loc="upper left",
                  columnspacing=1.0, handlelength=1.2)
        a2.set_title("(b) One-at-a-time weight scaling", fontsize=8.5)
        a2.grid(alpha=.25, lw=.5, axis="y")
        fig.tight_layout()

    # -- summary ------------------------------------------------------------
    def summary(self):
        r, d = self.res, self.d
        if not r:
            return "No results yet."
        s1, s2, cur, keep, crit = (r["s1"], r["s2"], r["cur"], r["keep"],
                                   r["crit"])
        red = sorted(j for j in crit if crit[j]["status"] == "Redundant")
        L = ["=" * 74,
             " EXAMINATION CENTER SELECTION - RESULT SUMMARY",
             "=" * 74,
             f" {d.n} provinces, {d.total:,} personnel"
             f" | N={r['N']} M={r['M']} km R={r['R']}",
             f" Model size: {s1['nvar']:,} binary variables"
             f" ({d.pair_count(r['M']):,} feasible pairs of {d.n**2:,})", ""]
        A = L.append
        A("-" * 74)
        A(" RISK LANDSCAPES")
        A("-" * 74)
        A(" Policy weights: " + ", ".join(f"{v:.2f}"
                                          for v in RiskModel.WEIGHTS))
        A(f" A: range [{r['SA'].min():.4f}, {r['SA'].max():.4f}]"
          f"  I_H = {r['HA'] or 'empty'}")
        A(f" B: S_30={r['SB'][29]:.4f} S_62={r['SB'][61]:.4f}"
          f"  I_H = {r['HB']}")
        unt = [i - 1 for i in d.I if i not in (30, 62)]
        A(f" Mean |dS| over the {len(unt)} untouched provinces: "
          f"{np.abs(r['SA'] - r['SB'])[unt].mean():.4f}")
        A("")
        A("-" * 74)
        A(" SCENARIOS")
        A("-" * 74)
        for tag, s in (("Scenario 1", s1), ("Scenario 2", s2)):
            A(f" {tag}: V={s['V']} q={s['q']} K={s['K']} ({s['driver']})")
            A(f"   centers {s['centres']}")
            A(f"   local {s['Z']:,} ({s['rate']:.2%})   "
              f"travel {s['pkm']:,} km   longest {s['maxdist']} km   "
              f"cpu {s['cpu']:.2f}s")
        A(f" Dropped {sorted(set(s1['centres']) - set(s2['centres']))}, "
          f"added {sorted(set(s2['centres']) - set(s1['centres']))}, "
          f"cost {s1['Z'] - s2['Z']:,} personnel")
        A("")
        A("-" * 74)
        A(" COVERAGE CRITICALITY")
        A("-" * 74)
        A(f" Critical {len(crit) - len(red)} of {len(crit)};"
          f" redundant {red}")
        worst = max(crit, key=lambda j: crit[j]["stranded"])
        A(f" Most critical: {worst}, closing it strands "
          f"{crit[worst]['stranded']:,} personnel")
        A("")
        A("-" * 74)
        A(" CURRENT INSTITUTIONAL PLAN")
        A("-" * 74)
        A(f" centers {list(r['current_plan'])}")
        A(f" local {cur['Z']:,} ({cur['rate']:.2%})   "
          f"travel {cur['pkm']:,} km   longest {cur['maxdist']} km")
        A(f" Same-day return rule violated for "
          f"{cur['violating_personnel']:,} personnel in "
          f"{len(cur['violations'])} provinces:")
        for i in cur["violations"]:
            A(f"   province {i:>2}: {d.P[i]:>4} personnel, nearest center "
              f"{cur['assign'][i]:>2} at {d.D[(i, cur['assign'][i])]} km")
        for tag, s in (("Scenario 1", s1), ("Scenario 2", s2)):
            A(f" {tag} vs current: "
              f"{100*(s['rate'] - cur['rate']):+.2f} pp, "
              f"travel {s['pkm'] - cur['pkm']:+,} km, violations 0")
        A(f" If no center may be closed: q={keep['q']}, added "
          f"{sorted(set(keep['centres']) - set(r['current_plan']))}, "
          f"local {keep['Z']:,} ({keep['rate']:.2%})")
        A("")
        A("-" * 74)
        A(" WEIGHT ROBUSTNESS")
        A("-" * 74)
        for g in r["rob"]["global_"]:
            A(f" sigma={g['sigma']:.2f} ({g['n']} draws): "
              f"mandatory set unchanged {g['hb']:.1%}, "
              f"province 30 present {g['p30']:.1%}, "
              f"province 62 present {g['p62']:.1%}")
        same = sum(o["same2"] for o in r["rob"]["oat"])
        A(f" One-at-a-time: mandatory set unchanged in "
          f"{sum(o['HB'] == r['rob']['base_HB'] for o in r['rob']['oat'])}"
          f" of {len(r['rob']['oat'])} perturbations; "
          f"Scenario 2 set unchanged in {same}")
        A("")
        A("-" * 74)
        A(" BENCHMARKS (same K, M and mandatory set for every model)")
        A("-" * 74)
        b = r["bench"]
        A(f" V=0: LP relaxation bound {b['lp']:,.1f} vs integer optimum "
          f"{s1['Z']:,}; greedy coverage checks {b['greedy']['checks']:,}")
        for blk in r["bench_v"]:
            A(f" V={blk['V']} K={blk['K']} mandatory {list(blk['forced'])}"
              f" ({blk['hr']:,} personnel)")
            for label, x in blk["methods"]:
                if x is None or not x.get("feasible", True):
                    A(f"   {label:<22} no K-center cover found")
                else:
                    A(f"   {label:<22} local {x['Z']:>6,} ({x['rate']:.2%})"
                      f"  travel {x['pkm']:>9,}  high-risk travelling "
                      f"{x['hr_travel']:>5,}")
        A("")
        A("-" * 74)
        A(" CAPACITY SENSITIVITY (uniform capacity C per center)")
        A("-" * 74)
        for c in r["capacity"]:
            cap = "none" if c["C"] is None else f"{c['C']:,}"
            A(f" C={cap:>6}: " + "   ".join(
                f"{tag.upper()} q_C={c[tag]['q']} K={c[tag]['K']} "
                f"local {c[tag]['Z']:,} travel {c[tag]['pkm']:,} "
                f"max load {max(c[tag]['load'].values()):,}"
                for tag in ("s1", "s2")))
        A("")
        A("-" * 74)
        A(f" BEST-EFFORT PLANS WITH THE BUDGET HELD AT N={r['N']}")
        A("-" * 74)
        for e in r["elastic"]:
            el = e["el"]
            A(f" V={e['V']:>2} q={e['q']:>2}: rule (10) K={e['K']} local "
              f"{e['Zpol']:,} | best effort at N: local {el['Z']:,}, "
              f"{el['violating_personnel']:,} personnel in "
              f"{len(el['violations'])} provinces beyond M, longest "
              f"{el['maxdist']} km")
        A("")
        A("-" * 74)
        A(" PROPOSITION 1 CHECK")
        A("-" * 74)
        A(f" q >= V in every tested instance: "
          f"{all(v['q'] >= v['V'] for v in r['vreg'])}")
        broke = [v["V"] for v in r["vreg"] if not v["old_feasible"]]
        A(f" Rule K=max(N,V) infeasible from V={min(broke)} onward"
          if broke else " Rule K=max(N,V) feasible throughout")
        A("=" * 74)
        return "\n".join(L)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class App(_TkBase):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x780")
        self.minsize(980, 640)
        self.configure(bg=PAPER)
        self.data = None
        self.study = None
        self.msgq = queue.Queue()
        self.busy = False
        self._style()
        self._build()
        default = os.path.join(HERE, "data.json")
        if os.path.exists(default):
            self._load(default)
        self.after(120, self._pump)

    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=PAPER, foreground=INK,
                     font=("Segoe UI", 10))
        st.configure("TFrame", background=PAPER)
        st.configure("Side.TFrame", background=PANEL)
        st.configure("TLabel", background=PAPER)
        st.configure("Side.TLabel", background=PANEL)
        st.configure("H1.TLabel", font=("Segoe UI Semibold", 13))
        st.configure("H2.TLabel", font=("Segoe UI Semibold", 10),
                     background=PANEL, foreground=ACC)
        st.configure("Hint.TLabel", background=PANEL, foreground=MUTED,
                     font=("Segoe UI", 8))
        st.configure("Run.TButton", font=("Segoe UI Semibold", 10), padding=8)
        st.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))
        st.configure("Treeview", rowheight=22, font=("Consolas", 9))

    def _build(self):
        head = ttk.Frame(self, padding=(14, 10, 14, 6))
        head.pack(fill="x")
        ttk.Label(head, text="Examination Center Selection",
                  style="H1.TLabel").pack(side="left")
        ttk.Label(head, text="  risk-constrained covering model",
                  foreground=MUTED).pack(side="left", padx=(6, 0))
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        side = ttk.Frame(body, style="Side.TFrame", padding=12, width=290)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self._sidebar(side)
        self.nb = ttk.Notebook(body)
        self.nb.pack(side="right", fill="both", expand=True, padx=(0, 10),
                     pady=(0, 6))
        self._tab_summary()
        self._tab_tables()
        self._tab_figures()
        self._tab_solver()
        bar = ttk.Frame(self, padding=(14, 4, 14, 8))
        bar.pack(fill="x")
        self.status = ttk.Label(bar, text="Ready.", foreground=MUTED)
        self.status.pack(side="left")
        self.prog = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.prog.pack(side="right")

    def _sidebar(self, side):
        ttk.Label(side, text="DATA", style="H2.TLabel").pack(anchor="w")
        self.lbl_data = ttk.Label(side, text="not loaded", style="Side.TLabel",
                                  wraplength=250)
        self.lbl_data.pack(anchor="w", pady=(2, 4))
        ttk.Button(side, text="Load data.json ...",
                   command=self.on_load).pack(fill="x")
        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Label(side, text="PARAMETERS", style="H2.TLabel").pack(anchor="w")
        self.var_N = tk.IntVar(value=17)
        self.var_M = tk.IntVar(value=250)
        self.var_R = tk.DoubleVar(value=0.75)
        for text, var, values in (
                ("Center budget  N", self.var_N, list(range(8, 41))),
                ("Travel limit  M (km)", self.var_M,
                 [150, 175, 200, 225, 250, 275, 300, 350, 400]),
                ("Risk threshold  R", self.var_R,
                 [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])):
            row = ttk.Frame(side, style="Side.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=text, style="Side.TLabel").pack(side="left")
            ttk.Combobox(row, textvariable=var, values=values, width=7,
                         state="readonly").pack(side="right")
        ttk.Label(side, text="Defaults reproduce the published results.",
                  style="Hint.TLabel", wraplength=250).pack(anchor="w",
                                                            pady=(6, 0))
        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Label(side, text="CURRENT PLAN", style="H2.TLabel").pack(anchor="w")
        self.txt_plan = tk.Text(side, height=3, width=28,
                                font=("Consolas", 9), relief="solid",
                                borderwidth=1, wrap="word")
        self.txt_plan.insert("1.0", ", ".join(map(str, CURRENT_PLAN)))
        self.txt_plan.pack(fill="x", pady=(2, 0))
        ttk.Separator(side).pack(fill="x", pady=12)
        self.btn_run = ttk.Button(side, text="Run full study",
                                  style="Run.TButton", command=self.on_run)
        self.btn_run.pack(fill="x")
        ttk.Button(side, text="Export tables (CSV) ...",
                   command=self.on_export_tables).pack(fill="x", pady=(6, 0))
        ttk.Button(side, text="Export figures (PNG) ...",
                   command=self.on_export_figures).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="Save summary (TXT) ...",
                   command=self.on_export_summary).pack(fill="x", pady=(4, 0))

    def _tab_summary(self):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="Summary")
        self.txt = tk.Text(f, wrap="none", font=("Consolas", 9), relief="flat",
                           background="#fbfcfd")
        ys = ttk.Scrollbar(f, orient="vertical", command=self.txt.yview)
        xs = ttk.Scrollbar(f, orient="horizontal", command=self.txt.xview)
        self.txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        self.txt.insert("1.0", "Load data.json and press Run full study.")

    def _tab_tables(self):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="Tables")
        top = ttk.Frame(f)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Table:").pack(side="left")
        self.cb_table = ttk.Combobox(top, values=Study.TABLES, width=22,
                                     state="readonly")
        self.cb_table.current(1)
        self.cb_table.pack(side="left", padx=6)
        self.cb_table.bind("<<ComboboxSelected>>", lambda e: self.show_table())
        ttk.Button(top, text="Copy to clipboard",
                   command=self.on_copy_table).pack(side="right")
        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(wrap, show="headings")
        ys = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

    def _tab_figures(self):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="Figures")
        top = ttk.Frame(f)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Figure:").pack(side="left")
        self.cb_fig = ttk.Combobox(top, values=Study.FIGURES, width=34,
                                   state="readonly")
        self.cb_fig.current(1)
        self.cb_fig.pack(side="left", padx=6)
        self.cb_fig.bind("<<ComboboxSelected>>", lambda e: self.show_figure())
        ttk.Button(top, text="Save figure (600 dpi) ...",
                   command=self.on_save_figure).pack(side="right")
        self.figframe = ttk.Frame(f)
        self.figframe.pack(fill="both", expand=True)
        self.canvas = None

    def _tab_solver(self):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="What-if solver")
        ttk.Label(f, text="Force provinces open and re-solve. Enter province "
                          "codes separated by commas.",
                  wraplength=760, foreground=MUTED).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Mandatory provinces:").pack(side="left")
        self.var_forced = tk.StringVar(value="30, 62")
        ttk.Entry(row, textvariable=self.var_forced, width=44).pack(
            side="left", padx=6)
        ttk.Button(row, text="Solve", command=self.on_whatif).pack(side="left")
        self.txt_whatif = tk.Text(f, wrap="word", font=("Consolas", 9),
                                  relief="flat", background="#fbfcfd")
        self.txt_whatif.pack(fill="both", expand=True, pady=(8, 0))

    # -- actions ------------------------------------------------------------
    def _load(self, path):
        try:
            self.data = ProblemData(path)
        except Exception as exc:
            messagebox.showerror("Data error", str(exc))
            return
        self.study = Study(self.data)
        self.lbl_data.configure(
            text=f"{os.path.basename(path)}\n{self.data.n} provinces, "
                 f"{self.data.total:,} personnel")
        self._say(f"Loaded {self.data.n} provinces.")

    def on_load(self):
        p = filedialog.askopenfilename(title="Select data.json",
                                       filetypes=[("JSON", "*.json"),
                                                  ("All files", "*.*")])
        if p:
            self._load(p)

    def on_run(self):
        if self.data is None:
            messagebox.showwarning("No data", "Load data.json first.")
            return
        if self.busy:
            return
        try:
            plan = tuple(sorted({int(t) for t in
                                 self.txt_plan.get("1.0", "end")
                                 .replace("\n", " ").replace(";", ",")
                                 .split(",") if t.strip()}))
        except ValueError:
            messagebox.showerror("Current plan", "Enter integer codes.")
            return
        bad = [c for c in plan if c not in self.data.P]
        if bad:
            messagebox.showerror("Current plan", f"Unknown codes: {bad}")
            return
        self.busy = True
        self.btn_run.state(["disabled"])
        self.prog.start(12)
        N, M, R = self.var_N.get(), self.var_M.get(), float(self.var_R.get())

        def work():
            try:
                self.study.run(N=N, M=M, R=R, current_plan=plan,
                               progress=lambda s: self.msgq.put(("s", s)))
                self.msgq.put(("done", None))
            except Exception as exc:
                self.msgq.put(("err", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _pump(self):
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "s":
                    self._say(payload)
                elif kind == "err":
                    self._finish()
                    messagebox.showerror("Run failed", payload)
                elif kind == "done":
                    self._finish()
                    self.txt.delete("1.0", "end")
                    self.txt.insert("1.0", self.study.summary())
                    self.show_table()
                    self.show_figure()
                    self._say("Study complete.")
        except queue.Empty:
            pass
        self.after(120, self._pump)

    def _finish(self):
        self.busy = False
        self.prog.stop()
        self.btn_run.state(["!disabled"])

    def _say(self, msg):
        self.status.configure(text=msg)

    def _ready(self):
        if not (self.study and self.study.res):
            messagebox.showinfo("No results", "Run the study first.")
            return False
        return True

    def show_table(self):
        if not (self.study and self.study.res):
            return
        cols, rows = self.study.table(self.cb_table.get())
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=300 if "province" in c.lower() else
                             max(80, min(300, 9 * (len(c) + 6))),
                             anchor="center")
        for row in rows:
            self.tree.insert("", "end", values=row)

    def show_figure(self):
        if not (self.study and self.study.res):
            return
        fig = self.study.figure(self.cb_fig.get())
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.figframe)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def on_whatif(self):
        if not self._ready():
            return
        try:
            forced = tuple(sorted({int(t) for t in
                                   self.var_forced.get().replace(";", ",")
                                   .split(",") if t.strip()}))
        except ValueError:
            messagebox.showerror("Input", "Enter integer codes.")
            return
        bad = [c for c in forced if c not in self.data.P]
        if bad:
            messagebox.showerror("Input", f"Unknown codes: {bad}")
            return
        r, opt = self.study.res, self.study.opt
        res = opt.solve_policy(r["N"], r["M"], forced)
        old = opt.solve(max(r["N"], len(forced)), r["M"], forced,
                        lexicographic=False)
        base = r["s1"]
        out = [f"Mandatory set = {list(forced)}   (V = {len(forced)})", "",
               f"  q(M, I_H)          {res['q']}",
               f"  K = max(N, V, q)   {res['K']}   ({res['driver']})",
               f"  centers            {res['centres']}",
               f"  examined locally   {res['Z']:,} ({res['rate']:.2%})",
               f"  total travel       {res['pkm']:,} personnel-km",
               f"  longest journey    {res['maxdist']} km", "",
               f"  vs Scenario 1      {res['Z'] - base['Z']:+,} personnel "
               f"({100 * (res['rate'] - base['rate']):+.2f} pp)",
               f"  rule K = max(N,V)  "
               f"{'feasible' if old else 'INFEASIBLE - coverage unreachable'}"]
        self.txt_whatif.delete("1.0", "end")
        self.txt_whatif.insert("1.0", "\n".join(out))

    def on_copy_table(self):
        if not self._ready():
            return
        cols, rows = self.study.table(self.cb_table.get())
        self.clipboard_clear()
        self.clipboard_append("\t".join(cols) + "\n" +
                              "\n".join("\t".join(str(c) for c in r)
                                        for r in rows))
        self._say("Copied to clipboard.")

    def on_export_tables(self):
        if not self._ready():
            return
        folder = filedialog.askdirectory(title="Folder for CSV files")
        if not folder:
            return
        import csv
        for name in Study.TABLES:
            cols, rows = self.study.table(name)
            with open(os.path.join(folder, name.replace(" ", "_") + ".csv"),
                      "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                w.writerows(rows)
        messagebox.showinfo("Export", f"Tables written to\n{folder}")

    def on_export_figures(self):
        if not self._ready():
            return
        folder = filedialog.askdirectory(title="Folder for PNG files")
        if not folder:
            return
        for k, name in enumerate(Study.FIGURES, 1):
            self.study.figure(name, fig=Figure()).savefig(
                os.path.join(folder, f"figure{k}.png"), dpi=600,
                bbox_inches="tight")
        messagebox.showinfo("Export", f"Figures written to\n{folder}")

    def on_save_figure(self):
        if not self._ready():
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.study.figure(self.cb_fig.get(), fig=Figure()).savefig(
                p, dpi=600, bbox_inches="tight")
            self._say(f"Saved to {p}")

    def on_export_summary(self):
        if not self._ready():
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Text", "*.txt")])
        if p:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self.study.summary())
            self._say(f"Saved to {p}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--cli", action="store_true",
                    help="run the study and print the summary without the GUI")
    ap.add_argument("--data", default=os.path.join(HERE, "data.json"))
    args = ap.parse_args()

    if args.cli:
        study = Study(ProblemData(args.data))
        study.run(progress=lambda s: print(" ..", s, flush=True))
        print()
        print(study.summary())
        return

    if tk is None:
        sys.exit("Tkinter is not installed. Install python3-tk for the GUI, "
                 "or use --cli for text output.")
    try:
        App().mainloop()
    except tk.TclError as exc:
        print("A graphical display is required. Use --cli for text output.")
        print(f"Tk reported: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
