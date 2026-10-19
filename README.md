# ECS-DSS — Examination Center Selection Decision Support System

Companion software for:

> C. Şimşek, "A Decision Support System for Risk-Constrained Examination Center
> Selection Under a Coverage Requirement," *IEEE Access*, 2026.

A security agency deploying personnel across all 81 provinces of Türkiye holds
its professional examinations at a limited number of centers. This application
selects those centers with an integer linear program in which high security
risk **compels** the inclusion of a province rather than its avoidance, and in
which the number of centers is governed by a coverage requirement rather than
by the budget alone.

The application solves the model on the institutional data, runs every analysis
reported in the article, and regenerates every computed table (Tables II–XIII)
and all seven figures. Table I of the article is a literature comparison and is
not computed.

---

## Model

```
max  Z = Σ_i P_i y_ii                                        (1)
s.t. Σ_{j ∈ N_i(M)} y_ij = 1              ∀ i                (2)
     y_ij ≤ x_j                           ∀ i, j             (3)
     y_ij = 0                             if d_ij > M         (4)
     x_j  = 1                             ∀ j ∈ I_H          (5)
     Σ_j x_j = K                                             (6)
     x, y ∈ {0,1}                                            (7)
```

Constraints (2) and (4) impose complete coverage of every province within the
travel radius, so the feasible center sets form a covering family. The
cardinality is

```
q(M, I_H) = minimum cover of I at radius M with I_H already open       (9)
K         = max{ N, V, q(M, I_H) }                                    (10)
```

**Proposition 1.** Since every province of `I_H` is open in any feasible
solution of (9), `q ≥ V` always holds. Hence `K = max{N, q}` and the mandatory
count `V` never binds on its own. A rule of the form `K = max(N, V)` becomes
infeasible once the mandatory set grows and clusters geographically.

### Solution procedure and extensions (Section III-G)

```
(15) min Σ_i Σ_j P_i d_ij y_ij   s.t. (2)–(7), Σ_i P_i y_ii = Z*     travel stage
(16) Σ_i P_i y_ij ≤ C_j x_j                                          capacity
(17) Σ_j y_ij = 1, y_ij ≤ x_j  (all pairs)                           elastic assignment
(18) u_i ≥ Σ_j (d_ij − M)^+ y_ij,  u_i ≥ 0                           penalty variables
(19) max Σ_i P_i y_ii − λ Σ_i P_i u_i,  λ > Σ_i P_i                  best-effort objective
```

* `Optimizer.solve(K, M, forced, cap=...)` adds the capacity constraint (16);
  `cap` is a number (uniform) or a dict {province: capacity}.
  `Optimizer.covering_minimum(M, forced, cap=...)` returns the capacitated
  covering minimum `q_C`, and `solve_policy` applies `K = max{N, q_C}`.
* `Optimizer.elastic(K, M, forced)` returns the least-violating plan with
  exactly `K` centers. (19) is solved as the equivalent lexicographic sequence
  (excess personnel-km, then home cohort, then travel).

---

## Installation

```bash
git clone https://github.com/cemilsimsekmail-rgb/ecs-dss.git
cd ecs-dss
pip install -r requirements.txt
```

PuLP is pinned to the 2.x series, with which the published results were
produced (the bundled CBC 2.10.3 solver is used).

The graphical interface needs Tk. On Debian or Ubuntu the Tk bindings are a
separate package (not needed for `--cli`):

```bash
sudo apt install python3-tk
```

## Usage

Graphical interface:

```bash
python ecs_dss.py
```

Text output, no display required:

```bash
python ecs_dss.py --cli
```

`data.json` must sit next to the script; it is loaded automatically at start-up.

### Interface

| Tab | Contents |
|---|---|
| Summary | Every numerical result, including the Proposition 1 check |
| Tables | Tables II–XIII and the full 81-province risk table, exportable as CSV |
| Figures | Figures 1–7, exportable at 600 dpi |
| What-if solver | Force any set of provinces open and re-solve |

The centre budget `N`, the travel limit `M`, the risk threshold `R`, and the
current institutional plan can all be changed from the sidebar. The default
values reproduce the results published in the article.

---

## Data

`data.json` holds two objects:

```json
{
  "personnel": {"1": 458, "2": 373, ...},
  "distance":  {"1_2": 339, "1_3": 517, ...}
}
```

`personnel` gives the number of examination candidates resident in each
province and `distance` the shortest highway distance in kilometres between
each ordered pair. Provinces are identified by their official vehicle
registration codes. The distance matrix is validated at load time for
completeness, symmetry, and a zero diagonal.

Both are real institutional data: 81 provinces and 45,264 personnel.

### Confidential inputs

The five provincial security indicators that feed the risk scores cannot be
released. `RiskModel.build_matrix` regenerates a synthetic indicator set from a
fixed seed, within the operational ranges of the real indicators, so that every
published number is reproducible. To work with real indicators, replace that
method with a loader; nothing downstream depends on how the matrix is produced.

The criterion weights in `RiskModel.WEIGHTS` are policy parameters supplied by
the agency. Section V-G of the article shows that the mandatory set is
insensitive to them: across 1,500 random perturbations and 20 one-at-a-time
scalings the set `{30, 62}` survived in at least 95.8% of cases, and in all 20
of the one-at-a-time perturbations.

---

## Reproducing the published results

```bash
python ecs_dss.py --cli > results.txt
```

Expected headline figures:

| Quantity | Value |
|---|---|
| Scenario 1 | 19,574 examined locally (43.24%), 4,242,518 personnel-km |
| Scenario 2 | 18,916 examined locally (41.79%), 4,232,145 personnel-km |
| Current institutional plan | 17,991 (39.75%), 4,541,198 personnel-km |
| Same-day rule violation, current plan | 1,488 personnel in 5 provinces, longest 314 km |
| Coverage-critical centers | 14 of 17; redundant {12, 47, 73} |
| Rule `K = max(N, V)` | infeasible from V = 10 onward |
| Benchmark, V = 2 (Table X) | proposed 18,916; p-median 16,403; MCLP 10,854; greedy 17,517 |
| Sequential insertion (Table X) | no feasible plan with K centers for V = 5, 8, 10 |
| Capacity C = 3,500 (Table XI) | Scenario 1: 19,574 local; Scenario 2: 18,435 local |
| Best effort, V = 10, N = 17 (Table XII) | 215 personnel of province 57 at 254 km |

Each instance of the main model closes at the root node in well under a
second; the full study, including the 1,500-draw robustness analysis, the
benchmarks, the capacity sweep and the best-effort plans, takes about a minute.
CPU times vary slightly between machines and runs.

---

## Structure

```
ecs_dss.py          single-file application
  ProblemData       data loading and validation
  RiskModel         TOPSIS risk scores under the policy weights
  Optimizer         covering minimum, main model, travel stage, capacity,
                    best-effort model, diagnostics, benchmarks
  Study             full experimental protocol, tables, figures, summary
  App               Tkinter interface
data.json           personnel counts and distance matrix
```

## License

MIT. See `LICENSE`.
