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
reported in the article, and regenerates all nine tables and all seven figures.

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

---

## Installation

```bash
git clone https://github.com/cemilsimsekmail-rgb/ecs-dss.git
cd ecs-dss
pip install -r requirements.txt
```

On Debian or Ubuntu the Tk bindings are a separate package:

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
| Tables | Tables 1–9 and the full 81-province risk table, exportable as CSV |
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

Solving takes well under a second per instance with CBC; the full study,
including the 1,500-draw robustness analysis, takes a few minutes.

---

## Structure

```
ecs_dss.py          single-file application
  ProblemData       data loading and validation
  RiskModel         TOPSIS risk scores under the policy weights
  Optimizer         covering minimum, main model, diagnostics, benchmarks
  Study             full experimental protocol, tables, figures, summary
  App               Tkinter interface
data.json           personnel counts and distance matrix
```

## License

MIT. See `LICENSE`.
