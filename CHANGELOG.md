# Changelog

## 2.1.0 (2026-10-19), revision for IEEE Access

Added
- Capacity constraint (16) and capacitated covering minimum `q_C`
  (`cap` argument of `Optimizer.solve`, `covering_minimum`, `solve_policy`).
- Best-effort model (17)–(19) for a hard center budget (`Optimizer.elastic`).
- Benchmarks under mandatory inclusion (Table X): mandatory set passed to the
  p-median and MCLP, sequential insertion (`Optimizer.sequential`) and the
  risk-blind optimum (`Optimizer.risk_blind`).
- LP-relaxation bound (`Optimizer.lp_bound`) and coverage-check counter for the
  heuristic.
- Capacity sweep (Table XI) and best-effort plans (Table XII) in the study.
- `--cli` runs without Tk installed.

Changed
- Table numbering follows the revised article (Tables II–XIII).
- Benchmarks report total travel and the longest journey for every method,
  with each province assigned to its nearest open center.
- Figure code reproduces the annotations of the published figures; Figure 1
  shows the new modules and the scenario feedback loop.
- PuLP pinned to 2.x.

Fixed
- The greedy heuristic did not enforce |C| = K when its coverage-repair phase
  opened more than K centers. It now closes surplus non-mandatory centers when
  coverage allows and otherwise reports failure. The V = 0 result (18,191) is
  unchanged.
- The heuristic's coverage test uses precomputed reachable sets (faster, same
  result).
