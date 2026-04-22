# Maxima Proof Tools

This folder contains proof-style tooling for the placement challenge.

The name comes from the optimization-landscape language used in the writeup, but
the certificate itself is a **lower-bound certificate**. The placement benchmark
is a minimization problem: lower wirelength is better. For this kind of problem,
a proof needs two sides:

- a legal placement from the solver, which gives a measured upper bound
- a mathematical relaxation, which gives a lower bound no legal solution can beat

If the legal score and the lower bound are close, that is evidence of
near-optimality. If they are far apart, the solver may still be strong, but the
proof is weak.

## What This Proves

`lower_bound_certificate.py` defaults to a rigorous edge-independent relaxation:

- same-cell pin edges are counted exactly because translating a cell moves both
  pins together
- each different-cell wire chooses its own best legal relative separation
- global consistency between cells is dropped, which makes the problem easier
  and therefore gives a valid lower bound

That independence makes the result optimistic, so it is a valid lower bound.
It does **not** prove global optimality.

The script also has `--mode bundled-estimate`, which groups all wires between
the same pair of cells and solves a small convex problem numerically. That mode
is useful diagnostics and is usually tighter, but it is not the default formal
certificate because a numerical optimizer is not a proof by itself.

## Run Lower Bounds Only

```bash
python3 -m maxima_proof.lower_bound_certificate
```

Latest rigorous all-ten result:

```text
Average lower bound: 0.077535
```

Run the tighter numerical diagnostic:

```bash
python3 -m maxima_proof.lower_bound_certificate --mode bundled-estimate
```

## Sparse LP Tangent Diagnostic

`lp_tangent_certificate.py` keeps one shared x/y position for every cell and
underestimates the smooth wirelength objective with tangent planes. This is
closer to the real global problem than the edge-independent certificate.

It is still not a machine-checkable proof because SciPy/HiGHS solves the LP in
floating point. The tangent planes are mathematically valid lower estimators,
and the script reports solver residuals, but the result should be described as a
strong diagnostic lower bound.

Run all ten cases. Cases 1-9 use all different-cell edges; case 10 uses a
conservative 3000-edge subset so it finishes quickly:

```bash
python3 -m maxima_proof.lp_tangent_certificate --radial-levels 4 --max-diff-edges 3000
```

Latest result:

```text
Average successful LP lower bound: 0.153127
```

## Run Lower Bounds Plus Current Solver Upper Bounds

This can be slow because it calls the full placement solver.

```bash
python3 -m maxima_proof.lower_bound_certificate --compute-upper-bound
```

You can restrict the cases:

```bash
python3 -m maxima_proof.lower_bound_certificate --cases 1 2 3
```

## Run Proof Tests

```bash
python3 -m maxima_proof.test_lower_bound_certificate
python3 -m maxima_proof.test_branch_and_bound_verifier
python3 -m maxima_proof.test_lp_tangent_certificate
python3 -m maxima_proof.test_milp_branch_verifier
python3 -m maxima_proof.test_z3_milp_certificate
```

## Experimental Global Verifier

`branch_and_bound_verifier.py` is the architecture needed for a global
optimality proof. It keeps shared x/y positions in the proof tree and branches
on the four non-overlap choices for a cell pair: left, right, below, or above.
Its default lower bound is exact and conservative: each edge is minimized
independently under the branch constraints that directly apply to its endpoint
pair. No numerical convex solve is used as a formal pruning certificate.

Run a tiny two-cell proof demo:

```bash
python3 -m maxima_proof.branch_and_bound_verifier --demo
```

Try benchmark case 1 with a bounded node budget:

```bash
python3 -m maxima_proof.branch_and_bound_verifier --case 1 --node-limit 200 --time-limit 90
```

Current audit result: the tiny demo closes, but benchmark case 1 does not yet
produce a rigorous proof. With 200 branch nodes, the rigorous lower bound rises
to `0.217704465`, but the current legal upper bound is `0.333636141`, leaving an
open gap of `0.115931676`.

## MILP Branch-And-Bound Verifier

`milp_branch_verifier.py` is the more direct branch-and-bound formulation. It
uses SciPy's HiGHS MILP backend:

- bounded x/y variables for every cell
- four binary side choices for every selected cell pair
- tangent-plane lower estimators for smooth wirelength
- optional edge-specific side tangent cuts for tighter per-edge lower bounds
- optional incumbent-derived coordinate-box certificates
- the MILP dual bound as the reported lower bound

This is the right shape for a solver-backed certificate, but it has an important
scope condition: big-M non-overlap requires a finite coordinate box. The result
is therefore a bounded-domain certificate unless a separate argument proves the
global optimum must lie inside that box. The `--coordinate-upper-bound` option
now supplies that argument when the different-cell graph is connected: the
incumbent score bounds every edge displacement, and shortest paths from the
anchor cell give a conservative coordinate box.

Run the all-case capped diagnostic:

```bash
python3 -m maxima_proof.milp_branch_verifier --radial-levels 3 --max-pairs 600 --max-diff-edges 3000 --position-bound 500 --time-limit 10
```

Latest all-case capped result:

```text
Average MILP lower bound: 0.153193
```

For cases 1-3 the capped command includes every pair and every different-cell
edge. Larger cases are explicit relaxations because pair constraints are capped.

Run the stronger full case-1 tight-box model:

```bash
python3 -m maxima_proof.milp_branch_verifier --cases 1 --radial-levels 3 --max-pairs -1 --max-diff-edges -1 --edge-side-tangent-limit 80 --position-bound 500 --time-limit 180 --mip-rel-gap 1e-6
```

Latest case-1 result:

```text
Lower bound: 0.333241
Full bounded model: True
LP fallback used: False
Coordinate box certified: False
```

That is close to the current legal upper bound, but it uses a working coordinate
box that is not certified from the incumbent.

Run the safer case-1 coordinate-certified model:

```bash
python3 -m maxima_proof.milp_branch_verifier --cases 1 --radial-levels 3 --max-pairs -1 --max-diff-edges -1 --edge-specific-side-tangents --position-bound 12600 --coordinate-upper-bound 0.333636141 --time-limit 60 --mip-rel-gap 1e-4
```

Latest coordinate-certified case-1 result:

```text
Lower bound: 0.302605
Full bounded model: True
Coordinate box certified: True
```

This is the cleaner proof claim for case 1: given the incumbent upper bound and
connected different-cell graph, the coordinate box is large enough to contain an
optimal solution after translation. The result remains a tangent-relaxation
lower bound from a floating-point MILP solve.

To make an exact global claim, this folder still needs two more certificates:

1. an objective certificate proving the tangent-plane model is tight enough for
   the smooth wirelength objective, or a solver that handles the smooth objective
   directly with certified bounds
2. a solver certificate that can be independently checked instead of relying
   only on floating-point HiGHS output

## Exact Z3 Checker For Small/Capped MILPs

`z3_milp_certificate.py` is an optional exact checker for rationalized MILP
models. It rebuilds the linear MILP, converts every floating coefficient to an
exact decimal rational, and asks Z3 whether a modeled placement exists below a
target bound.

Install the optional dependency:

```bash
pip install z3-solver
```

Run a capped case-1 exact check:

```bash
python3 -m maxima_proof.z3_milp_certificate --case 1 --target-bound 0.05 --radial-levels 2 --max-pairs 20 --max-diff-edges 20 --position-bound 500 --timeout-ms 10000
```

Observed result:

```text
Status: unsat
Variables: 144
Binary variables: 80
Constraints: 440
```

This closes the floating-point solver issue for the rationalized capped model
being checked. It does not close the full case-1 MILP, because the full model is
much larger and Z3 did not replace HiGHS as the production branch-and-bound
engine.
