# Proof Audit

## Verdict

The proof idea is sound, but only in the exact relaxation mode.

The original bundled pair calculation was mathematically valid in principle if
each convex subproblem was solved exactly. The implementation used SciPy, so it
was a numerical estimate rather than a formal proof certificate. A numerical
optimizer can return a value above the true relaxed minimum; that would be
unsafe to report as a lower bound.

The current default mode fixes this by using an exact edge-independent
relaxation.

## Rigorous Statement

For every legal placement:

- same-cell wire costs are counted exactly, because moving the cell translates
  both endpoints together
- each different-cell wire has a relative displacement that satisfies the
  two-cell non-overlap rule
- the exact minimum cost for that one wire under that two-cell rule is no larger
  than the wire's cost in the legal placement
- summing those per-wire minima gives a value no legal placement can beat

After dividing by the same positive normalizer used by the benchmark, the value
is a valid normalized lower bound.

## What Is Not Proven

This does not prove global optimality. It drops too much structure:

- different wires between the same two cells can choose different separations
- different cell pairs do not need to agree on one global cell position
- no chip-wide packing or region capacity constraints are enforced

The proof is sound but loose.

## Global Verifier Attempt

I added `branch_and_bound_verifier.py` as the next proof step. It is the right
shape for a global proof because it branches on the actual non-overlap choices:

- cell A left of cell B
- cell A right of cell B
- cell A below cell B
- cell A above cell B

The verifier's default branch bound is exact and conservative. For each branch,
it minimizes every edge independently under the branch constraints that directly
apply to that edge's endpoint pair. This loses global consistency, so it is
loose, but every pruning decision is mathematically safe.

This verifier can prove a tiny two-cell demo. That shows the architecture is
working in a case small enough for the edge-independent bound to close.

It does not yet prove benchmark case 1. A bounded case 1 run with 200 branch
nodes produced this result:

```text
Status: open
Upper bound: 0.333636141
Global lower bound: 0.217704465
Gap: 0.115931676
Nodes solved: 201
Open nodes: 151
```

This is rigorous but still open. The proof tree improved the lower bound beyond
the root edge-independent floor, but it did not close the gap to the legal
placement.

## Current Evidence

Commands run:

```bash
python3 -m maxima_proof.test_lower_bound_certificate
python3 -m maxima_proof.test_branch_and_bound_verifier
python3 -m maxima_proof.test_lp_tangent_certificate
python3 -m maxima_proof.test_milp_branch_verifier
python3 -m maxima_proof.lower_bound_certificate
python3 -m maxima_proof.lower_bound_certificate --cases 1 2 3 --mode bundled-estimate
python3 -m maxima_proof.branch_and_bound_verifier --demo
python3 -m maxima_proof.branch_and_bound_verifier --case 1 --node-limit 200 --time-limit 90
```

Observed results:

```text
Lower-bound unit tests: 7 passed
Branch verifier tests: 2 passed
LP tangent tests: 2 passed
MILP branch verifier tests: 2 passed
Rigorous average lower bound: 0.077535
Bundled estimate average for cases 1-3: 0.252743
Tiny branch-and-bound demo: proven
Case 1 branch-and-bound attempt: open, lower bound 0.217704465
```

The bundled estimate is tighter, but it is not the formal certificate.

## Sparse LP Tangent Diagnostic

I added `lp_tangent_certificate.py` as a stronger lower-bound diagnostic. It
keeps shared x/y cell positions and adds tangent-plane lower estimators for the
smooth wirelength objective. This is closer to the true global placement problem
than the exact edge-independent bound.

The tangent planes are mathematically valid because the smooth wirelength term
is convex. The limitation is the LP solve: SciPy/HiGHS is floating point, so the
number is residual-audited rather than machine-checkable.

Command run:

```bash
python3 -m maxima_proof.lp_tangent_certificate --radial-levels 4 --max-diff-edges 3000
```

Observed all-case diagnostic:

```text
Cases 1-9: full LP over all different-cell edges
Case 10: conservative LP over 3000 selected different-cell edges
Average successful LP lower bound: 0.153127
Maximum reported constraint violation: 9.95e-14
```

This is much stronger than the exact edge-independent average of `0.077535`,
but it remains a diagnostic rather than a formal exact-arithmetic proof.

## MILP Branch-And-Bound Verifier

I added `milp_branch_verifier.py` to model the branch-and-bound proof in the
standard MILP form:

- one bounded x/y coordinate pair per cell
- four binary side choices for selected cell pairs
- one cost variable per selected different-cell edge
- tangent-plane lower bounds for the smooth wirelength function
- optional edge-specific side tangent cuts
- an optional incumbent-derived coordinate-box certificate
- SciPy/HiGHS branch-and-bound over the side-choice binaries

The proof scope is bounded-domain because the big-M constraints need finite
coordinate limits. If every pair and every different-cell edge is included, the
model is a full bounded-domain tangent-relaxation verifier. If caps are used,
the model drops some non-overlap constraints or edge terms and becomes a safe
but weaker relaxation.

All-case capped command:

```bash
python3 -m maxima_proof.milp_branch_verifier --radial-levels 3 --max-pairs 600 --max-diff-edges 3000 --edge-side-tangent-limit 0 --position-bound 500 --time-limit 10 --mip-rel-gap 1e-4
```

Observed all-case result:

```text
Average MILP lower bound: 0.153193
Cases 1-3: full bounded model, but the short run used LP-relaxation fallback
Cases 4-10: capped relaxation
```

Stronger case-1 command:

```bash
python3 -m maxima_proof.milp_branch_verifier --cases 1 --radial-levels 3 --max-pairs -1 --max-diff-edges -1 --edge-side-tangent-limit 80 --position-bound 500 --time-limit 180 --mip-rel-gap 1e-6
```

Observed case-1 result:

```text
Lower bound: 0.333241
Primal bound inside the MILP relaxation: 0.337197
Full bounded model: True
LP relaxation fallback: False
Coordinate box certified: False
Nodes: 6880
Runtime: 180.18s
```

This is the strongest proof-style result so far for a public benchmark case. The
exact claim is:

```text
No placement inside the selected coordinate box can beat 0.333241 in the
piecewise-linear tangent relaxation solved by the MILP, up to the reported
solver tolerance.
```

That is useful, but this run used a tight working coordinate box, not a certified
coordinate box.

Coordinate-certified case-1 command:

```bash
python3 -m maxima_proof.milp_branch_verifier --cases 1 --radial-levels 3 --max-pairs -1 --max-diff-edges -1 --edge-specific-side-tangents --position-bound 12600 --coordinate-upper-bound 0.333636141 --time-limit 60 --mip-rel-gap 1e-4
```

Observed coordinate-certified case-1 result:

```text
Lower bound: 0.302605
Full bounded model: True
Coordinate box certified: True
```

The coordinate certificate works because the different-cell graph for case 1 is
connected. The incumbent upper bound gives a raw total wirelength budget. Since
every edge cost is nonnegative, each individual connected edge must fit inside
that budget in any solution no worse than the incumbent. Shortest paths from the
anchor cell then give a conservative box that contains some translated optimum.

This fixes the coordinate-box flaw for the case-1 lower-bound claim. It is still
not exact global optimality for the original smooth placement problem. To make
that stronger claim, the proof needs two more independently checkable pieces:

- an objective certificate proving the tangent-plane model is tight enough for
  the smooth wirelength objective, or a certified solver for the smooth
  objective directly
- a solver certificate that can be checked after the run instead of relying only
  on floating-point HiGHS output

## Exact Z3 Checker

I added `z3_milp_certificate.py` as an independent exact checker for small or
capped MILP models. It is not a faster optimizer. It is a certificate checker:
it rebuilds the linear model, converts floating coefficients to exact decimal
rationals, and asks Z3 whether a solution exists below a target bound.

Command run:

```bash
python3 -m maxima_proof.z3_milp_certificate --case 1 --target-bound 0.05 --radial-levels 2 --max-pairs 20 --max-diff-edges 20 --position-bound 500 --timeout-ms 10000
```

Observed result:

```text
Status: unsat
Variables: 144
Binary variables: 80
Constraints: 440
Runtime: 0.08s
```

This closes the floating-point certificate issue for that rationalized capped
MILP: Z3 independently proves there is no modeled placement below the target.

The full case-1 MILP is much larger. The exact checker is now present, tested,
and useful for capped models, but it has not replaced HiGHS as a full
branch-and-bound certificate for the largest proof run.
