# Intern Challenge Placement Submission

This fork contains a complete solution for the VLSI cell placement challenge.

## Result

Measured on the first 10 benchmark cases with `python3 test.py`:

| Name | Average Overlap | Average Wirelength | Total Runtime | Notes |
|------|-----------------|--------------------|---------------|-------|
| Omar Ramadan | 0.0000 | 0.2507 | 858.10s | Gated local-search portfolio; runtime not optimized |

Exact local run:

```text
AVERAGE_OVERLAP=0.000000
AVERAGE_WIRELENGTH=0.250709
TOTAL_REPORTED_RUNTIME=858.10s
```

## How To Run

Install dependencies:

```bash
pip install torch numpy scipy matplotlib
```

Optional proof checker dependency:

```bash
pip install z3-solver
```

Run the required first 10 tests:

```bash
python3 test.py
```

Extra-credit cases 11 and 12 are kept in `EXTRA_CREDIT_TEST_CASES` inside
`test.py`, but they are not run by default because the challenge submission asks
for the first 10 benchmark cases.

## Implementation Notes

The public challenge API remains in root-level `placement.py`, so the provided
`test.py` can still import the expected functions. The implementation lives in
the `solver/` package, split by algorithm stage:

1. `solver/core/`
   - Challenge constants and tensor feature indices
   - Synthetic netlist generation
   - Wirelength and differentiable overlap losses
   - Exact overlap and normalized leaderboard metrics

2. `solver/gradient/`
   - Main differentiable optimizer
   - Momentum SGD placement with overlap pressure
   - Short post-legalization wirelength descent

3. `solver/local_search/`
   - Legal single-cell projections
   - Same-size assignment refinement
   - Pairwise swaps that preserve legality
   - Shared local wirelength helpers

4. `solver/unlock/`
   - Temporary-overlap window optimization
   - Window removal and greedy legal reinsertion
   - Used to escape legal local minima where cells cannot pass through each
     other with purely legal moves

5. `solver/macro/`
   - `layouts.py`: macro contact-layout generation and topology populations
   - `relegalize.py`: macro port-aware standard-cell reinsertion
   - `search.py`: global macro topology and continuous macro-coordinate search

6. `solver/pipeline/`
   - Size-aware candidate portfolio
   - Full-pipeline selection: every candidate is trained, legalized, refined,
     and only then compared by the official metrics

7. `placement.py`
   - Thin compatibility wrapper exposing the expected challenge functions
   - Small demo `main()` for manual visualization runs

8. `maxima_proof/`
   - Pairwise lower-bound certificate for proof-style analysis
   - Sparse LP tangent lower-bound diagnostic
   - MILP branch-and-bound bounded-domain verifier
   - Unit tests for the certificate math and benchmark integration
   - Optional current-solver upper-bound computation for selected cases

The solver keeps overlap as the hard priority and then improves wirelength with
a legal full-pipeline portfolio:

- differentiable overlap repulsion for the main optimizer
- KD-tree overlap candidate pruning for large cases
- legal post-placement cleanup using assignment and pairwise swaps
- overlap-tolerant local unlock/refinement windows
- macro-aware topology and continuous macro-coordinate refinement
- outer portfolio selection after every candidate has gone through the full
  final refinement pipeline

I also fixed the baseline placeholder overlap loss and kept the public metric
calculation shared between training-time selection and final evaluation.

## Proof Tools

The repo now includes `maxima_proof/` next to `solver/`. Despite the name, the
main certificate is a lower-bound proof tool because the benchmark minimizes
wirelength.

Run proof tests:

```bash
python3 -m maxima_proof.test_lower_bound_certificate
python3 -m maxima_proof.test_branch_and_bound_verifier
python3 -m maxima_proof.test_lp_tangent_certificate
python3 -m maxima_proof.test_milp_branch_verifier
python3 -m maxima_proof.test_z3_milp_certificate
```

Run lower-bound certificates for the first 10 cases:

```bash
python3 -m maxima_proof.lower_bound_certificate
```

Optionally compute current legal upper bounds with the solver for selected
cases. This can be slow because it runs placement:

```bash
python3 -m maxima_proof.lower_bound_certificate --cases 1 --compute-upper-bound
```

The default proof tool does not prove global optimality. It proves a valid
mathematical floor from an exact edge-independent relaxation. The submitted
legal placement is the upper bound; the certificate lower bound is the floor. A
smaller gap means a stronger near-optimality argument.

The latest rigorous all-ten lower-bound average is:

```text
Average lower bound: 0.077535
```

There is also a tighter `--mode bundled-estimate`, but that mode is documented
as a numerical diagnostic rather than a formal certificate.

Run the sparse LP tangent diagnostic across all ten cases:

```bash
python3 -m maxima_proof.lp_tangent_certificate --radial-levels 4 --max-diff-edges 3000
```

This keeps shared x/y positions and uses tangent-plane lower estimators for the
wirelength objective. Latest diagnostic result:

```text
Average successful LP lower bound: 0.153127
```

Cases 1-9 use all different-cell edges. Case 10 uses a conservative 3000-edge
subset to keep the LP tractable; omitted edge terms are nonnegative, so dropping
them preserves lower-bound direction.

Run the MILP branch-and-bound verifier in all-case capped mode:

```bash
python3 -m maxima_proof.milp_branch_verifier --radial-levels 3 --max-pairs 600 --max-diff-edges 3000 --position-bound 500 --time-limit 10
```

Latest capped result:

```text
Average MILP lower bound: 0.153193
```

This MILP uses binary side choices for non-overlap and HiGHS branch-and-bound.
The reported value is a lower bound for the bounded tangent-relaxation model
that was actually solved. It now also supports `--coordinate-upper-bound`, which
derives a conservative coordinate box from an incumbent score when the cell
connectivity graph is connected.

The strongest tight-box public-case run so far is case 1:

```text
MILP lower bound: 0.333241
Full bounded model: True
LP fallback used: False
Coordinate box certified: False
```

That run uses a small working coordinate box, so it is a strong exploratory
bounded-model result. The safer case-1 run uses the incumbent-derived coordinate
certificate:

```bash
python3 -m maxima_proof.milp_branch_verifier --cases 1 --radial-levels 3 --max-pairs -1 --max-diff-edges -1 --edge-specific-side-tangents --position-bound 12600 --coordinate-upper-bound 0.333636141 --time-limit 60
```

```text
MILP lower bound: 0.302605
Full bounded model: True
Coordinate box certified: True
```

This fixes the coordinate-box issue for the case-1 proof claim. Remaining
limitations for full case 1 are the tangent relaxation and the use of
floating-point HiGHS output rather than a full independently checkable MILP
branch certificate.

For small and capped MILP models, the repo now includes an independent exact Z3
checker:

```bash
python3 -m maxima_proof.z3_milp_certificate --case 1 --target-bound 0.05 --radial-levels 2 --max-pairs 20 --max-diff-edges 20 --position-bound 500 --timeout-ms 10000
```

Latest capped exact check:

```text
Status: unsat
Variables: 144
Binary variables: 80
Constraints: 440
```

That closes the floating-point certificate issue for the rationalized capped
model being checked. It does not certify the full case-1 MILP because that model
is much larger than the current exact checker can exhaust quickly.

### Interview Framing

The proof tools are included as validation and engineering context, not as the
main scoring claim. The submitted result is still the heuristic placement solver
with zero overlap and low wirelength. The proof folder shows how I investigated
the gap between a strong empirical solution and a certifiable optimum:

- rigorous lower bounds for the original benchmark objective
- stronger LP/MILP relaxation diagnostics
- coordinate-box certification for case 1
- exact Z3 checks for small/capped rationalized MILP models

The important takeaway is not "this proves global optimality." It does not. The
takeaway is that the repo separates what is measured, what is proven, and what
would require stronger optimization tooling.
