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
