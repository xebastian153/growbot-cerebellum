# Conventions

The standard this repository holds itself to. Contributions are welcome under the
same rules.

## What the documentation says

1. **Every claim carries its number, every number carries its conditions.** Seeds,
   horizons, sample sizes. A number whose conditions are unstated is not reported.
2. **Negative results get the same prominence as positive ones**, labelled as
   negative in the heading. A flat result that closes a question is a result.
3. **Both horizons, always.** 100 ms and 500 ms — a gap invisible at one shows at
   the other, and this repo has been caught by that once.
4. **Comparisons share seeds.** Runs collected under different seeds are never
   compared; the 500 ms metric varies several points across collections.
5. **Corrections are recorded, not erased.** When an earlier number was wrong, the
   text says what changed and why.
6. **Every documented number is reproducible by a listed command.** `results/*.json`
   is the machine-readable source; prose quotes round numbers from it.
7. **Limits live next to the claim.** "Sim-only", "same model class", "n=60" go in
   the same sentence as the result, not in a distant caveats section.

## What the documentation never says

1. **No narrated conversations, no attribution to community members.** Cite what is
   citable: file paths, repositories, published papers, public talks, issue and PR
   links. Ideas from discussion enter as questions the experiments answer.
2. **No dates tied to conversations.** Version-control history carries time.
3. **No roadmap promises.** The repository documents what exists and runs today.
4. **No marketing adjectives.** The number is the adjective.
5. **No aggregates that hide regime differences.** When behaviour differs by regime
   (calm / fast / fallen), the split is reported, not the average alone.

## Code and repository hygiene

- Every experiment is one script: a docstring stating the question it answers, a
  working `--help`, output to `results/<name>.json`, run log in `results/logs/`.
- Deviations from a published source are gated behind explicit flags, never silent
  edits, so original and modified variants compare under identical seeds.
- Two implementations of the same math get an equivalence test
  (`forward-model/test_forward.mjs` is the standard: float32 tolerance, and it has
  caught a real convention bug).
- Parsers are validated round-trip: synthetic data generated in the target format
  must preserve, through the real code path, the signal the parser exists to carry.
- Nothing tracked that a listed command regenerates (`data/*.npz`).
- Commits are work-units with conventional prefixes; the message states the finding
  with its numbers.
- Third-party files are listed in `NOTICE` with their licence.
