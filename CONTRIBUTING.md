# Contributing to ANVIL

ANVIL is a harness that measures whether an agent got better. Almost every rule
below exists because a change that looked harmless made a *measurement*
meaningless, which is worse than a bug — a bug is visible.

Read [`docs/decisions.md`](docs/decisions.md) before proposing architecture, and
[`SECURITY.md`](SECURITY.md) before touching the optimizer's reach.

## Setup

Python 3.12 (pinned: `>=3.12,<3.13`).

```bash
uv sync --extra dev --extra optimizer
```

`uv.lock` is committed and CI installs with `--locked`, so the lock stays
meaningful rather than decorative. If you change dependencies, commit the
regenerated lock in the same commit.

**Behind a corporate index proxy** (Databricks employees are; external
contributors generally are not): set `UV_INDEX` for the single install rather
than re-locking against your proxy. A re-lock rewrites every URL in `uv.lock`
to a host nobody else can reach, and CI's `--locked` then fails for everyone.
The hazard is documented in `pyproject.toml`.

## The checks

All four run in CI. Run them before you push.

```bash
uv run ruff check .
uv run mypy
uv run pytest
uv run pytest --cov --cov-fail-under=78
```

If `uv run` cannot resolve the index in your environment, `.venv/bin/python -m
pytest` works the same way. Note that `addopts` already contains `-q`; adding a
second one suppresses the summary line, which is a good way to misread 600
passing tests as 30.

### Tests are offline, and that is enforced

The whole suite must pass on a laptop with the wifi off. `tests/conftest.py`
refuses off-machine socket connections for every test not marked `live`.

Filtering by marker alone was not enough: an *unmarked* test could make a real
call, pass in CI, and only fail once someone ran the suite without egress. So a
test that genuinely needs a workspace must be marked:

```python
@pytest.mark.live      # deselected by default; runs from .github/workflows/live.yml
```

The three markers are `unit`, `contract`, and `live`, described in
`pyproject.toml`. `--strict-markers` is on, so a typo in a marker is an error.

### The mypy exemption list only shrinks

`pyproject.toml` carries two override blocks: strictness relaxations, and
`disable_error_code` for real mismatches. Both are **ratchets — never append.**
The phase that rewrites a module fixes its errors and deletes its entry.

Every silenced code is itemised with its consequence in
[`docs/type-debt.md`](docs/type-debt.md). Read that before touching the block. If
you need a new suppression, the honest move is an inline `# type: ignore[code]`
with a comment saying why, not a module-wide code.

To see what a block is actually hiding, delete it and run `mypy`. That
measurement is how the last pass found two behaviour bugs under what the doc had
called annotation debt.

## Writing tests that are worth having

The bar is not "the test passes". It is **"the test fails when the behaviour is
wrong."** Those are different, and this repo has been caught on the difference:
a set of tests passed with the fix under test reverted, because they exercised
helpers rather than the call sites.

So for anything load-bearing, revert your own change and confirm the test goes
red. If it does not, the test is testing the wrong thing. Prefer a call-through
test over an assertion about source text or a helper in isolation.

Two failure modes specific to this project:

- **A test that cannot fail.** A golden-set row whose `must_not_include` string
  appears in no other document is a trap with nothing to catch; the row passes
  unconditionally. `tests/test_example_domains.py` checks for exactly this.
- **A guard tested through its helper.** The guard runs at a call site. Test it
  there, or a future change that stops calling it will not be noticed.

## Changes that need more than a passing suite

### Anything that changes what a score *means*

If you change which rows a scorer applies to, what it returns, or how an
aggregate is computed, you have changed the measurement without changing any
config the fingerprint can see. Bump the scorer's entry in
`SCORER_SEMANTICS_VERSIONS` (`src/anvil/eval/scorers.py`) so cached baselines
from before the change are *refused* rather than silently chased, and regenerate
`eval/runs/baseline.json`.

This is `docs/decisions.md` D10. Groundedness is at v4 because it happened three
times.

### Anything that touches the gate

`src/anvil/loop/frontier.py` and `src/anvil/loop/decision.py` are strict-clean
and should stay that way. There must be exactly **one** definition of
comparability and one of applicability, in `anvil.eval.cache` and
`anvil.eval.judgeability`, which the gate *calls*. It once kept its own copy and
kept the bug after the eval's copy was fixed.

### Anything that widens the optimizer's reach

The optimizer is graded by the eval and rewarded for the score going up, so any
path from "edit a file" to "score goes up" is one it may take. See D8 and
`SECURITY.md`. Two specifics:

- Nothing that affects the score may become writable — no threshold, endpoint, or
  judge prompt moves into `scaffold/`.
- A **read** leaves no diff, so post-hoc verification cannot catch it. The secret
  set is built from the paths a round is actually using, not a fixed list.

### Architecture

Cite `docs/decisions.md` by number. Append new decisions; **never renumber** —
`scaffold/README.md` cites D8 by number. Three that are not up for trade:

- **D1** — one LLM SDK (`openai`, against Databricks endpoints). No second
  provider SDK in runtime dependencies.
- **D3** — no agent framework inside the runtime. The optimizer rewrites files it
  can read; a framework moves behaviour into library internals it cannot.
- **D6** — five planes, one-way imports. The runtime never imports the optimizer;
  the eval never imports git; the loop is the only orchestrator.

The MiniMax M2.7 pattern this design is calibrated against is an external
reference and is **not vendored in this repository**, so cite the decision rather
than a path when you appeal to it.

## Commits and pull requests

- Explain **why**, not what. The diff already says what.
- One logical change per PR.
- Prove it works. For anything behavioural, show the difference between `main`
  and the branch — a failing test that now passes, a mutation run, or measured
  output. "Tests pass" is not evidence that the change does anything.
- Say what you did not verify. A claim you could not check is fine; an unmarked
  one is not.

## Adding a domain

You should not need to edit `src/` to point ANVIL at your own problem. A domain
is four paths supplied at the boundary — `--kb-dir`, `--golden-set-path`,
`--evaluator-path`, `--scaffold` — plus the judge's domain text in
`harness/config.yaml`. `examples/pyloom-docs/` is a worked example, and
`tests/test_example_domains.py` is parametrized over `examples/*`, so a new
domain needs no new test file.

If something forces you into `src/` to add a domain, that is the bug. Report it
as one.
