"""ANVIL loop plane — the orchestrator.

The only plane that knows about git, branches, cached baselines, and
keep/revert decisions. Drives one round end-to-end:

  1. Branch off ``anvil/exp`` → ``anvil/exp-round-N``.
  2. Build the round prompt from baseline + recent critiques + scaffold.
  3. Run the optimizer session (async, wrapped in asyncio.run).
  4. Apply the returned ``OptimizerAction`` to scaffold/.
  5. Commit.
  6. Run eval on the mutated branch (skip on noop).
  7. Compute score delta vs cached baseline.
  8. Write critique md + round JSON + mutations log row.
  9. Decide keep | revert | noop | infra_fail.
 10. ff-merge to ``anvil/exp`` (KEEP) or branch -D (everything else).
"""

from anvil.loop.builder import build_round_prompt
from anvil.loop.decision import Decision, decide
from anvil.loop.mutations_log import MutationRecord, append_mutation, load_mutations
from anvil.loop.round import RoundReport, run_round

__all__ = [
    "Decision",
    "MutationRecord",
    "RoundReport",
    "append_mutation",
    "build_round_prompt",
    "decide",
    "load_mutations",
    "run_round",
]
