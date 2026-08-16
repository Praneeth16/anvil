"""ANVIL optimizer plane.

Async ``ClaudeSDKClient`` session that proposes a single mutation per
round, structured as an ``OptimizerAction`` (Pydantic discriminated
union). The session emits exactly one fenced JSON action block which
the parser validates; any malformed output collapses to ``NoopAction``
so the loop is never blocked by a bad transcript.

The optimizer plane never runs git commands and never writes files
itself — the loop's applier does that. This separation makes the
optimizer mockable (loop tests inject a fake action) and the loop
auditable (every write goes through one place).
"""

from anvil.optimizer.actions import (
    AddRuleAction,
    AddSkillAction,
    ChangeSamplingAction,
    DeleteRuleAction,
    DeleteSkillAction,
    EditRuleAction,
    EditSkillAction,
    NoopAction,
    OptimizerAction,
)
from anvil.optimizer.applier import ApplyError, ApplyResult, apply_action
from anvil.optimizer.parser import ParseResult, parse_action
from anvil.optimizer.session import run_optimizer_session, setup_anthropic_env

__all__ = [
    "AddRuleAction",
    "AddSkillAction",
    "ApplyError",
    "ApplyResult",
    "ChangeSamplingAction",
    "DeleteRuleAction",
    "DeleteSkillAction",
    "EditRuleAction",
    "EditSkillAction",
    "NoopAction",
    "OptimizerAction",
    "ParseResult",
    "apply_action",
    "parse_action",
    "run_optimizer_session",
    "setup_anthropic_env",
]
