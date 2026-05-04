"""Pydantic models for the ANVIL harness configuration.

The harness configuration is split across two YAML files by
mutability:

* ``scaffold/harness.yaml`` — **mutable** by the optimizer.
  Sampling, skills, rules, tools.
* ``harness/config.yaml`` — **immutable** at runtime. Endpoints,
  experiment paths, eval modes, loop meta-config.

Both files are validated with ``extra="forbid"`` so a misplaced
field fails loudly. The loader catches those errors and reraises
with a domain-specific message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SamplingConfig(BaseModel):
    """Sampling parameters for a model call."""

    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int = 2048
    tool_choice: Literal["auto", "required", "none"] = "auto"
    max_tool_calls: int = 3


class SkillRef(BaseModel):
    file: str
    sampling: SamplingConfig | None = None

    @property
    def name(self) -> str:
        return Path(self.file).stem


class RuleRef(BaseModel):
    file: str

    @property
    def name(self) -> str:
        return Path(self.file).stem


class ToolRef(BaseModel):
    """Tool entry registered in the runtime agent's tool list."""

    name: str
    description: str | None = None


class LoopConfig(BaseModel):
    """Configuration for the ANVIL optimizer loop."""

    target_rounds: int = 50
    stretch_rounds: int = 100
    cost_budget_usd_per_round: float = 5.0
    early_stop_after_stalled_rounds: int = 10
    critique_lookback: int = 3
    revert_lookback: int = 20
    max_optimizer_turns: int = 30


class ExperimentsConfig(BaseModel):
    """MLflow experiment paths. Stable, declared in config.yaml."""

    runtime: str
    eval: str
    optimizer: str


class EvalModeConfig(BaseModel):
    rows: int
    buckets: dict[str, int] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    """Eval-side configuration."""

    default_mode: Literal["quick", "standard", "full"] = "standard"
    modes: dict[str, EvalModeConfig] = Field(default_factory=dict)
    n_workers: int = 4
    inter_row_cooldown_s: float = 0.0
    scorers: list[str] = Field(
        default_factory=lambda: ["correctness", "retrieval_groundedness", "refusal_appropriateness"]
    )
    safety_guard_threshold: float = 0.95


class ScaffoldYAML(BaseModel):
    """Schema of ``scaffold/harness.yaml`` — the optimizer-mutable file."""

    model_config = ConfigDict(extra="forbid")

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    skills: list[SkillRef] = Field(default_factory=list)
    rules: list[RuleRef] = Field(default_factory=list)
    tools: list[ToolRef] = Field(default_factory=list)


class RuntimeYAML(BaseModel):
    """Schema of ``harness/config.yaml`` — the immutable runtime file."""

    model_config = ConfigDict(extra="forbid")

    runtime_endpoint: str
    optimizer_endpoint: str
    judge_endpoint: str
    experiments: ExperimentsConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


class HarnessConfig(BaseModel):
    """Merged view of both YAML files. Built by the loader."""

    runtime_endpoint: str
    optimizer_endpoint: str
    judge_endpoint: str
    experiments: ExperimentsConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    skills: list[SkillRef] = Field(default_factory=list)
    rules: list[RuleRef] = Field(default_factory=list)
    tools: list[ToolRef] = Field(default_factory=list)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def from_split(cls, scaffold: ScaffoldYAML, runtime: RuntimeYAML) -> HarnessConfig:
        return cls(
            runtime_endpoint=runtime.runtime_endpoint,
            optimizer_endpoint=runtime.optimizer_endpoint,
            judge_endpoint=runtime.judge_endpoint,
            experiments=runtime.experiments,
            sampling=scaffold.sampling,
            skills=list(scaffold.skills),
            rules=list(scaffold.rules),
            tools=list(scaffold.tools),
            loop=runtime.loop,
            eval=runtime.eval,
        )


# Field names that belong in the *other* file. Used by the loader to
# generate domain-specific errors when an extra field happens to be
# one of the canonical fields on the opposite side of the split.
RUNTIME_FIELDS: frozenset[str] = frozenset(
    {"runtime_endpoint", "optimizer_endpoint", "judge_endpoint", "experiments", "loop", "eval"}
)
SCAFFOLD_FIELDS: frozenset[str] = frozenset({"sampling", "skills", "rules", "tools"})
