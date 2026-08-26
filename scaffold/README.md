# ANVIL scaffold — MultiHopRAG domain

The mutable agent harness. **This directory is the scaffold.** Every file
here is something the ANVIL optimizer may rewrite autonomously.

The immutable runtime configuration (endpoints, eval modes, gate) lives in
`harness/config.yaml`, OUTSIDE this directory. The optimizer's write scope
must not include that file. See `docs/decisions.md` D8.

This is the starting scaffold for the MultiHopRAG domain (issue #15): a
research assistant over a 2023 news-article knowledge base. It is
deliberately plain — a reasonable starting point, not a tuned one. Tuning
it is the optimizer's job; the previous domain's tuned scaffold, with ten
rounds of critique history, is preserved at `examples/neovolt/`.

## Layout

```
scaffold/
├── harness.yaml     # sampling, active skills[], rules[], tools[]
├── skills/          # identity (required, exactly one), citation, synthesis, refusal
└── rules/           # citation and scope discipline
```

On deploy, a DAB task syncs this directory into the UC Volume
`anvil.default.scaffold`. The runtime `ResponsesAgent` loads from the
Volume at startup.
