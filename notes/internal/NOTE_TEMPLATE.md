# Internal Investigation Note Template

Use this template for iterative debugging/performance notes that should remain
maintainer-facing (not public docs).

## Title

Short, specific title (feature + issue or experiment focus).

## Context

- Why this note exists
- Triggering user report / regression / release blocker
- Scope (files/systems involved)

## Reproduction

- Environment (robot/model/platform, Python/C++ path, key flags)
- Exact commands
- Expected vs observed behavior

## Findings

- Root-cause hypotheses considered
- Confirmed root cause(s)
- Relevant code locations

## Changes Tried

| Attempt | Change | Outcome | Keep/Revert |
|---------|--------|---------|-------------|
| A | ... | ... | ... |

## Final Decision

- What was kept
- What was reverted
- Why this trade-off was chosen

## Validation

- Tests run
- Benchmarks run
- Remaining risks / known gaps

## Follow-ups

- Next tasks
- Guardrail tests/docs to add
