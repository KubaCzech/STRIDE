---
name: agent-maintenance
description: Maintain and update the .agents/ directory structure to reflect the current state of the repository after completing features, updating models, or altering dependencies.
---

# Agent Repo Maintenance Skill

Use this skill when you finish implementing a major feature, detect a shift in the tech stack or dependencies, modify models/API endpoints, or resolve significant architectural debts.

## Maintenance Procedure

1. **Update Project Context (`.agents/project_context.md`)**: Check off `[x]` finished roadmap tasks, add new `[ ]` tasks, update the current objective.
2. **Synchronize Data Models & Architecture (`.agents/context/`)**: Update module specifications if new datasets, models, or explainers were introduced.
3. **Refine Rules & Standards (`.agents/rules/`)**: Add library conventions, anti-pattern table rows, or CI adjustments.
4. **Commit Updates**: Stage and commit `.agents/` documentation alongside feature code following `git-commit-protocol`.
