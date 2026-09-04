# Agent Context Self-Maintenance Standards

> [!IMPORTANT]
> **Trigger Paths**: `.agents/**`, `AGENTS.md`
> **When to Read**: MUST be read upon completing features, changing dependencies, adding new modules, or updating architectural patterns.

To maintain `.agents/` as the single authoritative Source of Truth, all AI assistants must proactively maintain and refine documentation, schema context, and rules.

## Mandatory Maintenance Triggers

| Trigger Event | Action Required | Target File(s) |
| :--- | :--- | :--- |
| **Feature / Bugfix Completed** | Mark task as `[x]`, add upcoming roadmap items, update objective. | `.agents/project_context.md` |
| **New Module or Algorithm Added** | Document module architecture, interface contracts, and data flows. | `.agents/context/architecture.md` |
| **Dependency / Stack Change** | Document new library conventions or configuration requirements. | Relevant rule in `.agents/rules/` |
| **Recurring Pitfall Resolved** | Add declarative guardrails and anti-pattern table rows. | Relevant rule in `.agents/rules/` |
| **CI / Build Modification** | Update local verification checklist and commands. | `.agents/rules/ci_standards.md` |

## Standard Maintenance Procedure

1. Review modified files (`git status` and `git diff --stat`).
2. Update corresponding `.agents/context/` or `.agents/rules/` files.
3. Update `Repository Status` in `.agents/project_context.md`.
4. Stage and commit `.agents/` updates alongside feature code.
