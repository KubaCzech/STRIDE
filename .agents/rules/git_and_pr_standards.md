# Git & Pull Request Standards

> [!IMPORTANT]
> **Trigger Paths**: Any Git source control action (branching, committing, pushing, PR/issue creation).
> **When to Read**: MUST be read before staging changes, formatting Conventional Commits, running pre-push CI verification, or opening GitHub PRs.

This rule document governs all version control, branching, committing, issue tracking, and pull request operations across STRIDE.

---

## 1. Branching Strategy

### Strict Pre-Implementation Invariant
Never write, modify, or stage code directly on the `main` branch. Before making ANY changes to the codebase, check `git branch --show-current`. If on `main`, pull the latest changes and create a dedicated branch first:

```bash
git checkout main
git pull origin main
git checkout -b <branch-name> origin/main
```

### Branch Naming Conventions
- **New Features**: `feat/<short-description>`
- **Bug Fixes**: `fix/<short-description>`
- **Refactoring & Architecture**: `refactor/<short-description>`
- **Documentation**: `docs/<short-description>`
- **Chores / Maintenance**: `chore/<short-description>`
- **Tests**: `test/<short-description>`

---

## 2. Conventional Commits & Atomic Commit Protocol

All commit messages must strictly adhere to the Conventional Commits specification:

```
<type>[optional scope]: <description>
```

### Allowed Types
- `feat`: A new feature or algorithm is introduced.
- `fix`: A bug or regression is patched.
- `refactor`: Code rewriting without changing external behavior or fixing a bug.
- `style`: Formatting changes that do not alter code logic (e.g. whitespace, formatting).
- `test`: Adding, updating, or correcting tests.
- `docs`: Documentation-only changes.
- `chore`: Updating build tasks, configurations, dependencies, or agent context.

### Scope Guidelines
- Scopes must be a single, meaningful lowercase noun describing the affected area (e.g., `package`, `api`, `drift`, `xai`, `dashboard`, `models`, `datasets`, `ci`, `agents`).
- Write `<description>` in the **imperative, present tense** (e.g., `feat(api): export public drift estimators`, NOT `added` or `adding`).

### Granularity Protocol (Atomic Commits Checklist)
Before staging and committing changes, evaluate this 3-point checklist:
1. **Single Purpose**: Does this commit solve exactly one logical problem or implement exactly one discrete change?
2. **State Stability**: If this commit is checked out independently, does the codebase build and do all tests pass?
3. **Diff Isolation**: Are logic changes kept separated from formatting, styling, or unrelated refactorings?

---

## 3. Mandatory Pre-Push Local Verification

Under no circumstances execute `git push` without first running local CI verification equivalents:

```bash
# 1. Linting check
ruff check .

# 2. Format check
ruff format --check .

# 3. Test suite
.venv\Scripts\python.exe -m unittest discover tests
```

*Ensure all checks pass with 0 errors before proceeding.*

---

## 4. Standardized Pull Request Protocol

### Mandatory Temporary Body File
**NEVER** pass multi-line or formatted markdown directly via inline CLI arguments (`gh pr create --body "..."`). On Windows/PowerShell shells, inline quotes, newlines, and markdown formatting get mangled.

**Always use a temporary markdown file:**
1. Write the PR content to `.tmp_pr_body.md`.
2. Run `gh pr create --title "<type>(<scope>): <description>" --body-file .tmp_pr_body.md`.
3. Delete `.tmp_pr_body.md` immediately after PR creation.

### Pull Request Structure Standard
```markdown
## Summary
- Bullet point overview of high-level changes.
- Key outcomes and system impacts.

## Motivation
- Why this change is necessary.
- Context on the problem solved.

## Key Changes
### 1. <Subsystem / Feature Area 1>
- Specific technical modifications made.

## Verification
- [x] `ruff check .` passed with 0 errors
- [x] `ruff format --check .` passed cleanly
- [x] `python -m unittest discover tests` passed (OK)
- [x] <Specific manual test step performed>
```
