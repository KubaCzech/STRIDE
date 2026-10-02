# Git & Pull Request Standards

> [!IMPORTANT]
> **Trigger Paths**: Any workspace Git source control action (branching, committing, pushing, PR/issue creation).
> **When to Read**: MUST be read before staging changes, formatting Conventional Commits, running pre-push CI verification, or opening GitHub PRs/issues.

This rule document governs all version control, branching, committing, issue tracking, and pull request operations across the **STRIDE** repository. All AI assistants and contributors must operate strictly according to this protocol.

---

## 1. Branching Strategy & GitHub Issues Protocol

### Strict Pre-Implementation Invariant
Never commit or stage code directly on the `main` branch. The `main` branch is protected by CI workflows and repository policies requiring all changes to pass automated checks and be merged via a Pull Request.

Before writing, modifying, or staging ANY changes to the codebase (even for trivial one-line fixes, typos, or minor UI adjustments), check `git branch --show-current`. Always create a dedicated branch based directly on the latest `origin/main`:

```bash
git fetch origin main
git checkout -b <branch-name> origin/main
```

> [!WARNING]
> **Avoid Branching Off Stale Local Branches**:
> Never execute a bare `git checkout -b <branch-name>` without specifying `origin/main` as the starting point.
> Because GitHub merges pull requests via "Squash and Merge", merged commits receive new SHAs on `main`.
> Branching off a local branch that contains old pre-squash commits will drag those old commits into your new PR as duplicate "rogue" commits!
> If uncommitted modifications already exist in the working directory before branching, stash them first:
> `git stash && git fetch origin main && git checkout -b <branch-name> origin/main && git stash pop`.

### Hybrid GitHub Issues Protocol
We follow a pragmatic, context-aware approach to GitHub Issues:

1. **Issue-Linked Work**: If an issue already exists on GitHub or the user provides an issue number:
   - Branch name: `feat/<issue-num>-<short-description>`, `fix/<issue-num>-<short-description>`, or `refactor/<issue-num>-<short-description>`
   - Link in PR: Include `Closes #<issue-num>` or `Resolves #<issue-num>` in the PR body.
2. **Direct Prompt Work (Default)**: For interactive feature requests, algorithm implementations, dashboard enhancements, and bugfixes without pre-existing issues:
   - Branch name: `feat/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`, etc.
   - Do NOT create redundant intermediate GitHub issues just to immediately close them. The Pull Request body serves as the primary artifact.
3. **Out-of-Scope Bug & Tech Debt Tracking**: If you discover significant defects, missing test coverage, or architectural debts outside the active task scope:
   - File a new GitHub Issue using the standard Issue templates (see Section 6) via `gh issue create --body-file .tmp_issue_body.md`.
   - Keep the active branch strictly focused on its primary objective without getting derailed.

### Branch Naming Conventions
- **New Features**: `feat/<short-description>` or `feat/<issue-number>-<short-description>`
- **Bug Fixes**: `fix/<short-description>` or `fix/<issue-number>-<short-description>`
- **Refactoring & Architecture**: `refactor/<short-description>` or `refactor/<issue-number>-<short-description>`
- **Documentation**: `docs/<short-description>`
- **Chores / Tooling**: `chore/<short-description>`
- **Tests**: `test/<short-description>`

*Examples:* `feat/drift-detection-cusum`, `fix/dashboard-slider-bounds`, `refactor/srp-cluster-detector`, `docs/ecml-citation-badge`

---

## 2. Conventional Commits & Atomic Commit Protocol

All commit messages must strictly adhere to the Conventional Commits specification:

```
<type>[optional scope]: <description>
```

### Allowed Types
- `feat`: A new feature, algorithm, or capability is introduced.
- `fix`: A bug, calculation error, or regression is patched.
- `refactor`: Code rewriting without changing external behavior or fixing a bug.
- `style`: Formatting changes that do not alter code logic (e.g., whitespace, linter formatting). Dashboard UI/styling adjustments should be categorized as `feat` or `fix`.
- `test`: Adding, updating, or correcting tests.
- `docs`: Documentation-only changes.
- `chore`: Updating build tasks, configurations, dependencies, or agent context.

### Scope Guidelines
The `[optional scope]` provides context on what module or architectural layer the commit affects:
1. **Noun-Based**: Must be a single, meaningful lowercase noun describing the affected area:
   - `package`: PyPA/PEP 621 packaging, `pyproject.toml`, build metadata
   - `api`: Public package exports in `stride` or top-level APIs
   - `drift`: Drift generators and statistical detectors (`stride.datasets`, `stride.DDM`)
   - `xai`: Decision boundary and feature importance analysis (`stride.decision_boundary`, `stride.feature_importance`)
   - `clustering`: Micro-cluster and clustering dynamics (`stride.clustering`)
   - `recurrence`: Recurring concept and prototype analysis (`stride.recurrence`)
   - `models`: Underlying classifiers and model adapters (`stride.models`)
   - `dashboard`: Streamlit UI, tabs, widgets, and config schemas (`dashboard/`)
   - `ci`: GitHub Actions workflows, linting, and CI config
   - `agents`: AI context files, rules, skills, and configuration in `.agents/`
2. **Short & Lowercase**: Keep it brief (one word) and strictly lowercase (e.g., `fix(drift): ...`).
3. **No File Paths**: Do NOT use file names or relative file paths as scopes (e.g., `fix(src/stride/drift.py)` is invalid).
4. **When to Omit**: Omit the scope if the commit affects multiple areas equally or represents a broad chore (e.g., `chore: update dependencies`).

### Linguistic Rule
- Write `<description>` in the **imperative, present tense** (e.g., `feat(api): export public drift estimators`, NOT `added` or `adding`).

### Granularity Protocol (Atomic Commits Checklist)
Before staging and committing changes, evaluate this internal 3-point checklist. If any answer is **"No"**, split the changes into smaller, discrete commits:
1. **Single Purpose**: Does this commit solve exactly one logical problem or implement exactly one discrete feature/fix?
2. **State Stability**: If this individual commit is checked out independently, does the codebase build cleanly and do all tests pass?
3. **Diff Isolation**: Are logic changes kept separated from formatting, styling, or unrelated refactoring changes?

### Execution Triggers
- **Commit Trigger**: Commit as soon as a single, logical unit of work (as defined by the checklist) is fully implemented and locally verified (or to create a safe fallback point prior to attempting experimental changes).
- **Push Trigger**: Push to remote when a branch is complete and ready for PR review, or when saving progress to prevent data loss.

### Repository Hygiene & Sensitive Data Safeguards
Before staging files (`git add`), verify repository hygiene:
1. **No Sensitive Info**: Check for secrets, credentials, API keys, or `.env` files.
2. **No Build/Local Artifacts**: Ensure build output (`dist/`, `build/`, `*.egg-info`), scratch files, cache folders (`.ruff_cache/`, `__pycache__/`), and OS metadata are ignored or not staged.
3. **Gitignore Compliance**: Check `.gitignore` before committing new file types.

---

## 3. Mandatory Pre-Push Local Verification

Under no circumstances execute `git push` without first running local CI verification equivalents:

```bash
# 1. Linting check with Ruff
ruff check .

# 2. Formatting check with Ruff
ruff format --check .

# 3. Test suite in activated virtual environment
python -m unittest discover tests
```

*Ensure all checks pass with 0 errors and 0 test failures before proceeding.*

---

## 4. Pre-Push Commit Ancestry Audit & CI Workflow Monitoring

### Mandatory Commit Ancestry Audit
Before executing `git push`, you MUST audit your branch's commit history relative to `origin/main`:

```bash
git fetch origin main
git log origin/main..HEAD --oneline
```

- **Pass**: The output lists ONLY commits authored specifically for this task.
- **Fail (Rogue / Duplicate Commits Detected)**: If commits from previous PRs or merged branches appear, rebase cleanly onto `origin/main` to drop them:
  ```bash
  git rebase --onto origin/main <last-rogue-commit> HEAD
  ```
- Re-run `git log origin/main..HEAD --oneline` to confirm a clean, isolated commit list before pushing.

### Push & CI Workflow Monitoring
1. Push your branch to the remote origin:
   ```bash
   git push -u origin <branch-name>
   ```
2. Monitor GitHub Actions CI in real-time:
   ```bash
   gh run list --limit 1
   gh run watch <run-id> --exit-status
   ```
3. If remote CI fails, immediately inspect logs (`gh run view <run-id> --log`), resolve discrepancies locally, commit, and re-push.

---

## 5. Standardized Pull Request Protocol

### ⚠️ Critical Rule: Mandatory Temporary Body File
**NEVER** pass multi-line or formatted markdown directly via inline CLI arguments (`gh pr create --body "..."`). On Windows/PowerShell shells, inline quotes, newlines, backticks, and markdown formatting get mangled, causing syntax errors or broken PR formatting.

**Always use a temporary markdown file:**
1. Write the PR content to `.tmp_pr_body.md`.
2. Run `gh pr create --title "<type>(<scope>): <description>" --body-file .tmp_pr_body.md`.
3. Delete `.tmp_pr_body.md` immediately after PR creation.

### Pull Request Structure Standard
Every PR description must follow this structure:

```markdown
## Summary
- Bullet point overview of high-level changes.
- Key outcomes and user-facing/system impacts.

## Motivation
- Why this change is necessary.
- Context on the problem solved.
- Issue reference: `Closes #<issue-num>` or `Resolves #<issue-num>` (if applicable).

## Key Changes (Optional / For multi-part features)
### 1. <Subsystem / Feature Area 1>
- Specific technical modifications made.
### 2. <Subsystem / Feature Area 2>
- Specific technical modifications made.

## Verification
- [x] `ruff check .` passed with 0 errors
- [x] `ruff format --check .` passed cleanly
- [x] `python -m unittest discover tests` passed with 0 failures
- [x] <Specific manual test step or behavioral verification performed>
```

---

## 6. Standardized GitHub Issue Structure

When creating GitHub Issues for new features, bug reports, or technical debt, always use a temporary file (`.tmp_issue_body.md`) and adhere to these standardized templates:

### Bug Report (`fix`)
```markdown
## Problem Summary
<Clear, concise description of the defect, incorrect drift metric, or UI malfunction.>

## Steps to Reproduce
1. Launch dashboard via `streamlit run dashboard/app.py`
2. Select dataset '...' and model '...'
3. Adjust parameter '...'
4. Observe unexpected behavior '...'

## Expected vs Actual Behavior
- **Expected**: <What should happen>
- **Actual**: <What currently happens>

## Affected Area & Environment
- Module: `package` | `drift` | `xai` | `clustering` | `dashboard` | `models`
- Environment: Python version, OS, browser (if dashboard issue)
```

### Feature / Enhancement Request (`feat`)
```markdown
## Feature Summary & Context
<High-level overview of the proposed xAI capability, statistical detector, or dashboard enhancement.>

## Proposed Solution & Implementation Details
- <Key algorithmic additions under src/stride/>
- <Decoupled config additions under dashboard/config/>
- <Dashboard UI view/tab integration under dashboard/views/>

## Acceptance Criteria
- [ ] <Algorithmic implementation requirement fulfilled>
- [ ] <Unit test coverage added under tests/>
- [ ] Local verification commands (`ruff check .`, `ruff format --check .`, `python -m unittest discover tests`) pass with 0 errors.
```

### Refactoring & Technical Debt (`refactor`)
```markdown
## Current State & Pain Points
<Detailed description of existing architectural debts, performance bottlenecks, or code duplication.>

## Proposed Refactoring Strategy
- <Target classes or modules to decompose/modernize>
- <Design pattern or interface decoupling improvement>

## Acceptance Criteria
- [ ] Existing algorithmic and dashboard functionality preserved without regressions.
- [ ] Clean separation of concerns (e.g., zero UI imports in core algorithms, headless plotting).
- [ ] All automated tests pass cleanly.
```

---

## 7. Context Self-Maintenance Check

Before submitting a Pull Request for significant changes:
- Check if `.agents/project_context.md`, `.agents/context/architecture.md`, or `.agents/rules/` require updates to reflect the new state.
- Include those context updates in the branch commits.
