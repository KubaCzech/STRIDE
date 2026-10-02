---
name: git-pr-workflow
description: End-to-end Git development workflow for STRIDE. Guides branch creation based on origin/main, atomic Conventional Commits, local CI verification, commit ancestry audit, push, CI monitoring, and Pull Request creation via gh CLI using temporary markdown body files.
---

# STRIDE Git & Pull Request Workflow

Use this skill when developing any feature, bugfix, refactor, algorithm, or maintenance task on the **STRIDE** repository.

---

## Process

### Step 1: Branch Creation & GitHub Issues Protocol

**Strict Pre-Implementation Invariant**: NEVER modify, create, or stage files directly on `main`. Before writing any code (even small bugfixes or UI adjustments), check your active branch (`git branch --show-current`). Always base your branch on the latest `origin/main`:

```bash
git fetch origin main
git checkout -b <branch-name> origin/main
```

> [!WARNING]
> **Avoid Branching Off Stale Local Branches**:
> Never execute a bare `git checkout -b <branch-name>` without specifying `origin/main`. Because GitHub merges PRs via "Squash and Merge", merged commits receive new hashes on `main`. Branching off a local branch that contains old pre-squash commits drags those old commits into your new PR as duplicate "rogue" commits.
> If uncommitted modifications already exist before branching:
> `git stash && git fetch origin main && git checkout -b <branch-name> origin/main && git stash pop`.

#### Branch Naming Guidelines
- **Issue-Linked Work** (if a GitHub issue exists or is assigned):
  - Features: `feat/<issue-num>-<short-description>`
  - Fixes: `fix/<issue-num>-<short-description>`
  - Refactors: `refactor/<issue-num>-<short-description>`
- **Direct Prompt Work** (interactive user tasks without pre-existing issues):
  - Features: `feat/<short-description>`
  - Fixes: `fix/<short-description>`
  - Refactors: `refactor/<short-description>`
  - Chores / Docs: `chore/<short-description>` or `docs/<short-description>`
  - Tests: `test/<short-description>`

*Note: Do not create redundant intermediate GitHub issues for direct prompt tasks just to close them immediately. PR descriptions serve as the primary artifact.*

---

### Step 2: Implement & Commit with Conventional Commits

Write atomic commits as you complete discrete, logical units of work:

```bash
git add <files>
git commit -m "<type>[optional scope]: <description>"
```

- **Allowed types**: `feat`, `fix`, `refactor`, `style`, `test`, `docs`, `chore`
- **Scope**: Noun describing affected area (`package`, `api`, `drift`, `xai`, `dashboard`, `models`, `datasets`, `clustering`, `recurrence`, `ci`, `agents`, etc.), or omitted for broad chores.
- **Linguistic rule**: Imperative, present tense (e.g., `feat(api): export public drift estimators`)
- **Atomic Checklist** (ensure all 3 are "Yes" before committing):
  1. *Single Purpose*: Diff solves exactly one problem or discrete feature.
  2. *State Stability*: Codebase compiles and tests pass at this commit checkpoint.
  3. *Diff Isolation*: Logic changes are isolated from formatting/style changes.
- Refer to [git_and_pr_standards.md](../../rules/git_and_pr_standards.md#2-conventional-commits--atomic-commit-protocol) for complete guidelines.

#### Tracking Out-of-Scope Issues (Optional)
If you discover bugs or technical debt outside the scope of your active branch:
1. Write the issue body to `.tmp_issue_body.md` using the standard Issue templates (see [git_and_pr_standards.md](../../rules/git_and_pr_standards.md#6-standardized-github-issue-structure)).
2. Create the issue via `gh issue create --title "<type>(<scope>): <description>" --body-file .tmp_issue_body.md`.
3. Delete `.tmp_issue_body.md`.
4. Remain focused on your current branch.

---

### Step 3: Mandatory Local CI Verification

Run the exact equivalents of the repository's GitHub Actions CI workflows:

```bash
# 1. Linting check
ruff check .

# 2. Formatting check
ruff format --check .

# 3. Automated test suite (run in activated virtual environment)
python -m unittest discover tests
```

*Ensure there are 0 lint errors, 0 format warnings, and all tests pass cleanly before proceeding.*

---

### Step 4: Check & Synchronize `.agents/` Context

Evaluate if your changes triggered any self-maintenance updates (refer to `agent-maintenance` skill):
- Added or modified models/estimators/generators? -> update [`.agents/context/architecture.md`](../../context/architecture.md)
- Completed milestone or roadmap goals? -> check off `[x]` in [`.agents/project_context.md`](../../project_context.md)
- Added dependencies or architectural rules? -> update [`.agents/rules/`](../../rules/) and [`.agents/acs.yaml`](../../acs.yaml)

Include any necessary `.agents/` updates in your branch commits.

---

### Step 5: Pre-Push Commit Ancestry Audit, Push & Actions Monitoring

#### 1. Mandatory Commit Ancestry Audit
Before pushing, audit your branch's commit history relative to `origin/main`:

```bash
git fetch origin main
git log origin/main..HEAD --oneline
```

- **Pass**: Output lists ONLY commits authored specifically for this task.
- **Fail (Rogue Commits Detected)**: If commits from previous PRs or merged branches appear, rebase cleanly onto `origin/main`:
  ```bash
  git rebase --onto origin/main <last-rogue-commit> HEAD
  ```

#### 2. Push & Actions Monitoring
Push your branch to GitHub and observe the remote workflow run:

```bash
git push -u origin <branch-name>
```

Track the Actions run:
```bash
gh run list --limit 1
gh run watch <run-id> --exit-status
```

If the remote CI fails, immediately inspect logs (`gh run view <run-id> --log`), resolve discrepancies locally, commit, and re-push.

---

### Step 6: Create Pull Request via Temporary Body File

To prevent escaping errors and broken markdown on Windows/PowerShell shells, **never pass inline multi-line markdown strings** to `gh`. Always use a temporary file.

> [!CAUTION]
> **No Git Meta-Process in Verification**:
> The `## Verification` section is strictly for automated tests, linting, and application runtime checks. **NEVER** include internal Git commands or ancestry audits (e.g., `git log origin/main..HEAD`) in the PR body.

1. **Write PR body to `.tmp_pr_body.md`**:
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
- [x] <Specific manual test step or behavioral verification performed (optional)>
```

2. **Execute `gh pr create`**:
```bash
gh pr create --title "<type>(<scope>): <short description>" --body-file .tmp_pr_body.md
```

3. **Delete `.tmp_pr_body.md`**:
```bash
rm .tmp_pr_body.md
```
