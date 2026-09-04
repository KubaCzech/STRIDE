# CI Standards & Local Verification Rules

> [!IMPORTANT]
> **Trigger Paths**: `.github/workflows/**`, `requirements.txt`, `tests/**`
> **When to Read**: MUST be read before running local verification, committing code, or creating pull requests.

To maintain high code quality and prevent CI build failures, all changes must be verified locally against the CI pipeline equivalents before pushing.

## Mandatory Local Verification Checklist

Execute these commands from the repository root:

1. **Ruff Linting**:
   ```bash
   ruff check .
   ```
   *Requirement: Must exit with 0 errors.*

2. **Ruff Format Check**:
   ```bash
   ruff format --check .
   ```
   *Requirement: All files must already be formatted.*

3. **Automated Test Suite**:
   ```bash
   python -m unittest discover tests
   ```
   *Requirement: All test cases must pass (OK).*

---

## Pre-Push Verification Protocol

Under no circumstances execute `git push` without running the above checks. If a test or check fails:
1. Investigate the failure log and identify the root cause.
2. Fix the regression in application/test code or run `ruff format .` / `ruff check --fix .`.
3. Re-run Ruff and Unittest to ensure complete green status.
