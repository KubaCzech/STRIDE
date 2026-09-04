# CI Standards & Local Verification Rules

> [!IMPORTANT]
> **Trigger Paths**: `.github/workflows/**`, `requirements.txt`, `tests/**`
> **When to Read**: MUST be read before running local verification, committing code, or creating pull requests.

To maintain high code quality and prevent CI build failures, all changes must be verified locally against the CI pipeline equivalents before pushing.

## Mandatory Local Verification Checklist

Execute these commands from the repository root:

1. **Flake8 Syntax & Critical Error Linting**:
   ```bash
   flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude .venv,sdbm
   ```
   *Requirement: Must exit with 0 errors.*

2. **Flake8 Extended Style Check**:
   ```bash
   flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude .venv,sdbm
   ```

3. **Automated Test Suite**:
   ```bash
   python -m unittest discover tests
   ```
   *Requirement: All test cases must pass (OK).*

---

## Pre-Push Verification Protocol

Under no circumstances execute `git push` without running the above checks. If a test fails:
1. Investigate the failure log and identify the root cause.
2. Fix the regression in application/test code.
3. Re-run both Flake8 and Unittest to ensure complete green status.
