---
name: release-management
description: Autonomous release management and PyPI deployment workflow for STRIDE. Guides version bumping adhering to SemVer, updating CHANGELOG.md, validating package distributions via build and twine, creating GitHub Releases with temporary note files, and monitoring PyPI OIDC publishing workflows.
---

# STRIDE Release Management & Deployment Workflow

Use this skill when preparing, validating, or publishing new releases of **STRIDE** (`stride-xai`) to GitHub Releases and PyPI.

---

## Workflow Steps

### Step 1: Version Evaluation & Pre-Flight Checks

1. Verify you are on `main` and fully synchronized with `origin/main`:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Determine commits since the last release tag:
   ```bash
   git describe --tags --abbrev=0
   git log $(git describe --tags --abbrev=0)..HEAD --oneline
   ```
3. Determine version increment per [`.agents/rules/release_standards.md`](../../rules/release_standards.md):
   - **`MAJOR`**: Breaking public API changes.
   - **`MINOR`**: New backwards-compatible features, datasets, estimators, or modules.
   - **`PATCH`**: Bug fixes, performance optimizations, CI/docs improvements.

---

### Step 2: Prepare Release Branch

If `__version__` in `src/stride/__init__.py` or `CHANGELOG.md` needs to be updated:
1. Create a release preparation branch:
   ```bash
   git checkout -b chore/release-vX.Y.Z origin/main
   ```
2. Update `__version__ = "X.Y.Z"` in [`src/stride/__init__.py`](../../../src/stride/__init__.py).
3. Update [`CHANGELOG.md`](../../../CHANGELOG.md) moving unreleased changes into a new header:
   ```markdown
   ## [X.Y.Z] - YYYY-MM-DD

   ### Added
   ...
   ### Fixed
   ...
   ```
4. Run local CI verification:
   ```bash
   ruff check .
   ruff format --check .
   python -m unittest discover tests
   ```
5. Commit and push:
   ```bash
   git add src/stride/__init__.py CHANGELOG.md
   git commit -m "chore(release): prepare vX.Y.Z release"
   git push -u origin chore/release-vX.Y.Z
   ```
6. Open PR, monitor CI, and merge into `main`.

---

### Step 3: Validate Local Distribution Build

Ensure the package builds cleanly and passes PyPI metadata checks:
```bash
python -m build
twine check --strict dist/*
```
Ensure output states `PASSED` for all distribution artifacts. Remove temporary build artifacts if needed (`rm -r dist/ build/`).

---

### Step 4: Draft and Publish GitHub Release

1. Write release notes to `.tmp_release_notes.md` following this structure:
   ```markdown
   # STRIDE vX.Y.Z — <Title / Theme>

   <Brief high-level overview of key additions and milestones in this release.>

   ### Highlights & Key Features
   - **<Feature 1>**: <Description>
   - **<Feature 2>**: <Description>

   ### Changelog Summary
   See [CHANGELOG.md](https://github.com/KubaCzech/STRIDE/blob/main/CHANGELOG.md) for full details.

   ### Installation
   ```bash
   pip install --upgrade stride-xai
   ```
   ```
2. Create the release using the GitHub CLI:
   - For standard production release (publishes to **PyPI**):
     ```bash
     gh release create vX.Y.Z --title "vX.Y.Z - <Title>" --notes-file .tmp_release_notes.md
     ```
   - For staging pre-release (publishes to **TestPyPI**):
     ```bash
     gh release create vX.Y.Z --title "vX.Y.Z - <Title>" --notes-file .tmp_release_notes.md --prerelease
     ```
3. Remove temporary release notes:
   ```bash
   rm .tmp_release_notes.md
   ```

---

### Step 5: Monitor PyPI Publication Workflow

Observe the automated publication job:
```bash
gh run list --workflow=release.yml --limit 1
gh run watch <run-id> --exit-status
```

Verify that the `publish-to-pypi` job completes successfully via OIDC Trusted Publishing.
