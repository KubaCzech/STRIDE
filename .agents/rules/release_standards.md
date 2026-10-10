# Release & Version Management Standards

> [!IMPORTANT]
> **Trigger Paths**: `CHANGELOG.md`, `.github/workflows/release.yml`, `pyproject.toml`, `src/stride/__init__.py`, and any GitHub Release operations.
> **When to Read**: MUST be read before drafting releases, incrementing package versions, publishing to PyPI, or updating `CHANGELOG.md`.

This rule document establishes strict deterministic standards for managing versions, documenting release history, and executing automated releases to PyPI within the **STRIDE** repository.

---

## 1. Release Architecture & PyPI Publication Pipeline

Publication to PyPI is governed by [`.github/workflows/release.yml`](../../.github/workflows/release.yml):
- **Trigger**: Publication runs **strictly** when a GitHub Release event of type `published` occurs (or via manual `workflow_dispatch`).
- **Target Environments**:
  - **TestPyPI** (`https://test.pypi.org/p/stride-xai`): Triggered when a GitHub Release is marked as **Pre-release** (`prerelease: true`). Runs an automated packaging smoke test validating pip installation.
  - **PyPI Production** (`https://pypi.org/p/stride-xai`): Triggered when a GitHub Release is published as a **Full Release** (`prerelease: false`). Uses GitHub OpenID Connect (OIDC) Trusted Publishing (no static API tokens stored).

> [!WARNING]
> Merging a Pull Request into `main` does **NOT** publish a new version to PyPI. A GitHub Release must be explicitly drafted and published on GitHub.

---

## 2. Semantic Versioning (SemVer) Invariant

STRIDE adheres strictly to [Semantic Versioning 2.0.0](https://semver.org/):

$$\text{Version} = \text{MAJOR}.\text{MINOR}.\text{PATCH}$$

| Increment | Trigger Conditions | Examples |
| :--- | :--- | :--- |
| **MAJOR** (`X.0.0`) | Breaking public API changes; removing or fundamentally modifying public class interfaces, method signatures, or supported stream schemas without backwards compatibility. | Signature overhaul of `BaseDataset.generate_data()` or removing core models. |
| **MINOR** (`0.X.0`) | Backwards-compatible new functionality; adding new datasets, drift generators, estimators, xAI explanation modules, or dashboard views. | Adding Shaker Protocol semi-synthetic drift (`stride.datasets.SemiSyntheticDriftDataset`), adding ADWIN. |
| **PATCH** (`0.0.X`) | Backwards-compatible bug fixes, performance optimizations, documentation improvements, CI or packaging metadata fixes. | Fixing slider bounds in dashboard, correcting typos, updating README badges. |

---

## 3. Mandatory Pre-Release Checklist

Before creating any GitHub Release or tag, the following conditions MUST be verified:

1. **Clean `main` Branch Invariant**: The release tag must be cut from the latest commit on `origin/main`. Never release from an unmerged feature branch.
2. **Version Synchronization**: The version string `__version__ = "X.Y.Z"` in [`src/stride/__init__.py`](../../src/stride/__init__.py) must exactly match the proposed release tag `vX.Y.Z`.
3. **Changelog Integrity**: [`CHANGELOG.md`](../../CHANGELOG.md) must have an explicit section for `## [X.Y.Z] - YYYY-MM-DD` adhering to [Keep a Changelog](https://keepachangelog.com/):
   - Categorized subsections: `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, `### Security`.
   - All merged features since the previous tag documented clearly.
4. **Local Build & Metadata Validation**:
   ```bash
   python -m build
   twine check --strict dist/*
   ```
   *Must output `PASSED` for both source distribution (`.tar.gz`) and wheel (`.whl`).*
5. **Local CI Validation**:
   - `ruff check .` passes with 0 errors.
   - `ruff format --check .` passes cleanly.
   - `python -m unittest discover tests` passes with 0 failures.

---

## 4. Release Execution Protocol

When releasing a new version:

### Step 1: Release Preparation PR (if version bump is needed)
If `src/stride/__init__.py` or `CHANGELOG.md` need updates:
1. Create a branch: `chore/release-vX.Y.Z` off `origin/main`.
2. Update `__version__ = "X.Y.Z"` in `src/stride/__init__.py`.
3. Update `CHANGELOG.md` by moving items from `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`.
4. Verify local CI and build.
5. Commit: `chore(release): prepare vX.Y.Z release`.
6. Push and merge PR into `main`.

### Step 2: Drafting & Publishing GitHub Release
1. Write release notes to a temporary file `.tmp_release_notes.md` (never use inline `--notes "..."` CLI arguments on Windows).
2. Execute GitHub CLI:
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z - <Descriptive Title>" --notes-file .tmp_release_notes.md
   ```
   *(Add `--prerelease` if targeting TestPyPI for staging verification).*
3. Clean up `.tmp_release_notes.md`.
4. Monitor the automated PyPI publication workflow:
   ```bash
   gh run list --workflow=release.yml --limit 1
   gh run watch <run-id> --exit-status
   ```
5. Confirm package availability on PyPI: `https://pypi.org/p/stride-xai`.
