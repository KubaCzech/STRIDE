# Contributing Guidelines

We welcome community and academic contributions to STRIDE! To ensure reproducibility, stability, and high code quality, please adhere to these guidelines.

---

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/STRIDE.git
   cd STRIDE
   ```
2. Create and activate a virtual environment (Python 3.10+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install package and developer dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e ".[all,docs,dev]"
   ```

---

## Local Verification Gates

Before submitting any Pull Request, you must verify that all local checks pass:

```bash
# 1. Lint code with Ruff
ruff check .

# 2. Check code formatting
ruff format --check .

# 3. Run unit tests
python -m unittest discover tests

# 4. Validate documentation builds with zero warnings
mkdocs build --strict
```

---

## Git Workflow & Branching

- Always base feature branches off `origin/main`:
  ```bash
  git checkout -b feat/your-feature-name origin/main
  ```
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Keep commits atomic: single purpose, passing tests at each checkpoint.
