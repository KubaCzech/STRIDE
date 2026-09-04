# Streamlit Dashboard Standards & Guidelines

> [!IMPORTANT]
> **Trigger Paths**: `dashboard/**`, `streamlit_app.py`
> **When to Read**: MUST be read before creating, editing, or refactoring dashboard UI components, widgets, or state logic.
> **Associated Skill**: Read [`.agents/skills/developing-with-streamlit/SKILL.md`](../skills/developing-with-streamlit/SKILL.md) for deep reference documentation.

## 1. Core Principles & Stack

1. **State Isolation**: When storing model parameters, dataset settings, or analysis artifacts in `st.session_state`, namespace them under model/dataset keys (e.g. `f"{model_name}_{param_name}"`) to prevent parameter leakage when the user switches models or datasets.
2. **Modern Layout**: Use `width="stretch"` instead of the deprecated `use_container_width`.
3. **Container Hierarchy**: Prefer `st.container(border=True)` for visual card grouping and `st.container(horizontal=True)` for responsive horizontal arrangements.
4. **Performance Gating**: Wrap expensive drift simulations, decision boundary grids, or SHAP calculations in cached functions (`@st.cache_data`) or isolate them in `@st.fragment`.

## 2. Declarative Code Standards (Golden Patterns)

```python
# Isolating session state keys per model/dataset
def get_model_param(model_type: str, param_name: str, default: any):
    key = f"model_{model_type}_{param_name}"
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Modern layout sizing
st.dataframe(df, width="stretch")
st.altair_chart(chart, width="stretch")
```

---

## 3. Anti-Pattern & Pitfall Traps

| Anti-Pattern Trap | Why It Fails | Golden Pattern |
| :--- | :--- | :--- |
| **`use_container_width=True`** | Deprecated in modern Streamlit; raises warnings or breaks in future releases. | Use `width="stretch"` or omit for default container-stretching behavior. |
| **Shared Flat Session State Keys** (e.g. `st.session_state["learning_rate"]`) | Switching between models (e.g. MLP to LogisticRegression) causes invalid parameters or resets. | Prefix keys by model type: `st.session_state[f"{model}_{param}"]`. |
| **Unguarded Heavy Computations in Tabs** | Code inside inactive `st.tabs` runs on every full script rerun by default. | Gate computation behind button callbacks, `st.form`, or dynamic tab state checks. |
| **Custom CSS Injections for Basic Spacing** | Fragile across theme updates and breaks responsiveness. | Use native Streamlit containers: `st.container(border=True)`, `st.columns`, and theme tokens. |
| **Direct Mutation of Stream Data in UI Callbacks** | Unintended state resets when other unrelated widgets trigger a rerun. | Treat dataset arrays as immutable copies or store in dedicated session cache. |
