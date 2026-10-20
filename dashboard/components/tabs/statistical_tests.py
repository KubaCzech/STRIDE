import traceback
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType

DRIFT_STYLE = "background-color: rgba(231, 76, 60, 0.25)"

# Tests offered in the dashboard. Left out: Spearman rank correlation, which correlates the
# samples of two independent windows paired by position and is not a two-sample test, and
# Kullback-Leibler divergence, whose histogram estimate explodes on empty bins and flags
# most identically distributed windows at the default threshold.
TESTS = {
    "kolmogorov_smirnov": {
        "type": StatisticalTestType.KolmogorovSmirnov,
        "name": "Kolmogorov–Smirnov",
        "short": "KS p-value",
        "key": "p_value",
        "hypothesis": True,
        "description": "Largest vertical distance between the empirical CDFs; sensitive to any change of the distribution.",
    },
    "anderson_darling": {
        "type": StatisticalTestType.AD,
        "name": "Anderson–Darling",
        "short": "AD p-value",
        "key": "p_value",
        "hypothesis": True,
        "description": "Weighted CDF distance, more sensitive to the tails than KS. SciPy clips its p-value to [0.001, 0.25].",
    },
    "wasserstein": {
        "type": StatisticalTestType.WassersteinDistance,
        "name": "Wasserstein distance",
        "short": "Wasserstein",
        "key": "wd",
        "hypothesis": False,
        "threshold_attr": "wasserstein_thr",
        "description": "Earth mover's distance between the distributions of min-max scaled features.",
    },
    "jensen_shannon": {
        "type": StatisticalTestType.JensenShannon,
        "name": "Jensen–Shannon divergence",
        "short": "JS divergence",
        "key": "js_div",
        "hypothesis": False,
        "threshold_attr": "js_thr",
        "description": "Symmetric divergence between histograms of min-max scaled features (0 = identical).",
    },
}


def _class_name(label) -> str:
    """Readable class label: integral numbers without decimals, anything else as text."""
    if isinstance(label, (bool, np.bool_)):
        return str(label)
    if isinstance(label, (int, float, np.integer, np.floating)) and float(label).is_integer():
        return str(int(label))
    return str(label)


def _display_intro():
    st.header("Statistical Tests")
    st.markdown(
        """
    This tab shows **which features** changed between the **Before** and **After** windows;
    whether and when drift occurs is the job of the drift detectors (see the
    *Drift Detection* tab).

    Two-sample tests and distribution distances compare the distribution of every feature
    between the two windows, separately within each class ($P(X_j \\mid Y = y)$).
    Hypothesis tests (Kolmogorov–Smirnov, Anderson–Darling) flag a feature when
    $p < \\alpha$; distances (Wasserstein, Jensen–Shannon) flag it when they exceed a threshold.
    """
    )


def _render_config() -> dict:
    """Renders the test settings and returns them as keyword arguments."""
    with st.expander("Test Configuration", expanded=False):
        tests = st.multiselect(
            "Tests",
            options=list(TESTS),
            default=list(TESTS),
            format_func=lambda t: TESTS[t]["name"],
            help="Tests and distances to compute.",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.select_slider(
                "Significance level (α)",
                options=[0.001, 0.01, 0.05, 0.1],
                value=0.05,
                help="A hypothesis test flags a feature when its p-value is below α.",
            )
        with col2:
            wasserstein_thr = st.number_input(
                "Wasserstein threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f"
            )
        with col3:
            js_thr = st.number_input(
                "Jensen–Shannon threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f"
            )
    return {
        "tests": tests,
        "alpha": float(alpha),
        "wasserstein_thr": float(wasserstein_thr),
        "js_thr": float(js_thr),
    }


@st.cache_data(show_spinner=False)
def _run_tests(X_before, y_before, X_after, y_after, settings: dict) -> StatisticalTestsDriftDetector:
    """Run the selected tests once per pair of windows and settings."""
    params = {k: v for k, v in settings.items() if k != "tests"}
    detector = StatisticalTestsDriftDetector(X_before, y_before, X_after, y_after, **params)
    with warnings.catch_warnings():
        # SciPy warns whenever it clips the Anderson-Darling p-value and about a deprecated argument.
        warnings.simplefilter("ignore")
        detector.detect([TESTS[t]["type"] for t in settings["tests"]])
    return detector


def _evaluated_pairs(detector: StatisticalTestsDriftDetector, test: str) -> list[dict]:
    """Per-pair results of one test, for classes present in both windows."""
    key = TESTS[test]["key"]
    return [d for per_class in detector.drift_details[test].values() for d in per_class.values() if d[key] is not None]


def _tests_summary_dataframe(detector: StatisticalTestsDriftDetector) -> pd.DataFrame:
    """One row per test: criterion and flagged feature–class pairs."""
    rows = []
    for test in detector.drift_details:
        info = TESTS[test]
        pairs = _evaluated_pairs(detector, test)
        flagged = sum(d["drift"] for d in pairs)
        if info["hypothesis"]:
            criterion = f"p < {detector.alpha:g}"
        else:
            criterion = f"> {getattr(detector, info['threshold_attr']):g}"
        rows.append(
            {
                "Test": info["name"],
                "Criterion": criterion,
                "Flagged Pairs": f"{flagged}/{len(pairs)}",
                "Share": flagged / len(pairs) if pairs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _feature_results(detector: StatisticalTestsDriftDetector) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Values and drift flags per (feature, class) pair and test."""
    values, flags = {}, {}
    for test, per_class in detector.drift_details.items():
        info = TESTS[test]
        for cl, per_feature in per_class.items():
            for feature, d in per_feature.items():
                row = (str(feature), _class_name(cl))
                values.setdefault(row, {})[info["short"]] = d[info["key"]]
                flags.setdefault(row, {})[info["short"]] = bool(d["drift"])
                if test == "kolmogorov_smirnov":
                    values[row]["KS statistic"] = d["stat"]
                    flags[row]["KS statistic"] = False

    columns = [c for t in TESTS for c in ([TESTS[t]["short"]] + (["KS statistic"] if t == "kolmogorov_smirnov" else []))]
    index = pd.MultiIndex.from_tuples(list(values), names=["Feature", "Class"])
    values_df = pd.DataFrame(list(values.values()), index=index).reindex(columns=columns).dropna(axis=1, how="all")
    flags_df = pd.DataFrame(list(flags.values()), index=index).reindex(columns=values_df.columns, fill_value=False)
    return values_df.astype(float), flags_df.astype(bool)


def _format_p(v: float) -> str:
    if pd.isna(v):
        return "–"
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def _display_tests_table(detector: StatisticalTestsDriftDetector):
    st.subheader("Results by Test")
    st.dataframe(
        _tests_summary_dataframe(detector),
        column_config={
            "Test": st.column_config.TextColumn("Test"),
            "Criterion": st.column_config.TextColumn("Criterion", help="Condition under which a pair is flagged"),
            "Flagged Pairs": st.column_config.TextColumn("Flagged Pairs", help="Flagged / evaluated feature–class pairs"),
            "Share": st.column_config.ProgressColumn(
                "Share",
                help="Share of the evaluated feature–class pairs flagged by the test.",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Every feature–class pair is tested separately, so even without any change a hypothesis test flags about "
        "a share α of the pairs by chance."
    )


def _display_feature_table(detector: StatisticalTestsDriftDetector):
    st.subheader("Results by Feature and Class")
    values, flags = _feature_results(detector)

    only_flagged = st.checkbox("Show only flagged features", value=False)
    if only_flagged:
        keep = flags.any(axis=1)
        values, flags = values[keep], flags[keep]
        if values.empty:
            st.info("No feature–class pair is flagged by any test.")
            return

    p_columns = [c for c in values.columns if c.endswith("p-value")]
    other_columns = [c for c in values.columns if c not in p_columns]
    styled = (
        values.style.apply(lambda _: flags.map(lambda f: DRIFT_STYLE if f else ""), axis=None)
        .format(_format_p, subset=p_columns)
        .format("{:.3f}", subset=other_columns, na_rep="–")
    )
    st.dataframe(styled, width="stretch")
    st.caption(
        "Highlighted: the pair is flagged by that test. The KS statistic (0–1) measures the size of the change and "
        "stays comparable when p-values are all close to zero."
    )

    with st.expander("About the tests"):
        for test in detector.drift_details:
            st.markdown(f"**{TESTS[test]['name']}**: {TESTS[test]['description']}")


def render_statistical_tests_tab(X_before, y_before, X_after, y_after, feature_names):
    """Renders the Statistical Tests tab.

    Args:
        X_before: Feature matrix for 'before' window.
        y_before: Target variable for 'before' window.
        X_after: Feature matrix for 'after' window.
        y_after: Target variable for 'after' window.
        feature_names: Selected feature names.
    """
    _display_intro()
    settings = _render_config()
    if not settings["tests"]:
        st.warning("Select at least one test in the Test Configuration.")
        return

    with st.spinner("Running statistical tests..."):
        try:
            detector = _run_tests(X_before, y_before, X_after, y_after, settings)
            _display_tests_table(detector)
            _display_feature_table(detector)
        except Exception as e:
            st.error(f"An error occurred while running the statistical tests: {e}")
            st.text(traceback.format_exc())
