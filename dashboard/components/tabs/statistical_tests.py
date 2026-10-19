import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType
from stride.xai.stats.visualization import (
    PlotOptions,
    plot_boxplot,
    plot_ecdf,
    plot_histogram,
    plot_kde,
    plot_qq,
)

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

PLOTS = {
    "ECDF": (plot_ecdf, {}),
    "Histogram": (plot_histogram, {}),
    "KDE": (plot_kde, {}),
    "Boxplot": (plot_boxplot, {"show_": PlotOptions.Both}),
    "Q-Q plot": (plot_qq, {}),
}

PLOT_CAPTIONS = {
    "ECDF": "Empirical cumulative distribution functions. The KS statistic is the largest vertical gap between the two curves.",
    "Histogram": "Overlapping histograms of both windows on common bins.",
    "KDE": "Kernel density estimates of both windows.",
    "Boxplot": "Quartiles, whiskers, median (solid) and mean (dashed) of both windows.",
    "Q-Q plot": (
        "Quantiles of the After window against the same quantiles of the Before window. Points on the diagonal mean "
        "equal distributions; a parallel offset is a shift, a different slope a change of spread."
    ),
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
    Two-sample tests and distribution distances compare the distribution of every feature
    between the **Before** and **After** windows, separately within each class
    ($P(X_j \\mid Y = y)$). Hypothesis tests (Kolmogorov–Smirnov, Anderson–Darling) flag a
    feature when $p < \\alpha$; distances (Wasserstein, Jensen–Shannon) flag
    it when they exceed a threshold. A test signals drift when the share of flagged
    feature–class pairs exceeds the pair threshold, and drift is reported when the share of
    tests signalling drift exceeds the decision threshold.
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
            wasserstein_thr = st.number_input(
                "Wasserstein threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f"
            )
        with col2:
            js_thr = st.number_input(
                "Jensen–Shannon threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f"
            )
        with col3:
            drift_thr = st.slider(
                "Pair threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="A test signals drift when more than this share of feature–class pairs is flagged.",
            )
            decision_thr = st.slider(
                "Decision threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Drift is reported when more than this share of the selected tests signals drift.",
            )
    return {
        "tests": tests,
        "alpha": float(alpha),
        "wasserstein_thr": float(wasserstein_thr),
        "js_thr": float(js_thr),
        "drift_thr": float(drift_thr),
        "decision_thr": float(decision_thr),
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
    """One row per test: criterion, flagged pairs, expected false alarms and the test-level decision."""
    rows = []
    for test in detector.drift_details:
        info = TESTS[test]
        pairs = _evaluated_pairs(detector, test)
        flagged = sum(d["drift"] for d in pairs)
        if info["hypothesis"]:
            criterion = f"p < {detector.alpha:g}"
            expected = detector.alpha * len(pairs)
        else:
            criterion = f"> {getattr(detector, info['threshold_attr']):g}"
            expected = None
        rows.append(
            {
                "Test": info["name"],
                "Criterion": criterion,
                "Flagged Pairs": f"{flagged}/{len(pairs)}",
                "Share": flagged / len(pairs) if pairs else 0.0,
                "Expected by Chance": expected,
                "Signals Drift": bool(detector.drift_flags[test]),
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


def _display_summary(detector: StatisticalTestsDriftDetector):
    n_tests = len(detector.drift_flags)
    n_drift = sum(detector.drift_flags.values())
    if detector.drift_flag:
        st.success(f"**Drift detected**: {n_drift} of {n_tests} tests signal drift.")
    else:
        st.info(f"**No drift detected**: {n_drift} of {n_tests} tests signal drift.")

    _, flags = _feature_results(detector)
    flagged_pairs = int(flags.any(axis=1).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Drift Detected", "Yes" if detector.drift_flag else "No")
    col2.metric(
        "Tests Signalling Drift",
        f"{n_drift}/{n_tests}",
        help=f"Drift is reported when more than {detector.decision_thr:.0%} of the tests signal drift.",
    )
    col3.metric(
        "Feature–Class Pairs Flagged",
        f"{flagged_pairs}/{len(flags)}",
        help="Pairs flagged by at least one test.",
    )


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
                help=f"The test signals drift when the share exceeds {detector.drift_thr:.0%}.",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "Expected by Chance": st.column_config.NumberColumn(
                "Expected by Chance",
                help="Pairs a hypothesis test flags on average without any drift (α × evaluated pairs).",
                format="%.1f",
            ),
            "Signals Drift": st.column_config.CheckboxColumn("Signals Drift"),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Every feature–class pair is tested separately, so with many pairs some p-values fall below α by chance; "
        "compare the flagged count with the count expected by chance."
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


def _default_feature(detector: StatisticalTestsDriftDetector, features: list) -> int:
    """Index of the feature with the largest KS statistic (or the first feature without KS)."""
    if "kolmogorov_smirnov" not in detector.drift_details:
        return 0
    values, _ = _feature_results(detector)
    ks = values["KS statistic"].groupby(level="Feature").max()
    return features.index(ks.idxmax()) if ks.notna().any() else 0


def _display_distribution_plot(X_before, y_before, X_after, y_after, detector: StatisticalTestsDriftDetector):
    st.subheader("Feature Distributions")
    features = [str(c) for c in X_before.columns]
    col_feature, col_plot = st.columns([1, 2])
    with col_feature:
        feature = st.selectbox(
            "Feature",
            options=features,
            index=_default_feature(detector, features),
            key="stats_feature_selector",
            help="Defaults to the feature with the largest KS statistic.",
        )
    with col_plot:
        plot_name = st.radio("Plot", options=list(PLOTS), horizontal=True, key="stats_plot_selector")

    column = X_before.columns[features.index(feature)]
    plot_func, kwargs = PLOTS[plot_name]
    try:
        fig = plot_func(X_before[[column]], y_before, X_after[[column]], y_after, **kwargs)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(PLOT_CAPTIONS[plot_name])
    except Exception as e:
        plt.close("all")
        st.warning(f"The {plot_name} cannot be drawn for feature '{feature}': {e}")


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
            _display_summary(detector)
            _display_tests_table(detector)
            _display_feature_table(detector)
            _display_distribution_plot(X_before, y_before, X_after, y_after, detector)
        except Exception as e:
            st.error(f"An error occurred while running the statistical tests: {e}")
            st.text(traceback.format_exc())
