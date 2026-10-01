"""Marketing performance explorer.

Upload a campaign CSV, check the data quality, see key numbers and charts, and
find out whether a machine-learning model can predict conversion rate, revenue
or ROI better than simply guessing the average.

Run with:  streamlit run app.py
"""

import hashlib
import io
import math
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Must be the first Streamlit command.
st.set_page_config(
    page_title="Marketing performance explorer",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Clicks",
    "Conversions",
    "Budget",
    "Revenue_Generated",
    "Units_Sold",
    "ROI",
]
# Columns that cannot be negative (ROI can be, when a campaign loses money).
NON_NEGATIVE_COLUMNS = ["Clicks", "Conversions", "Budget", "Revenue_Generated", "Units_Sold"]
# Columns the app calculates itself. They are never used as model inputs.
DERIVED_COLUMNS = ["Conversions_rate", "A_O_V"]

TARGETS = {
    "Conversion rate": {"col": "Conversions_rate", "unit": "%", "err_unit": " percentage points"},
    "Revenue": {"col": "Revenue_Generated", "unit": "", "err_unit": ""},
    "ROI": {"col": "ROI", "unit": "", "err_unit": ""},
}

BASELINE = "Baseline (always guess the average)"
MODEL_NAMES = ["Linear regression", "Random forest", "Gradient boosting"]
N_FOLDS = 5
MIN_ROWS = 50
MAX_CATEGORIES = 30
RANDOM_STATE = 42

PRETTY_NAMES = {
    "A_O_V": "Average order value",
    "Conversions_rate": "Conversion rate (%)",
    "Revenue_Generated": "Revenue",
    "Units_Sold": "Units sold",
    "ROI": "ROI",
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def nice(col: str) -> str:
    """Turn a column name into a short, readable label in sentence case."""
    if col in PRETTY_NAMES:
        return PRETTY_NAMES[col]
    text = str(col).replace("_", " ").strip()
    if text.isupper():
        return text
    return text[:1].upper() + text[1:].lower()


def plain(label: str) -> str:
    """Lowercase a label for use inside a sentence, but keep acronyms like ROI."""
    return label if label.isupper() else label.lower()


def fmt(value, unit: str = "") -> str:
    """Format a number for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"
    return f"{text}{unit}"


def compact(value: float) -> str:
    """Short format for big headline numbers (1.2M, 34.5K)."""
    size = abs(value)
    if size >= 1e9:
        return f"{value / 1e9:.2f}B"
    if size >= 1e6:
        return f"{value / 1e6:.2f}M"
    if size >= 1e4:
        return f"{value / 1e3:.1f}K"
    return f"{value:,.2f}"


def find_id_columns(df: pd.DataFrame) -> list:
    """Columns that only identify a row (Campaign_ID, Customer_ID, ...)."""
    ids = []
    for col in df.columns:
        if re.search(r"(^|[_\s])id$", str(col), flags=re.IGNORECASE):
            ids.append(col)
        elif (
            not is_numeric_dtype(df[col])
            and len(df) > 20
            and df[col].nunique(dropna=True) > 0.9 * len(df)
        ):
            ids.append(col)  # text column that is almost unique in every row
    return ids


# ----------------------------------------------------------------------------
# Data loading and cleaning (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read the uploaded CSV. Tries UTF-8 first, then Latin-1, and guesses the separator."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            if frame.shape[1] == 1:  # probably ';' or tab separated
                frame = pd.read_csv(
                    io.BytesIO(file_bytes), encoding=encoding, sep=None, engine="python"
                )
            frame.columns = [str(c).strip() for c in frame.columns]
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError("The file's text encoding is not supported.")


@st.cache_data(show_spinner=False)
def clean_data(raw: pd.DataFrame):
    """Remove impossible rows and add the KPI columns. Returns (clean data, report)."""
    df = raw.copy()
    report = {"uploaded": len(df), "id_cols": find_id_columns(df)}

    # 1. Make the key columns numeric. Text such as "n/a" becomes missing.
    unreadable = 0
    for col in REQUIRED_COLUMNS:
        before = int(df[col].notna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        unreadable += before - int(df[col].notna().sum())
    report["unreadable_cells"] = unreadable

    # 2. Tidy text columns (trim spaces, consistent capitals).
    for col in df.columns:
        if col in report["id_cols"] or is_numeric_dtype(df[col]) or is_datetime64_any_dtype(df[col]):
            continue
        df[col] = (
            df[col]
            .astype("object")
            .map(lambda v: (v.strip().title() or np.nan) if isinstance(v, str) else v)
        )

    # 3. Remove rows, one rule at a time, so the counts add up.
    n = len(df)
    df = df.drop_duplicates()
    report["duplicates"] = n - len(df)

    n = len(df)
    df = df.loc[~(df["Conversions"] > df["Clicks"])]
    report["conv_over_clicks"] = n - len(df)

    n = len(df)
    df = df.loc[~(df[NON_NEGATIVE_COLUMNS] < 0).any(axis=1)]
    report["negative"] = n - len(df)

    df = df.reset_index(drop=True)
    report["kept"] = len(df)
    report["removed"] = report["uploaded"] - report["kept"]

    # 4. KPI columns. Division by zero gives "missing" instead of infinity.
    clicks = df["Clicks"].where(df["Clicks"] > 0)
    units = df["Units_Sold"].where(df["Units_Sold"] > 0)
    df["Conversions_rate"] = df["Conversions"] / clicks * 100
    df["A_O_V"] = df["Revenue_Generated"] / units
    report["zero_clicks"] = int(df["Clicks"].le(0).sum())
    return df, report


# ----------------------------------------------------------------------------
# Choosing which columns the model may use
# ----------------------------------------------------------------------------
def roi_is_calculated_from_revenue(df: pd.DataFrame) -> bool:
    """True if ROI is (almost exactly) a formula of Revenue and Budget."""
    part = df[["Revenue_Generated", "Budget", "ROI"]].dropna()
    part = part[part["Budget"] > 0]
    if len(part) < 30:
        return False
    revenue, budget = part["Revenue_Generated"], part["Budget"]
    for guess in ((revenue - budget) / budget, revenue / budget):
        if guess.corr(part["ROI"]) > 0.99:
            return True
    return False


def feature_candidates(df: pd.DataFrame, target_col: str, id_cols: list):
    """Return (usable columns, {excluded column: reason}). Removes IDs and data leakage."""
    excluded = {}
    for col in id_cols:
        excluded[col] = "ID column. It labels a row but does not explain results."
    for col in DERIVED_COLUMNS:
        if col in df.columns and col != target_col:
            excluded[col] = "Calculated from other columns, so you would not know it in advance."

    if target_col == "Conversions_rate":
        for col in ("Clicks", "Conversions"):
            excluded[col] = "The conversion rate is calculated from this column (data leakage)."
    if roi_is_calculated_from_revenue(df):
        if target_col == "Revenue_Generated":
            excluded["ROI"] = "ROI is calculated from revenue and budget (data leakage)."
        if target_col == "ROI":
            excluded["Revenue_Generated"] = "ROI is calculated from revenue and budget (data leakage)."

    for col in df.columns:
        if col == target_col or col in excluded:
            continue
        series = df[col]
        if series.notna().sum() == 0:
            excluded[col] = "The column is empty."
        elif series.nunique(dropna=True) <= 1:
            excluded[col] = "Same value in every row."
        elif is_numeric_dtype(series):
            corr = series.corr(df[target_col])
            if pd.notna(corr) and abs(corr) > 0.95:
                excluded[col] = "Almost identical to the target (correlation above 0.95)."
        elif is_datetime64_any_dtype(series):
            excluded[col] = "Date column. Not used by these models."
        elif series.nunique(dropna=True) > MAX_CATEGORIES:
            excluded[col] = f"More than {MAX_CATEGORIES} different values."

    candidates = [c for c in df.columns if c != target_col and c not in excluded]
    return candidates, excluded


def build_model_frame(df: pd.DataFrame, target_col: str, features: list) -> pd.DataFrame:
    """Rows the model can learn from: features plus a known target."""
    data = df[list(features) + [target_col]].copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in features:
        if not is_numeric_dtype(data[col]):
            data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
    return data.dropna(subset=[target_col]).reset_index(drop=True)


def split_types(frame: pd.DataFrame, features) -> tuple:
    numeric = [c for c in features if is_numeric_dtype(frame[c])]
    text = [c for c in features if c not in numeric]
    return numeric, text


# ----------------------------------------------------------------------------
# Models (preprocessing and model live in one Pipeline)
# ----------------------------------------------------------------------------
def make_estimator(name: str):
    if name == BASELINE:
        return DummyRegressor(strategy="mean")
    if name == "Linear regression":
        return LinearRegression()
    if name == "Random forest":
        return RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        )
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, numeric_cols: list, text_cols: list) -> Pipeline:
    steps = []
    if numeric_cols:
        steps.append(
            (
                "numbers",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if text_cols:
        steps.append(
            (
                "categories",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                text_cols,
            )
        )
    return Pipeline([("prepare", ColumnTransformer(steps)), ("model", make_estimator(name))])


@st.cache_data(show_spinner=False)
def run_cross_validation(data: pd.DataFrame, target_col: str, features: tuple) -> pd.DataFrame:
    """5-fold cross-validation for the baseline and every model, on identical folds."""
    X, y = data[list(features)], data[target_col]
    numeric_cols, text_cols = split_types(X, features)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in [BASELINE] + MODEL_NAMES:
        scores = cross_validate(
            build_pipeline(name, numeric_cols, text_cols),
            X,
            y,
            cv=folds,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
        )
        mae_folds = [float(-v) for v in scores["test_mae"]]
        rows.append(
            {
                "model": name,
                "r2": float(np.mean(scores["test_r2"])),
                "r2_std": float(np.std(scores["test_r2"])),
                "mae": float(np.mean(mae_folds)),
                "mae_folds": mae_folds,
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def fit_final_model(data: pd.DataFrame, target_col: str, features: tuple, model_name: str):
    """Fit one model on all rows, for the Predict tab."""
    X, y = data[list(features)], data[target_col]
    numeric_cols, text_cols = split_types(X, features)
    pipeline = build_pipeline(model_name, numeric_cols, text_cols)
    pipeline.fit(X, y)
    return pipeline


def judge(results: pd.DataFrame) -> dict:
    """Does the best model really beat the 'always guess the average' baseline?"""
    base = results.loc[results["model"] == BASELINE].iloc[0]
    best = results.loc[results["model"] != BASELINE].sort_values("mae").iloc[0]
    wins = sum(m < b for m, b in zip(best["mae_folds"], base["mae_folds"]))
    gain = (base["mae"] - best["mae"]) / base["mae"] * 100 if base["mae"] else 0.0
    if best["r2"] >= 0.10 and wins == N_FOLDS and gain >= 5:
        level = "clear"
    elif best["r2"] > 0 and gain > 0:
        level = "weak"
    else:
        level = "none"
    return {
        "level": level,
        "best": best["model"],
        "best_mae": best["mae"],
        "base_mae": base["mae"],
        "r2": best["r2"],
        "gain": gain,
        "wins": wins,
    }


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def bar_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str):
    summary = (
        df.groupby(group_col, dropna=True)[value_col]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    summary.columns = ["group", "value"]
    chart = (
        alt.Chart(summary)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("value:Q", title=f"Average {nice(value_col).lower()}"),
            y=alt.Y("group:N", sort="-x", title=nice(group_col)),
            tooltip=[
                alt.Tooltip("group:N", title=nice(group_col)),
                alt.Tooltip("value:Q", title="Average", format=",.2f"),
            ],
        )
        .properties(title=title, height=max(160, 28 * len(summary)))
    )
    st.altair_chart(chart, width="stretch")


def scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    points = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(points) > 3000:
        points = points.sample(3000, random_state=RANDOM_STATE)
    points.columns = ["x", "y"]
    chart = (
        alt.Chart(points)
        .mark_circle(size=25, opacity=0.35, color="#4C78A8")
        .encode(
            x=alt.X("x:Q", title=nice(x_col)),
            y=alt.Y("y:Q", title=nice(y_col)),
            tooltip=[
                alt.Tooltip("x:Q", title=nice(x_col), format=",.2f"),
                alt.Tooltip("y:Q", title=nice(y_col), format=",.2f"),
            ],
        )
        .properties(title=title, height=320)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def correlation_heatmap(df: pd.DataFrame, id_cols: list):
    numbers = df.drop(columns=[c for c in id_cols if c in df.columns]).select_dtypes(include="number")
    numbers = numbers.loc[:, numbers.nunique() > 1]
    if numbers.shape[1] < 2:
        st.info("Not enough number columns to draw a correlation heatmap.")
        return
    corr = numbers.corr().round(2)
    names = [nice(c) for c in corr.columns]
    corr.index, corr.columns = names, names
    long = corr.reset_index(names="row").melt(id_vars="row", var_name="column", value_name="r")
    base = alt.Chart(long).encode(
        x=alt.X("column:N", sort=names, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("row:N", sort=names, title=None),
    )
    cells = base.mark_rect().encode(
        color=alt.Color(
            "r:Q",
            scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            title="Correlation",
        ),
        tooltip=[
            alt.Tooltip("row:N", title="Column A"),
            alt.Tooltip("column:N", title="Column B"),
            alt.Tooltip("r:Q", title="Correlation", format=".2f"),
        ],
    )
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.5", alt.value("white"), alt.value("black")),
    )
    st.altair_chart(
        (cells + labels).properties(
            title="How strongly the number columns move together", height=max(320, 38 * len(names))
        ),
        width="stretch",
    )
    st.caption(
        "Values near +1 or -1 mean two columns move together closely. Values near 0 mean little link. "
        "A link does not prove that one column causes the other."
    )


# ----------------------------------------------------------------------------
# Page sections
# ----------------------------------------------------------------------------
def show_empty_state():
    st.title("Marketing performance explorer")
    st.info("Upload a CSV file in the sidebar to get started.")
    st.markdown(
        """
**What you can do here**

- Check how clean your campaign data is and see what was removed.
- See key numbers and charts for budget, revenue, ROI and conversions.
- Find out whether a model can predict conversion rate, revenue or ROI, and whether it
  does better than simply guessing the average.
- Try a what-if prediction with your own numbers.
        """
    )
    with st.expander("Which columns does my file need?"):
        st.markdown(
            "**Required:** " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS) + "\n\n"
            "**Optional (used if present):** `Subscription_Tier`, `Common_Keywords`, "
            "`Subscription_Length`, `Discount_Level` and any other number or short text columns. "
            "Columns that end in `_ID` are ignored by the models."
        )


def render_overview(raw: pd.DataFrame, df: pd.DataFrame, report: dict):
    st.subheader("Data overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows uploaded", f"{report['uploaded']:,}")
    share = report["removed"] / report["uploaded"] * 100 if report["uploaded"] else 0
    c2.metric("Rows removed", f"{report['removed']:,}", delta=f"{share:.1f}% of rows", delta_color="off")
    c3.metric("Rows kept", f"{report['kept']:,}")

    if report["removed"] > 0:
        st.warning(
            f"We removed {report['removed']:,} rows ({share:.1f}%) that cannot be right. "
            "Keeping them would distort every average and chart."
        )
    else:
        st.success("No impossible rows found. All rows were kept.")

    removed_table = pd.DataFrame(
        {
            "Reason": [
                "Exact duplicate rows",
                "More conversions than clicks",
                "Negative clicks, budget, conversions, revenue or units",
            ],
            "Rows removed": [report["duplicates"], report["conv_over_clicks"], report["negative"]],
        }
    )
    st.dataframe(removed_table, hide_index=True, width="stretch", key="removed_table")

    notes = []
    if report["unreadable_cells"]:
        notes.append(f"{report['unreadable_cells']:,} cells in key columns were not numbers and were treated as missing.")
    if report["zero_clicks"]:
        notes.append(f"{report['zero_clicks']:,} rows have zero clicks, so their conversion rate is left empty.")
    for note in notes:
        st.info(note)

    st.markdown("#### Data preview")
    st.dataframe(df.head(100), width="stretch", key="preview_table")
    st.caption("Showing the first 100 rows after cleaning.")

    st.markdown("#### Summary statistics")
    numbers = df.select_dtypes(include="number")
    st.dataframe(numbers.describe().T.round(2), width="stretch", key="summary_numbers")
    text_cols = [c for c in df.columns if not is_numeric_dtype(df[c]) and c not in report["id_cols"]]
    if text_cols:
        with st.expander("Text columns"):
            st.dataframe(df[text_cols].describe().T, width="stretch", key="summary_text")
    with st.expander("Missing values per column"):
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.write("No missing values.")
        else:
            st.dataframe(
                missing.rename("Missing cells").to_frame(), width="stretch", key="summary_missing"
            )


def render_kpis(df: pd.DataFrame, report: dict):
    st.subheader("Key metrics")
    total_clicks = df["Clicks"].sum()
    total_units = df["Units_Sold"].sum()
    conversion = df["Conversions"].sum() / total_clicks * 100 if total_clicks else float("nan")
    order_value = df["Revenue_Generated"].sum() / total_units if total_units else float("nan")

    row1 = st.columns(3)
    row1[0].metric("Total budget", compact(df["Budget"].sum()))
    row1[1].metric("Total revenue", compact(df["Revenue_Generated"].sum()))
    row1[2].metric("Average ROI", fmt(df["ROI"].mean()))
    row2 = st.columns(3)
    row2[0].metric("Conversion rate (all clicks)", fmt(conversion, "%"))
    row2[1].metric("Average order value", fmt(order_value))
    row2[2].metric("Campaigns", f"{len(df):,}")

    st.subheader("Charts")
    text_cols = [c for c in df.columns if not is_numeric_dtype(df[c]) and c not in report["id_cols"]]
    tier_col = "Subscription_Tier" if "Subscription_Tier" in df.columns else (text_cols[0] if text_cols else None)
    other = [c for c in text_cols if c != tier_col]
    keyword_col = "Common_Keywords" if "Common_Keywords" in df.columns else (other[0] if other else None)

    left, right = st.columns(2)
    with left:
        if tier_col:
            bar_by_group(df, tier_col, "ROI", f"Average ROI by {nice(tier_col).lower()}")
        else:
            st.info("No text category column found, so the ROI-by-group chart is skipped.")
    with right:
        if keyword_col:
            bar_by_group(
                df, keyword_col, "Revenue_Generated", f"Average revenue by {nice(keyword_col).lower()}"
            )
        else:
            st.info("No other text category column found, so the revenue-by-group chart is skipped.")

    scatter_chart(df, "Budget", "ROI", "Does a bigger budget bring a better ROI?")
    correlation_heatmap(df, report["id_cols"])


def render_models(df, target_label, target_cfg, features, excluded, sig):
    target_col = target_cfg["col"]
    st.subheader(f"Can a model predict {plain(target_label)}?")
    st.write(
        "We test three models with 5-fold cross-validation. The data is split into five parts, and each "
        "part is used once as a test the model has never seen. A baseline that always guesses the "
        "average shows the score a model has to beat."
    )

    if not features:
        st.error("No columns are selected for the model. Pick at least one in the sidebar under 'Columns the model can use'.")
        return
    data = build_model_frame(df, target_col, features)
    skipped = len(df) - len(data)
    if skipped:
        st.caption(f"{skipped:,} rows were left out because {plain(target_label)} could not be calculated for them (for example, zero clicks).")
    if len(data) < MIN_ROWS:
        st.error(f"Only {len(data)} usable rows are left, and at least {MIN_ROWS} are needed. Upload a larger file or choose a different target.")
        return

    with st.expander("Columns used and left out"):
        st.write("**Used by the models:** " + ", ".join(nice(c) for c in features))
        if excluded:
            st.write("**Left out:**")
            st.dataframe(
                pd.DataFrame({"Column": [nice(c) for c in excluded], "Why": list(excluded.values())}),
                hide_index=True,
                width="stretch",
                key="excluded_table",
            )

    if st.button("Run model comparison", key="run_cv", type="primary"):
        try:
            with st.spinner("Testing the models. This can take up to a minute for large files..."):
                results = run_cross_validation(data, target_col, tuple(features))
            st.session_state["cv"] = {"sig": sig, "results": results}
        except Exception as err:  # noqa: BLE001 - show a friendly message instead of a traceback
            st.error(
                "The model test failed. Try removing text columns with many different values in the "
                f"sidebar, or upload a cleaner file. Details: {err}"
            )
            return

    saved = st.session_state.get("cv")
    if not saved or saved["sig"] != sig:
        st.info("Click **Run model comparison** to see the results for the current settings.")
        return

    results = saved["results"]
    verdict = judge(results)
    err_unit = target_cfg["err_unit"]
    if verdict["level"] == "clear":
        st.success(
            f"Yes, a model beats the baseline. {verdict['best']} misses by {fmt(verdict['best_mae'], err_unit)} on "
            f"average, against {fmt(verdict['base_mae'], err_unit)} for guessing the average "
            f"({verdict['gain']:.0f}% smaller), and it won in all {N_FOLDS} test rounds."
        )
    elif verdict["level"] == "weak":
        st.warning(
            f"Only a small edge. {verdict['best']} is a little better than guessing the average "
            f"({verdict['gain']:.0f}% smaller misses, R² {verdict['r2']:.2f}) but it is not reliable "
            f"(it won {verdict['wins']} of {N_FOLDS} rounds). Use its predictions as rough hints only."
        )
    else:
        st.warning(
            f"No. None of the models beat the baseline. Guessing the average works about as well, so these "
            f"columns do not predict {plain(target_label)} in this data. Try another target, different "
            "columns, or more data."
        )

    best_row = results.loc[results["model"] == verdict["best"]].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", verdict["best"])
    m2.metric("Its R² (higher is better)", f"{best_row['r2']:.2f}")
    m3.metric("Average miss: model vs baseline", fmt(verdict["best_mae"]), delta=f"baseline {fmt(verdict['base_mae'])}", delta_color="off")

    base_folds = results.loc[results["model"] == BASELINE].iloc[0]["mae_folds"]
    table = pd.DataFrame(
        {
            "Model": results["model"],
            "Average miss (lower is better)": results["mae"].round(3),
            "R² (higher is better)": results["r2"].round(3),
            "R² spread across rounds": results["r2_std"].round(3),
            "Beats baseline in": [
                "-" if m == BASELINE else f"{sum(a < b for a, b in zip(f, base_folds))} of {N_FOLDS} rounds"
                for m, f in zip(results["model"], results["mae_folds"])
            ],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch", key="cv_table")

    chart_data = pd.DataFrame(
        {
            "model": results["model"],
            "miss": results["mae"],
            "kind": np.where(results["model"] == BASELINE, "Baseline", "Model"),
        }
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("miss:Q", title=f"Average miss{err_unit} (lower is better)"),
            y=alt.Y("model:N", sort="x", title=None),
            color=alt.Color(
                "kind:N", scale=alt.Scale(domain=["Baseline", "Model"], range=["#9AA0A6", "#4C78A8"]), legend=None
            ),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("miss:Q", title="Average miss", format=",.3f")],
        )
        .properties(title="How far off each model is on unseen data", height=200)
    )
    st.altair_chart(chart, width="stretch")

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Average miss** is how far the prediction is from the real value, on average.\n"
            "- **R²** is the share of the ups and downs the model explains. 0 means no better than the average, "
            "1 means perfect.\n"
            "- **Beats baseline in** counts the test rounds where the model's average miss was smaller than the baseline's."
        )


def numeric_input(label: str, series: pd.Series, key: str):
    """Number box limited to the range seen in the data."""
    values = series.dropna()
    low, high, middle = values.min(), values.max(), values.median()
    if is_numeric_dtype(series) and bool((values % 1 == 0).all()):
        low, high = int(low), int(high)
        return st.number_input(label, min_value=low, max_value=high, value=int(np.clip(round(middle), low, high)), step=1, key=key)
    low, high, middle = float(low), float(high), float(middle)
    span = high - low
    step = 10 ** math.floor(math.log10(span / 100)) if span > 0 else 1.0
    return st.number_input(label, min_value=low, max_value=high, value=middle, step=float(step), format="%.2f", key=key)


def render_predict(df, target_label, target_cfg, features, sig, file_id):
    target_col, unit = target_cfg["col"], target_cfg["unit"]
    st.subheader(f"Predict {plain(target_label)}")
    if not features:
        st.error("No columns are selected for the model. Pick at least one in the sidebar under 'Columns the model can use'.")
        return
    data = build_model_frame(df, target_col, features)
    if len(data) < MIN_ROWS:
        st.error(f"Only {len(data)} usable rows are left, and at least {MIN_ROWS} are needed to build a model.")
        return

    saved = st.session_state.get("cv")
    verdict = judge(saved["results"]) if saved and saved["sig"] == sig else None
    default_model = verdict["best"] if verdict else "Random forest"
    model_name = st.selectbox(
        "Model to use",
        MODEL_NAMES,
        index=MODEL_NAMES.index(default_model),
        key=f"predict_model_{file_id}_{target_col}_{'tested' if verdict else 'untested'}",
        help="Defaults to the best model from the comparison tab.",
    )
    if verdict is None:
        st.info("You have not tested the models yet. Run the comparison first to learn how far you can trust a prediction.")
    elif verdict["level"] == "clear":
        st.success("The model comparison shows these models beat the baseline, so predictions are meaningful.")
    else:
        st.warning("These models did not clearly beat the baseline in the comparison. Treat any prediction as a rough guess.")

    st.caption("Fill in the details below. Values are limited to the range found in your data.")
    with st.form(key=f"predict_form_{file_id}_{target_col}"):
        form_cols = st.columns(3)
        values = {}
        for i, col in enumerate(features):
            with form_cols[i % 3]:
                key = f"input_{file_id}_{target_col}_{col}"
                if is_numeric_dtype(data[col]):
                    values[col] = numeric_input(nice(col), data[col], key)
                else:
                    options = sorted(data[col].dropna().astype(str).unique())
                    top = str(data[col].mode(dropna=True).iloc[0])
                    values[col] = st.selectbox(nice(col), options, index=options.index(top), key=key)
        submitted = st.form_submit_button("Predict", key="predict_submit", type="primary")

    if submitted:
        try:
            with st.spinner("Building the model and making the prediction..."):
                model = fit_final_model(data, target_col, tuple(features), model_name)
                row = pd.DataFrame([values], columns=list(features))
                value = float(model.predict(row)[0])
            st.session_state["prediction"] = {"sig": sig, "model": model_name, "value": value}
        except Exception as err:  # noqa: BLE001
            st.error(f"The prediction failed. Check your inputs and try again. Details: {err}")
            return

    result = st.session_state.get("prediction")
    if not result or result["sig"] != sig:
        return

    average = float(data[target_col].mean())
    spread = float(data[target_col].std())
    diff = result["value"] - average
    st.markdown("#### Result")
    r1, r2 = st.columns(2)
    r1.metric(f"Predicted {plain(target_label)} ({result['model']})", fmt(result["value"], unit), delta=f"{diff:+,.2f}{unit} vs average")
    r2.metric("Dataset average", fmt(average, unit))

    percent = f" ({diff / abs(average) * 100:+.0f}%)" if average else ""
    if abs(diff) <= 0.25 * spread:
        st.info(f"About average. The prediction is close to the dataset average{percent}.")
    elif diff > 0:
        st.success(f"Above average. The prediction is higher than the dataset average{percent}.")
    else:
        st.warning(f"Below average. The prediction is lower than the dataset average{percent}.")

    compare = pd.DataFrame({"what": ["Prediction", "Dataset average"], "value": [result["value"], average]})
    chart = (
        alt.Chart(compare)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=nice(target_col)),
            y=alt.Y("what:N", sort=["Prediction", "Dataset average"], title=None),
            color=alt.Color("what:N", scale=alt.Scale(domain=["Prediction", "Dataset average"], range=["#4C78A8", "#9AA0A6"]), legend=None),
            tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("value:Q", title="Value", format=",.2f")],
        )
        .properties(title="Prediction compared with the dataset average", height=140)
    )
    st.altair_chart(chart, width="stretch")


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Your data")
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="upload_csv",
        help="Export your campaign table from Excel or your analytics tool as CSV.",
    )

if uploaded is None:
    show_empty_state()
    st.stop()

file_bytes = uploaded.getvalue()
file_id = hashlib.sha256(file_bytes).hexdigest()[:10]

try:
    with st.spinner("Reading your file..."):
        raw = load_csv(file_bytes)
except Exception as err:  # noqa: BLE001
    st.error(
        "We could not read this file as a CSV. In Excel, choose Save As > CSV UTF-8, "
        f"then upload it again. Details: {err}"
    )
    st.stop()

missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing_columns:
    st.error(
        "These required columns are missing: " + ", ".join(missing_columns) + ". "
        "Check the spelling in your file's header row (names are case-sensitive), then upload again."
    )
    st.caption("Columns found in your file: " + ", ".join(map(str, raw.columns)))
    st.stop()

try:
    with st.spinner("Cleaning your data..."):
        df, report = clean_data(raw)
except Exception as err:  # noqa: BLE001
    st.error(f"Cleaning the data failed. Check that the file has the expected columns and values. Details: {err}")
    st.stop()

if len(df) < MIN_ROWS:
    st.error(
        f"Only {len(df)} rows are left after cleaning, and at least {MIN_ROWS} are needed. "
        "Upload a larger file or check why so many rows were removed."
    )
    st.stop()

with st.sidebar:
    st.header("What to predict")
    target_label = st.selectbox("Predict this", list(TARGETS), key="target_choice")
    target_cfg = TARGETS[target_label]
    candidates, excluded = feature_candidates(df, target_cfg["col"], report["id_cols"])
    features = st.multiselect(
        "Columns the model can use",
        options=candidates,
        default=candidates,
        format_func=nice,
        key=f"features_{file_id}_{target_cfg['col']}",
        help="Remove columns you would not know before a campaign runs, such as revenue when predicting conversion rate.",
    )
    st.caption(f"{report['kept']:,} clean rows loaded.")

signature = (file_id, target_cfg["col"], tuple(features))

st.title("Marketing performance explorer")
tab_overview, tab_kpis, tab_models, tab_predict = st.tabs(
    ["Overview", "KPIs & charts", "Model comparison", "Predict"]
)
with tab_overview:
    render_overview(raw, df, report)
with tab_kpis:
    render_kpis(df, report)
with tab_models:
    render_models(df, target_label, target_cfg, features, excluded, signature)
with tab_predict:
    render_predict(df, target_label, target_cfg, features, signature, file_id)
