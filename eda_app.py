"""
================================================================================
 INTERACTIVE EDA STUDIO
 A Streamlit app that performs Exploratory Data Analysis using a structured
 15-STEP APPROACH organized across the 4 CORE DIMENSIONS of data analysis:

     1) COMPOSITION   - what are the parts of the whole?
     2) DISTRIBUTION   - how are values spread out?
     3) COMPARISON     - how do groups/categories differ?
     4) RELATIONSHIP   - how do variables relate to one another?

 Works with any uploaded tabular file (CSV / Excel / TSV) or a set of built-in
 sample datasets loaded from the seaborn library (titanic, tips, iris, etc.)
================================================================================
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Optional: statsmodels enables OLS trendlines on scatter plots.
try:
    import statsmodels.api as sm  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ==============================================================================
# PAGE CONFIG & STYLE
# ==============================================================================
st.set_page_config(
    page_title="Interactive EDA Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1.2rem;}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif;}
    .step-badge {
        display: inline-block; background: #6366F1; color: white;
        border-radius: 999px; padding: 2px 12px; font-size: 0.75rem;
        font-weight: 600; margin-right: 8px;
    }
    .dim-header {
        padding: 10px 16px; border-radius: 10px; margin-bottom: 10px;
        color: white; font-weight: 700; font-size: 1.05rem;
    }
    .comp-header {background: linear-gradient(90deg,#F59E0B,#F97316);}
    .dist-header {background: linear-gradient(90deg,#3B82F6,#06B6D4);}
    .cmp-header  {background: linear-gradient(90deg,#10B981,#22C55E);}
    .rel-header  {background: linear-gradient(90deg,#8B5CF6,#EC4899);}
    .ovw-header  {background: linear-gradient(90deg,#64748B,#334155);}
    .ins-header  {background: linear-gradient(90deg,#EF4444,#DC2626);}
    .insight-box {
        background: #F8FAFC; border-left: 4px solid #6366F1;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
        font-size: 0.95rem;
    }
    .metric-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 14px; text-align:center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================
SAMPLE_DATASETS = [
    "titanic", "tips", "iris", "penguins", "diamonds",
    "mpg", "flights", "car_crashes", "planets", "healthexp",
]


@st.cache_data(show_spinner=False)
def load_seaborn_dataset(name: str) -> pd.DataFrame:
    return sns.load_dataset(name)


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if filename.lower().endswith(".tsv"):
        return pd.read_csv(buf, sep="\t")
    return pd.read_csv(buf)


def get_column_types(df: pd.DataFrame):
    """Split columns into numeric / categorical / datetime, attempting date parsing."""
    df = df.copy()
    numeric_cols, categorical_cols, datetime_cols = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Try to detect a date-like column (only if it actually parses well)
            if df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        datetime_cols.append(col)
                        continue
                except Exception:
                    pass
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, datetime_cols


# ==============================================================================
# INTERPRETATION HELPERS (auto-generated, plain-English insights)
# ==============================================================================
def interpret_missing(df: pd.DataFrame) -> str:
    miss = df.isna().sum()
    total_missing = miss.sum()
    if total_missing == 0:
        return "No missing values detected — the dataset is complete."
    worst = miss.sort_values(ascending=False)
    worst = worst[worst > 0]
    top = worst.index[0]
    pct = worst.iloc[0] / len(df) * 100
    return (f"{len(worst)} column(s) contain missing values, totaling {total_missing} cells. "
            f"'{top}' is the most affected at {pct:.1f}% missing. "
            f"Columns above ~30% missing are candidates for dropping; lower rates are usually "
            f"safe to impute (mean/median for numeric, mode for categorical).")


def interpret_duplicates(df: pd.DataFrame) -> str:
    dup = df.duplicated().sum()
    if dup == 0:
        return "No duplicate rows found."
    return (f"{dup} duplicate row(s) detected ({dup/len(df)*100:.1f}% of the data). "
            f"Consider removing them unless repeated records are legitimate (e.g. repeated events).")


def interpret_distribution(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) < 3:
        return "Not enough data points to assess distribution shape."
    skewness = stats.skew(s)
    kurt = stats.kurtosis(s)
    shape = "roughly symmetric"
    if skewness > 0.5:
        shape = "right-skewed (long tail toward higher values)"
    elif skewness < -0.5:
        shape = "left-skewed (long tail toward lower values)"
    tail = "heavier-tailed than normal (more outliers likely)" if kurt > 1 else (
        "lighter-tailed than normal" if kurt < -1 else "close to normal tail-weight")
    return (f"Mean = {s.mean():.2f}, Median = {s.median():.2f}, Std Dev = {s.std():.2f}. "
            f"The distribution is **{shape}** (skew={skewness:.2f}) and is **{tail}** (kurtosis={kurt:.2f}). "
            f"{'Mean and median are close, suggesting few extreme values.' if abs(s.mean()-s.median()) < 0.1*s.std() else 'Mean and median diverge — extreme values are pulling the mean.'}")


def interpret_categorical(series: pd.Series) -> str:
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return "No data available."
    top_cat, top_count = vc.index[0], vc.iloc[0]
    top_pct = top_count / vc.sum() * 100
    n_unique = series.nunique()
    balance = "highly imbalanced" if top_pct > 60 else ("moderately balanced" if top_pct > 35 else "fairly balanced")
    return (f"'{top_cat}' is the most frequent category at {top_pct:.1f}% of records "
            f"({n_unique} unique categories total). Distribution across categories is **{balance}**.")


def interpret_outliers(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) < 4:
        return "Not enough data to detect outliers."
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    pct = len(outliers) / len(s) * 100
    if len(outliers) == 0:
        return f"No outliers detected using the IQR method (bounds: {lower:.2f} to {upper:.2f})."
    return (f"{len(outliers)} outlier(s) detected ({pct:.1f}% of values), lying outside the "
            f"IQR bounds [{lower:.2f}, {upper:.2f}]. "
            f"{'This is a notable share — consider investigating or applying a transform (log/winsorize).' if pct > 5 else 'This is a small share and likely safe to leave as-is or review case-by-case.'}")


def interpret_correlation(corr: pd.DataFrame) -> str:
    corr_unstacked = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
    if corr_unstacked.empty:
        return "Not enough numeric columns to compute correlations."
    corr_unstacked = corr_unstacked.sort_values(key=lambda x: x.abs(), ascending=False)
    pairs_seen = set()
    lines = []
    for (a, b), val in corr_unstacked.items():
        key = frozenset([a, b])
        if key in pairs_seen:
            continue
        pairs_seen.add(key)
        strength = "very strong" if abs(val) > 0.8 else ("strong" if abs(val) > 0.6 else ("moderate" if abs(val) > 0.4 else "weak"))
        direction = "positive" if val > 0 else "negative"
        lines.append(f"**{a} ↔ {b}**: r = {val:.2f} ({strength} {direction})")
        if len(lines) == 3:
            break
    return "Top correlated pairs — " + "; ".join(lines) + "."


def interpret_scatter(x: pd.Series, y: pd.Series) -> str:
    df_ = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_) < 3:
        return "Not enough overlapping data points to assess the relationship."
    r, p = stats.pearsonr(df_["x"], df_["y"])
    strength = "very strong" if abs(r) > 0.8 else ("strong" if abs(r) > 0.6 else ("moderate" if abs(r) > 0.4 else ("weak" if abs(r) > 0.2 else "negligible")))
    direction = "positive" if r > 0 else "negative"
    sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p ≥ 0.05)"
    return (f"Pearson correlation r = {r:.3f} — a **{strength} {direction}** relationship, "
            f"which is {sig}. {'As one variable increases, the other tends to increase as well.' if r>0.2 else ('As one variable increases, the other tends to decrease.' if r < -0.2 else 'The two variables show little linear relationship — a non-linear pattern may still exist.')}")


def interpret_comparison(df: pd.DataFrame, cat_col: str, num_col: str, agg: str) -> str:
    grouped = df.groupby(cat_col)[num_col].agg(agg).sort_values(ascending=False)
    if grouped.empty:
        return "No data available for comparison."
    top_group, top_val = grouped.index[0], grouped.iloc[0]
    bottom_group, bottom_val = grouped.index[-1], grouped.iloc[-1]
    spread = ((top_val - bottom_val) / bottom_val * 100) if bottom_val != 0 else float("nan")
    return (f"'{top_group}' has the highest {agg} {num_col} ({top_val:.2f}), while '{bottom_group}' has the "
            f"lowest ({bottom_val:.2f}). That's a "
            f"{'gap of {:.1f}% relative to the lowest group.'.format(spread) if not np.isnan(spread) else 'notable gap between groups.'}")


# ==============================================================================
# STEP HEADER WIDGET
# ==============================================================================
def step_header(step_no, title, dimension_class, dim_label):
    st.markdown(
        f"""<div class="dim-header {dimension_class}">
        <span class="step-badge">STEP {step_no}</span>{title}
        <span style="float:right; opacity:0.85; font-weight:400;">{dim_label}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f'<div class="insight-box">🔎 <b>Interpretation:</b> {text}</div>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR — DATA SOURCE SELECTION
# ==============================================================================
st.sidebar.title("📊 EDA Studio")
st.sidebar.caption("15-Step EDA across 4 dimensions of analysis")

data_source = st.sidebar.radio("Data source", ["Sample dataset (seaborn)", "Upload your own file"])

df = None
dataset_label = ""

if data_source == "Sample dataset (seaborn)":
    choice = st.sidebar.selectbox("Choose a sample dataset", SAMPLE_DATASETS, index=0)
    try:
        with st.spinner(f"Loading '{choice}' dataset..."):
            df = load_seaborn_dataset(choice)
        dataset_label = choice
    except Exception as e:
        st.sidebar.error(f"Could not load '{choice}': {e}\n\nThis requires internet access (seaborn fetches "
                          f"sample data from GitHub the first time). Try uploading your own file instead.")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV / TSV / Excel", type=["csv", "tsv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = load_uploaded_file(uploaded.getvalue(), uploaded.name)
            dataset_label = uploaded.name
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About the 15-step approach"):
    st.markdown("""
**Overview (Steps 1–4)**
1. Data snapshot
2. Missing values
3. Duplicates
4. Data types & summary stats

**Distribution (Steps 5–7)**
5. Numeric distribution
6. Categorical distribution
7. Outlier detection

**Composition (Steps 8–9)**
8. Part-to-whole (pie/donut)
9. Stacked composition

**Comparison (Steps 10–11)**
10. Group comparison (bar)
11. Group comparison (box/violin)

**Relationship (Steps 12–14)**
12. Correlation heatmap
13. Scatter relationship
14. Multivariate scatter matrix

**Insights (Step 15)**
15. Auto-generated summary
""")

if df is None:
    st.title("📊 Interactive EDA Studio")
    st.info("👈 Pick a sample dataset or upload your own file from the sidebar to get started.")
    st.markdown("""
    This app walks through a **15-step EDA approach** spanning the **4 core dimensions**
    of data analysis: **Composition, Distribution, Comparison,** and **Relationship**.
    Every chart is interactive (hover, zoom, pan) and comes with an automatically
    generated plain-English interpretation.
    """)
    st.stop()

numeric_cols, categorical_cols, datetime_cols = get_column_types(df)

st.title("📊 Interactive EDA Studio")
st.caption(f"Currently analyzing: **{dataset_label}** &nbsp;|&nbsp; {df.shape[0]} rows × {df.shape[1]} columns")

tabs = st.tabs(["🧾 Overview", "📈 Distribution", "🥧 Composition", "📊 Comparison", "🔗 Relationship", "💡 Insights"])

# ==============================================================================
# TAB 0 — OVERVIEW (Steps 1-4)
# ==============================================================================
with tabs[0]:
    step_header(1, "Data Snapshot", "ovw-header", "OVERVIEW")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3>Rows</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3>Columns</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>{len(numeric_cols)}</h3>Numeric cols</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h3>{len(categorical_cols)}</h3>Categorical cols</div>', unsafe_allow_html=True)
    st.write("")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("")
    step_header(2, "Missing Values", "ovw-header", "OVERVIEW")
    miss_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().sum().values / len(df) * 100).round(2),
    }).sort_values("missing_pct", ascending=False)
    fig_miss = px.bar(miss_df, x="column", y="missing_pct", color="missing_pct",
                       color_continuous_scale="Reds", labels={"missing_pct": "% missing"},
                       title="Missing values by column")
    st.plotly_chart(fig_miss, use_container_width=True)
    insight(interpret_missing(df))

    st.write("")
    step_header(3, "Duplicate Rows", "ovw-header", "OVERVIEW")
    n_dup = df.duplicated().sum()
    st.metric("Duplicate rows", n_dup, f"{n_dup/len(df)*100:.2f}% of data")
    insight(interpret_duplicates(df))

    st.write("")
    step_header(4, "Data Types & Summary Statistics", "ovw-header", "OVERVIEW")
    dtype_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True, height=150)
    st.markdown("**Descriptive statistics (numeric columns)**")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found.")

# ==============================================================================
# TAB 1 — DISTRIBUTION (Steps 5-7)
# ==============================================================================
with tabs[1]:
    step_header(5, "Numeric Distribution", "dist-header", "DISTRIBUTION")
    if numeric_cols:
        col_a, col_b = st.columns([3, 1])
        num_choice = col_a.selectbox("Numeric column", numeric_cols, key="dist_num")
        n_bins = col_b.slider("Bins", 5, 100, 30, key="dist_bins")
        fig_hist = px.histogram(df, x=num_choice, nbins=n_bins, marginal="box",
                                 color_discrete_sequence=["#3B82F6"],
                                 title=f"Distribution of {num_choice}")
        st.plotly_chart(fig_hist, use_container_width=True)
        insight(interpret_distribution(df[num_choice]))
    else:
        st.info("No numeric columns available.")

    st.write("")
    step_header(6, "Categorical Distribution", "dist-header", "DISTRIBUTION")
    if categorical_cols:
        cat_choice = st.selectbox("Categorical column", categorical_cols, key="dist_cat")
        vc = df[cat_choice].value_counts(dropna=True).reset_index()
        vc.columns = [cat_choice, "count"]
        fig_bar_cat = px.bar(vc, x=cat_choice, y="count", color="count",
                              color_continuous_scale="Blues", title=f"Frequency of {cat_choice}")
        st.plotly_chart(fig_bar_cat, use_container_width=True)
        insight(interpret_categorical(df[cat_choice]))
    else:
        st.info("No categorical columns available.")

    st.write("")
    step_header(7, "Outlier Detection", "dist-header", "DISTRIBUTION")
    if numeric_cols:
        out_choice = st.selectbox("Numeric column", numeric_cols, key="dist_outlier")
        fig_box = px.box(df, y=out_choice, points="outliers", color_discrete_sequence=["#06B6D4"],
                          title=f"Box plot (IQR outlier view) — {out_choice}")
        st.plotly_chart(fig_box, use_container_width=True)
        insight(interpret_outliers(df[out_choice]))
    else:
        st.info("No numeric columns available.")

# ==============================================================================
# TAB 2 — COMPOSITION (Steps 8-9)
# ==============================================================================
with tabs[2]:
    step_header(8, "Part-to-Whole Composition", "comp-header", "COMPOSITION")
    if categorical_cols:
        comp_col = st.selectbox("Category column", categorical_cols, key="comp_pie")
        donut = st.checkbox("Donut style", value=True, key="comp_donut")
        vc = df[comp_col].value_counts(dropna=True).reset_index()
        vc.columns = [comp_col, "count"]
        fig_pie = px.pie(vc, names=comp_col, values="count", hole=0.45 if donut else 0,
                          title=f"Composition of {comp_col}")
        st.plotly_chart(fig_pie, use_container_width=True)
        insight(interpret_categorical(df[comp_col]))
    else:
        st.info("No categorical columns available.")

    st.write("")
    step_header(9, "Stacked Composition (two categories)", "comp-header", "COMPOSITION")
    if len(categorical_cols) >= 2:
        c1, c2 = st.columns(2)
        stack_x = c1.selectbox("X-axis category", categorical_cols, key="stack_x")
        remaining = [c for c in categorical_cols if c != stack_x]
        stack_color = c2.selectbox("Stacked-by category", remaining, key="stack_color")
        stacked_data = df.groupby([stack_x, stack_color]).size().reset_index(name="count")
        fig_stack = px.bar(stacked_data, x=stack_x, y="count", color=stack_color, barmode="stack",
                            title=f"{stack_x} composition broken down by {stack_color}")
        st.plotly_chart(fig_stack, use_container_width=True)
        top_combo = stacked_data.sort_values("count", ascending=False).iloc[0]
        insight(f"The largest combination is **{stack_x} = {top_combo[stack_x]}** with "
                f"**{stack_color} = {top_combo[stack_color]}** ({int(top_combo['count'])} records). "
                f"Use this to spot which sub-groups dominate each category.")
    else:
        st.info("Need at least 2 categorical columns for a stacked composition chart.")

# ==============================================================================
# TAB 3 — COMPARISON (Steps 10-11)
# ==============================================================================
with tabs[3]:
    step_header(10, "Group Comparison (Aggregated Bar)", "cmp-header", "COMPARISON")
    if categorical_cols and numeric_cols:
        c1, c2, c3 = st.columns(3)
        cmp_cat = c1.selectbox("Group by (category)", categorical_cols, key="cmp_cat")
        cmp_num = c2.selectbox("Measure (numeric)", numeric_cols, key="cmp_num")
        agg_func = c3.selectbox("Aggregation", ["mean", "median", "sum", "max", "min"], key="cmp_agg")
        grouped = df.groupby(cmp_cat)[cmp_num].agg(agg_func).reset_index().sort_values(cmp_num, ascending=False)
        fig_cmp = px.bar(grouped, x=cmp_cat, y=cmp_num, color=cmp_num, color_continuous_scale="Greens",
                          title=f"{agg_func.title()} of {cmp_num} by {cmp_cat}")
        st.plotly_chart(fig_cmp, use_container_width=True)
        insight(interpret_comparison(df, cmp_cat, cmp_num, agg_func))
    else:
        st.info("Need at least one categorical and one numeric column.")

    st.write("")
    step_header(11, "Group Comparison (Distribution Shape)", "cmp-header", "COMPARISON")
    if categorical_cols and numeric_cols:
        c1, c2, c3 = st.columns(3)
        cmp2_cat = c1.selectbox("Group by (category)", categorical_cols, key="cmp2_cat")
        cmp2_num = c2.selectbox("Measure (numeric)", numeric_cols, key="cmp2_num")
        plot_kind = c3.radio("Chart type", ["Box", "Violin"], key="cmp2_kind", horizontal=True)
        if plot_kind == "Box":
            fig_cmp2 = px.box(df, x=cmp2_cat, y=cmp2_num, color=cmp2_cat,
                               title=f"{cmp2_num} distribution across {cmp2_cat}")
        else:
            fig_cmp2 = px.violin(df, x=cmp2_cat, y=cmp2_num, color=cmp2_cat, box=True,
                                  title=f"{cmp2_num} distribution across {cmp2_cat}")
        st.plotly_chart(fig_cmp2, use_container_width=True)
        insight(interpret_comparison(df, cmp2_cat, cmp2_num, "median") +
                " Wider boxes/violins indicate more internal variability within that group.")
    else:
        st.info("Need at least one categorical and one numeric column.")

# ==============================================================================
# TAB 4 — RELATIONSHIP (Steps 12-14)
# ==============================================================================
with tabs[4]:
    step_header(12, "Correlation Heatmap", "rel-header", "RELATIONSHIP")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation matrix (numeric columns)")
        st.plotly_chart(fig_heat, use_container_width=True)
        insight(interpret_correlation(corr))
    else:
        st.info("Need at least 2 numeric columns to compute correlation.")

    st.write("")
    step_header(13, "Scatter Relationship", "rel-header", "RELATIONSHIP")
    if len(numeric_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, index=0, key="rel_x")
        y_options = [c for c in numeric_cols if c != x_col] or numeric_cols
        y_col = c2.selectbox("Y axis", y_options, index=0, key="rel_y")
        color_options = ["(none)"] + categorical_cols
        color_col = c3.selectbox("Color by", color_options, key="rel_color")
        trend = "ols" if HAS_STATSMODELS else None
        fig_scatter = px.scatter(df, x=x_col, y=y_col,
                                  color=None if color_col == "(none)" else color_col,
                                  trendline=trend, opacity=0.7,
                                  title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig_scatter, use_container_width=True)
        insight(interpret_scatter(df[x_col], df[y_col]))
    else:
        st.info("Need at least 2 numeric columns for a scatter plot.")

    st.write("")
    step_header(14, "Multivariate Scatter Matrix", "rel-header", "RELATIONSHIP")
    if len(numeric_cols) >= 3:
        chosen = st.multiselect("Numeric columns to include (3–5 recommended)",
                                 numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))],
                                 key="rel_matrix_cols")
        color_options2 = ["(none)"] + categorical_cols
        color_col2 = st.selectbox("Color by", color_options2, key="rel_matrix_color")
        if len(chosen) >= 2:
            fig_matrix = px.scatter_matrix(df, dimensions=chosen,
                                            color=None if color_col2 == "(none)" else color_col2,
                                            title="Pairwise relationships")
            fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
            st.plotly_chart(fig_matrix, use_container_width=True)
            insight("Look for diagonal-ish clouds (strong relationship), circular clouds (no relationship), "
                    "or separated color clusters (the grouping variable explains structure in the data).")
        else:
            st.info("Select at least 2 columns.")
    else:
        st.info("Need at least 3 numeric columns for a scatter matrix.")

# ==============================================================================
# TAB 5 — INSIGHTS (Step 15)
# ==============================================================================
with tabs[5]:
    step_header(15, "Auto-Generated Summary of Findings", "ins-header", "SYNTHESIS")

    st.markdown("#### Dataset Health")
    st.markdown(f"- {interpret_missing(df)}")
    st.markdown(f"- {interpret_duplicates(df)}")

    if numeric_cols:
        st.markdown("#### Distribution Highlights")
        for col in numeric_cols[:5]:
            st.markdown(f"- **{col}**: {interpret_distribution(df[col])}")

    if categorical_cols:
        st.markdown("#### Composition Highlights")
        for col in categorical_cols[:5]:
            st.markdown(f"- **{col}**: {interpret_categorical(df[col])}")

    if len(numeric_cols) >= 2:
        st.markdown("#### Relationship Highlights")
        corr_full = df[numeric_cols].corr(numeric_only=True)
        st.markdown(f"- {interpret_correlation(corr_full)}")

    if categorical_cols and numeric_cols:
        st.markdown("#### Comparison Highlights")
        st.markdown(f"- {interpret_comparison(df, categorical_cols[0], numeric_cols[0], 'mean')}")

    st.write("")
    st.success("This summary is generated automatically from statistical properties of the current dataset. "
               "Use the tabs above to dig deeper into any specific column or relationship.")

    csv_bytes = df.describe(include="all").to_csv().encode("utf-8")
    st.download_button("⬇️ Download summary statistics (CSV)", csv_bytes, "summary_statistics.csv", "text/csv")
