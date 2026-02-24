# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Provisional Natality Data Dashboard")
st.subheader("Birth Analysis by State and Gender")


REQUIRED_LOGICAL_FIELDS = [
    "state_of_residence",
    "month",
    "month_code",
    "year_code",
    "sex_of_infant",
    "births",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def _token_match_score(target: str, candidate: str) -> int:
    target_tokens = [t for t in target.split("_") if t]
    cand = candidate
    score = 0
    for t in target_tokens:
        if t in cand:
            score += 1
    return score


def map_logical_fields(df_cols: list[str], logical_fields: list[str]) -> dict[str, str]:
    cols_set = set(df_cols)
    mapping: dict[str, str] = {}

    # 1) exact matches first
    for lf in logical_fields:
        if lf in cols_set:
            mapping[lf] = lf

    # 2) token-based best match for remaining fields
    for lf in logical_fields:
        if lf in mapping:
            continue
        best_col = None
        best_score = -1

        for c in df_cols:
            score = _token_match_score(lf, c)
            if score > best_score:
                best_score = score
                best_col = c

        # accept only if we matched all tokens OR a strong partial match
        # strong partial match = all but one token present and at least 2 tokens total
        tokens = [t for t in lf.split("_") if t]
        if best_col is not None:
            if best_score == len(tokens):
                mapping[lf] = best_col
            elif len(tokens) >= 2 and best_score >= len(tokens) - 1 and best_score >= 2:
                mapping[lf] = best_col

    return mapping


def add_all_option(values) -> list:
    uniq = pd.Series(values).dropna().unique().tolist()
    try:
        uniq_sorted = sorted(uniq)
    except Exception:
        uniq_sorted = uniq
    return ["All"] + uniq_sorted


def apply_multiselect_filter(df: pd.DataFrame, col: str, selected: list) -> pd.DataFrame:
    if not selected or "All" in selected:
        return df
    return df[df[col].isin(selected)]


# STEP 3 — Load Data
try:
    df_raw = pd.read_csv("Provisional_Natality_2025_CDC.csv")
except FileNotFoundError:
    st.error("Dataset file not found in repository.")
    st.stop()
except Exception as e:
    st.error("Failed to load dataset.")
    st.write(str(e))
    st.stop()

df = normalize_columns(df_raw)

# Validate / map required logical fields
mapped = map_logical_fields(df.columns.tolist(), REQUIRED_LOGICAL_FIELDS)
missing = [lf for lf in REQUIRED_LOGICAL_FIELDS if lf not in mapped]

if missing:
    st.error(
        "Missing required logical fields after dynamic matching: "
        + ", ".join(missing)
    )
    st.write(df.columns)
    st.stop()

# Rename to logical fields for consistent downstream use
rename_map = {mapped[lf]: lf for lf in REQUIRED_LOGICAL_FIELDS}
df_work = df.rename(columns=rename_map).copy()

# Convert births to numeric and drop nulls
df_work["births"] = pd.to_numeric(df_work["births"], errors="coerce")
df_work = df_work.dropna(subset=["births"])

# STEP 4 — Sidebar Filters (multiselect only, with "All")
months_options = add_all_option(df_work["month"])
sex_options = add_all_option(df_work["sex_of_infant"])
state_options = add_all_option(df_work["state_of_residence"])

selected_months = st.sidebar.multiselect(
    "Month", options=months_options, default=["All"]
)
selected_sexes = st.sidebar.multiselect(
    "Gender", options=sex_options, default=["All"]
)
selected_states = st.sidebar.multiselect(
    "State of Residence", options=state_options, default=["All"]
)

# STEP 5 — Filtering Logic
df_filtered = df_work.copy()
df_filtered = apply_multiselect_filter(df_filtered, "month", selected_months)
df_filtered = apply_multiselect_filter(df_filtered, "sex_of_infant", selected_sexes)
df_filtered = apply_multiselect_filter(df_filtered, "state_of_residence", selected_states)

# STEP 9 — Edge Case Handling: empty results
if df_filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# STEP 6 — Aggregation
df_agg = (
    df_filtered.groupby(["state_of_residence", "sex_of_infant"], as_index=False)["births"]
    .sum()
)
df_agg = df_agg.sort_values(by=["state_of_residence", "sex_of_infant"], kind="stable")

# STEP 7 — Plot
fig = px.bar(
    df_agg,
    x="state_of_residence",
    y="births",
    color="sex_of_infant",
    title="Total Births by State and Gender",
    labels={
        "state_of_residence": "State of Residence",
        "births": "Births",
        "sex_of_infant": "Gender",
    },
)
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="Gender",
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# STEP 8 — Show Filtered Table (clean display, no index)
display_cols = ["state_of_residence", "month", "month_code", "year_code", "sex_of_infant", "births"]
df_table = df_filtered.loc[:, [c for c in display_cols if c in df_filtered.columns]].copy()

# Prefer modern Streamlit hide_index, fall back safely
try:
    st.dataframe(df_table, use_container_width=True, hide_index=True)
except TypeError:
    st.dataframe(df_table.reset_index(drop=True), use_container_width=True)
