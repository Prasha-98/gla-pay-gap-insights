import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ───────────────────────────────────────────
# 1.  Helpers to locate and parse each block
# ───────────────────────────────────────────
def _year_row(df: pd.DataFrame):
    """
    Return (row_idx, [years]) for the first row that contains ≥2
    four-digit year values.  Ignores anything that can’t be converted
    cleanly to an int.
    """
    for i, row in df.iterrows():
        years = []
        for val in row:
            if pd.isna(val):
                continue
            try:
                num = int(val)
            except (ValueError, TypeError):
                continue            # skip anything non-numeric
            if 2000 <= num <= 2100:
                years.append(num)
        if len(years) >= 2:
            return i, years
    return None, []

def _block(df: pd.DataFrame, start: int, yrs: list[int]):
    """Grab consecutive lines until blank / 'change' / next header."""
    out, r, n = {}, start, len(yrs)
    while r < len(df):
        first, second = df.iat[r, 0], df.iat[r, 1] if df.shape[1] > 1 else np.nan
        # stop if we hit a blank line or a header for the next section
        if pd.isna(first) and pd.isna(second):
            break
        if isinstance(first, str) and "change" in first.lower():
            break
        if isinstance(first, str) and "pay gap" in first.lower():
            break  # this ends the Hourly-Pay block

        label = second if pd.notna(second) else first
        if label is None or pd.isna(label):
            break

        values = [
            pd.to_numeric(df.iat[r, 2 + j], errors="coerce") if 2 + j < df.shape[1] else np.nan
            for j in range(n)
        ]
        out[str(label).strip()] = values
        r += 1
    return out, r  # r points at the first non-processed row

def _gap_block(df: pd.DataFrame, start: int, yrs: list[int]):
    """Read the Pay-Gap lines (start row contains the text 'Pay Gap')."""
    out, r, n = {}, start, len(yrs)
    while r < len(df):
        first, second = df.iat[r, 0], df.iat[r, 1] if df.shape[1] > 1 else np.nan
        if pd.isna(first) and pd.isna(second):
            break
        if isinstance(first, str) and "change" in first.lower():
            break

        label = second if pd.notna(second) else first
        if label is None or pd.isna(label):
            r += 1
            continue

        values = [
            pd.to_numeric(df.iat[r, 2 + j], errors="coerce") if 2 + j < df.shape[1] else np.nan
            for j in range(n)
        ]
        out[str(label).strip()] = values
        r += 1
    return out

def parse_sheet(path: str, sheet_name: str = "1"):
    """Return `(pay_df, gap_df)` extracted from the given sheet."""
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    row_y, years = _year_row(df)
    if row_y is None:
        return None, None

    # locate “Hourly pay”
    row_hp = next(
        (i for i in range(row_y + 1, len(df)) if isinstance(df.iat[i, 0], str) and "hourly" in df.iat[i, 0].lower()),
        None,
    )
    if row_hp is None:
        return None, None

    pay_dict, row_after_pay = _block(df, row_hp, years)

    # locate “Pay gap”
    row_gap = next(
        (i for i in range(row_after_pay, len(df)) if isinstance(df.iat[i, 0], str) and "pay gap" in df.iat[i, 0].lower()),
        None,
    )
    gap_dict = _gap_block(df, row_gap, years) if row_gap is not None else {}

    pay_df = pd.DataFrame(pay_dict, index=years)
    gap_df = pd.DataFrame(gap_dict, index=years)
    pay_df.index.name = gap_df.index.name = "Year"
    return pay_df, gap_df

# ───────────────────────────────────────────
# 2.  Load all three workbooks (cached)
# ───────────────────────────────────────────
@st.cache_data
def load_all():
    return {
        "Disability": parse_sheet("Disability pay gap data tables 2021-2024.xlsx"),
        "Gender":     parse_sheet("Gender pay gap data tables 2017-2024.xlsx"),
        "Ethnicity":  parse_sheet("Ethnicity pay gap data tables 2017-2024.xlsx"),
    }

data_map = load_all()

# ───────────────────────────────────────────
# 3.  Insight + forecast utilities
# ───────────────────────────────────────────
def insight_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    start, end = df.index.min(), df.index.max()
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        a, b = s.get(start, np.nan), s.get(end, np.nan)
        delta = b - a if pd.notna(a) and pd.notna(b) else np.nan
        pct = delta / a if pd.notna(a) and a != 0 else np.nan
        rows.append(
            dict(
                Category=col,
                Start=a,
                End=b,
                AbsChange=delta,
                PctChange=pct,
                Avg=s.mean(),
                Std=s.std(),
                Trend="↑" if delta > 0 else "↓" if delta < 0 else "→",
            )
        )
    return pd.DataFrame(rows).set_index("Category")

def forecast(df: pd.DataFrame) -> pd.Series:
    """Linear regression extrapolation (one step)."""
    if df.empty:
        return pd.Series(dtype=float)
    next_year = df.index.max() + 1
    X = df.index.values.reshape(-1, 1)
    preds = {}
    for col in df.columns:
        y = df[col].astype(float).values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue
        reg = LinearRegression().fit(X[mask], y[mask])
        preds[col] = reg.predict([[next_year]])[0]
    return pd.Series(preds, name=next_year)

# ───────────────────────────────────────────
# 4.  Streamlit UI
# ───────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("GLA Pay Gap — Deep-Dive Dashboard")

dataset = st.sidebar.selectbox("Dataset", ["Disability", "Gender", "Ethnicity"])
metric  = st.sidebar.radio("Metric", ["Hourly Pay", "Pay Gap"])

pay_df, gap_df = data_map[dataset]
df = pay_df if metric == "Hourly Pay" else gap_df

# handle missing case robustly
if df is None or df.empty:
    st.warning("No Pay-Gap data found for this selection ❗️")
    st.stop()

fmt = "{:.2f}" if metric == "Hourly Pay" else "{:.2%}"

st.subheader(f"{dataset} — {metric}")
st.dataframe(df.style.format(fmt))

# line plot
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(marker="o", ax=ax)
ax.set_ylabel("£" if metric == "Hourly Pay" else "Gap (fraction)")
ax.set_title(f"{metric} trend ({dataset})")
ax.grid(True)
st.pyplot(fig, use_container_width=True)

# insight table
ins = insight_table(df)
st.markdown("### 🔍 Key Insights")
st.dataframe(
    ins.style.format(
        {
            "Start": fmt,
            "End": fmt,
            "AbsChange": fmt,
            "PctChange": "{:.2%}",
            "Avg": fmt,
            "Std": fmt,
        }
    )
)

if not ins.empty:
    pos = ins["AbsChange"].idxmax()
    neg = ins["AbsChange"].idxmin()
    st.markdown(
        f"""
**Highlights**

• Biggest ↑ change: **{pos}** ({ins.loc[pos,'AbsChange']:+.2f}, {ins.loc[pos,'PctChange']:+.2%})  
• Biggest ↓ change: **{neg}** ({ins.loc[neg,'AbsChange']:+.2f}, {ins.loc[neg,'PctChange']:+.2%})  
• Average volatility (σ): {ins["Std"].mean():.2f}
"""
    )

# forecast
st.markdown("### 🔮 Next-Year Forecast")
pred = forecast(df)
if pred.empty:
    st.info("Insufficient data for forecasting.")
else:
    st.write(pred.to_frame("Forecast").style.format(fmt))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df_plus = df.copy()
    df_plus.loc[pred.name] = pred
    df_plus.sort_index().plot(marker="o", ax=ax2, linestyle="--")
    ax2.set_title(f"Projection for {pred.name}")
    ax2.grid(True)
    st.pyplot(fig2, use_container_width=True)

