"""mcdaio Streamlit GUI.

Run with::

    streamlit run mcdaio/app.py

Features
--------
- Methods: MABAC, TOPSIS, VIKOR, ARAS (imported from the ``mcdaio`` package)
- Dark / light mode toggle
- CSV upload or manual matrix input
- Criteria weights and types (cost / benefit)
- Sensitivity analysis (leave-one-out criterion removal)
- Bar chart and stability line chart
- CSV export of the results
"""

from __future__ import annotations
from io import StringIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mcdaio import METHOD_REGISTRY, sensitivity_analysis, spearman_corr

st.set_page_config(page_title="mcdaio", layout="wide")

DARK_CSS = """
:root {
    --bg: #0e1117;
    --card: #0f1720;
    --text: #e6eef6;
    --muted: #9aa8b2;
    --accent: #00c26a;
}
body { background: var(--bg) !important; color: var(--text) !important; }
.stApp { background: linear-gradient(90deg, rgba(14,17,23,1) 0%, rgba(19,23,29,1) 100%) !important; }
.stButton > button { background-color: var(--accent); color: #001018; border: 0; }
.css-1d391kg { background-color: transparent; }
"""

LIGHT_CSS = """
:root {
    --bg: #f7fafc;
    --card: #ffffff;
    --text: #0b1726;
    --muted: #556b7a;
    --accent: #0b84ff;
}
body { background: var(--bg) !important; color: var(--text) !important; }
.stApp { background: linear-gradient(90deg, rgba(247,250,252,1) 0%, rgba(255,255,255,1) 100%) !important; }
.stButton > button { background-color: var(--accent); color: #ffffff; border: 0; }
.stMarkdown, .css-1d391kg, label,
.stTextInput label, .stTextArea label, .stFileUploader label,
.stSelectbox label, .stCheckbox label { color: var(--text) !important; }
.stSidebar .stSelectbox label:first-of-type,
.stSidebar .stMarkdown b { color: #ffffff !important; }
"""

def parse_matrix(uploaded_file, manual_text: str):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            return df.values.astype(float)
        except Exception:
            try:
                text = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(text, header=None)
                return df.values.astype(float)
            except Exception as exc:
                raise ValueError(f"File loading error: {exc}") from exc

    if manual_text and manual_text.strip():
        rows = [r.strip() for r in manual_text.strip().splitlines() if r.strip()]
        data = []
        for row in rows:
            parts = [p.strip() for p in row.split(",") if p.strip()]
            if not parts:
                continue
            try:
                data.append([float(x) for x in parts])
            except ValueError as exc:
                raise ValueError(
                    "Matrix parsing error: ensure numeric values separated by commas."
                ) from exc
        return np.array(data, dtype=float)

    return None


def parse_vector(text: str, expected_len: int | None = None, dtype=float, default=None):
    if text is None or text.strip() == "":
        return default
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vec = np.array([dtype(p) for p in parts], dtype=dtype)
    if expected_len is not None and len(vec) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(vec)}.")
    return vec

def plot_results_bar(res_df: pd.DataFrame, method_name: str, dark_mode_flag: bool):
    fig = px.bar(
        res_df,
        x="Alternative",
        y="Score",
        color="Rank",
        text="Rank",
        title=f"Results: {method_name}",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_dark" if dark_mode_flag else "plotly_white",
        yaxis_title="Score",
    )
    return fig

def plot_stability_line(methods, ranks_matrix, dark_mode_flag: bool):
    if not ranks_matrix:
        return go.Figure()
    n_alts = len(ranks_matrix[0])
    fig = go.Figure()
    for i in range(n_alts):
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[r[i] for r in ranks_matrix],
                mode="lines+markers",
                name=f"Alt {i + 1}",
                marker=dict(symbol="star", size=12, color="green"),
                line=dict(dash="dash", color="green"),
            )
        )
    fig.update_layout(
        title="Stability of the alternatives' positions",
        xaxis_title="Variant / method",
        yaxis_title="Ranking position (1 = best)",
        template="plotly_dark" if dark_mode_flag else "plotly_white",
    )
    fig.update_yaxes(autorange="reversed")
    return fig

with st.sidebar:
    st.title("mcdaio")
    dark_mode = st.checkbox("Dark mode", value=False)
    st.markdown("---")
    method = st.selectbox("Select MCDA method:", list(METHOD_REGISTRY.keys()))
    do_sensitivity = st.checkbox("Sensitivity analysis (remove criteria)", value=True)
    show_corr = st.checkbox("Method comparison (Spearman)", value=True)
    st.markdown("---")
    st.info(
        "Upload a decision matrix (CSV/TXT) or paste it manually.\n"
        "Provide weights and criterion types as comma-separated values."
    )

st.markdown(f"<style>{DARK_CSS if dark_mode else LIGHT_CSS}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])

with col1:
    uploaded = st.file_uploader("Select matrix file (CSV/TXT)", type=["csv", "txt"])
    manual = st.text_area(
        "Or paste matrix manually (rows, comma-separated values)", height=140
    )
    weights_text = st.text_input("Weights (comma-separated)", "")
    types_text = st.text_input(
        "Criterion types (0 = cost, 1 = benefit) (comma-separated)", ""
    )
    analyze_btn = st.button("Analyze", type="primary")

with col2:
    st.markdown("Preview / Instructions")
    st.markdown(
        "- Matrix: row = alternative, column = criterion\n"
        "- Example matrix:\n"
        "```\n"
        "25000,7.5,3\n"
        "30000,8.0,4\n"
        "27000,7.0,5\n"
        "```\n"
        "- Weights: e.g. `0.4,0.35,0.25` (if empty - equal weights)\n"
        "- Types: `0` = cost, `1` = benefit, e.g. `0,1,1`"
    )

if analyze_btn:
    try:
        matrix = parse_matrix(uploaded, manual)
        if matrix is None:
            st.error("No matrix. Please upload a file or paste the data manually.")
            st.stop()

        n_alts, n_crit = matrix.shape

        weights = parse_vector(weights_text, expected_len=n_crit, dtype=float)
        types = parse_vector(types_text, expected_len=n_crit, dtype=int)

        if weights is None:
            weights = np.ones(n_crit, dtype=float)
        if types is None:
            types = np.ones(n_crit, dtype=int)

        weights = np.array(weights, dtype=float)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / np.sum(weights)
        types = np.array(types, dtype=int)

        method_cls = METHOD_REGISTRY[method]
        scores, ranks = method_cls(matrix, weights, types).run()

        alt_names = [f"Alt {i + 1}" for i in range(n_alts)]
        res_df = pd.DataFrame(
            {
                "Alternative": alt_names,
                "Score": np.round(scores.astype(float), 6),
                "Rank": ranks,
            }
        ).sort_values("Rank")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("Results table")
            st.dataframe(res_df.reset_index(drop=True))
            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results (CSV)",
                data=csv_bytes,
                file_name=f"results_{method}.csv",
            )

        with right:
            st.markdown("Bar chart")
            st.plotly_chart(
                plot_results_bar(res_df, method, dark_mode), use_container_width=True
            )

        if do_sensitivity:
            st.markdown("---")
            st.markdown("Sensitivity analysis - removing single criteria:")
            ranks_list, corrs = sensitivity_analysis(
                ranks, matrix, weights, types, method_cls
            )
            sens_df = pd.DataFrame(
                {
                    "Removed Criterion": [f"C{j + 1}" for j in range(n_crit)],
                    "Spearman with reference": np.round(corrs, 4),
                }
            )
            st.dataframe(sens_df)

            methods_list = ["Ref."] + [f"w/o C{j + 1}" for j in range(n_crit)]
            ranks_matrix = [ranks] + ranks_list
            st.plotly_chart(
                plot_stability_line(methods_list, ranks_matrix, dark_mode),
                use_container_width=True,
            )

            fig_corr = px.line(
                sens_df,
                x="Removed Criterion",
                y="Spearman with reference",
                markers=True,
                title="Spearman (after removing the criterion)",
            )
            fig_corr.update_layout(
                template="plotly_dark" if dark_mode else "plotly_white",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        if show_corr:
            st.markdown("---")
            st.markdown("Quick method comparison (Spearman vs. selected method)")
            comparisons = {}
            for name, cls in METHOD_REGISTRY.items():
                _, other_ranks = cls(matrix, weights, types).run()
                comparisons[name] = round(spearman_corr(ranks, other_ranks), 4)
            comp_df = pd.DataFrame(
                {
                    "Method": list(comparisons.keys()),
                    "Spearman vs selected": list(comparisons.values()),
                }
            )
            st.dataframe(comp_df)
            fig_comp = px.bar(
                comp_df,
                x="Method",
                y="Spearman vs selected",
                text="Spearman vs selected",
                title="Comparison of methods (Spearman)",
                range_y=[0, 1],
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.success("Analysis completed successfully.")
    except Exception as exc: 
        st.exception(exc)
else:
    st.info("Enter your data, choose your settings and click 'Analyze'.")
