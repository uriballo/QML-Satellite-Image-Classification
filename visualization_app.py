from __future__ import annotations

import io
import os
from typing import Dict, List

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml  # for manual meta.yaml parsing when APIs are missing

# ------------------------------------------------------------------------------------
# App config
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="Flowsight – MLflow Explorer", layout="wide")

# ------------------------------------------------------------------------------------
# Sidebar – data source
# ------------------------------------------------------------------------------------
st.sidebar.header("1 · Data source")
tracking_dir = st.sidebar.text_input("MLflow tracking directory", value="./mlruns")

if not os.path.isdir(tracking_dir):
    st.sidebar.error("⚠️ Folder not found. Enter a valid `mlruns` path.")
    st.stop()

mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
client = mlflow.tracking.MlflowClient()

# ------------------------------------------------------------------------------------
# Experiment discovery – robust against ancient MLflow versions
# ------------------------------------------------------------------------------------

def discover_experiments(root: str) -> Dict[str, str]:
    """Return mapping {experiment_name: experiment_id}. Works even if APIs are absent."""
    # 1. Newest API (MLflow ≥1.30): client.search_experiments / list_experiments
    for attr in ("list_experiments", "search_experiments"):
        if hasattr(client, attr):
            try:
                exps = getattr(client, attr)()
                if exps:
                    return {e.name: e.experiment_id for e in exps}
            except Exception:
                pass  # fall through to folder scan

    # 2. Global helper (mlflow.search_experiments) in some versions
    if hasattr(mlflow, "search_experiments"):
        try:
            exps = mlflow.search_experiments(filter_string="")
            if exps:
                return {e.name: e.experiment_id for e in exps}
        except Exception:
            pass

    # 3. Brute‑force: parse meta.yaml files inside the tracking directory
    experiments: Dict[str, str] = {}
    for entry in os.scandir(root):
        if entry.is_dir() and entry.name.isdigit():  # each experiment dir is its id
            exp_id = entry.name
            meta_path = os.path.join(entry.path, "meta.yaml")
            name = f"Experiment {exp_id}"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf‑8") as f:
                        meta = yaml.safe_load(f) or {}
                    name = meta.get("name", name)
                except Exception:
                    pass
            experiments[name] = exp_id
    return experiments

experiments = discover_experiments(tracking_dir)
if not experiments:
    st.error("No experiments found in this tracking directory.")
    st.stop()

# Sidebar selection
exp_names = st.sidebar.multiselect(
    "Select experiment(s)", options=list(experiments.keys()), default=list(experiments.keys())
)
if not exp_names:
    st.stop()
exp_ids = [experiments[n] for n in exp_names]

# ------------------------------------------------------------------------------------
# Fetch runs & enrich with experiment name
# ------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading runs …")
def fetch_runs(eids: List[str]) -> pd.DataFrame:
    try:
        df_runs = mlflow.search_runs(experiment_ids=eids)
    except Exception as e:
        st.error(f"Failed to read runs via MLflow API → {e}")
        st.stop()
    df_runs["experiment_name"] = df_runs["experiment_id"].map({v: k for k, v in experiments.items()})
    return df_runs

df = fetch_runs(exp_ids)
if df.empty:
    st.warning("Selected experiments contain no runs.")
    st.stop()

# ------------------------------------------------------------------------------------
# Sidebar – run / column selection
# ------------------------------------------------------------------------------------
st.sidebar.header("2 · Run / column selection")
run_ids = st.sidebar.multiselect(
    "Runs (by run_id)", options=df["run_id"].tolist(), default=df["run_id"].tolist()
)
filtered = df[df["run_id"].isin(run_ids)].copy()

metric_cols = [c for c in filtered.columns if c.startswith("metrics.")]
param_cols = [c for c in filtered.columns if c.startswith("params.")]
tag_cols = [c for c in filtered.columns if c.startswith("tags.")]

# Helper to strip prefixes for UI
nice = lambda col: col.split(".", 1)[1] if "." in col else col

base_map: Dict[str, str] = {nice(c): c for c in metric_cols + param_cols + tag_cols}
base_map["experiment_name"] = "experiment_name"

# ------------------------------------------------------------------------------------
# Main UI – DataFrame explorer
# ------------------------------------------------------------------------------------
st.title("📊 Flowsight: MLflow result visualizer")

st.subheader("DataFrame explorer")
with st.expander("Show DataFrame", expanded=False):
    st.dataframe(filtered.drop(columns=[c for c in filtered.columns if c.startswith("tags.")]), hide_index=True)

# ------------------------------------------------------------------------------------
# Chart builder
# ------------------------------------------------------------------------------------
st.subheader("Chart builder")

col1, col2, col3 = st.columns(3)
with col1:
    x_display = st.selectbox("X‑axis", list(base_map.keys()), index=0)
with col2:
    y_display = st.selectbox("Y‑axis", list(base_map.keys()), index=1)
with col3:
    color_display = st.selectbox("Color / series", ["None"] + list(base_map.keys()), index=0)

x = base_map[x_display]
y = base_map[y_display]
color = base_map.get(color_display) if color_display != "None" else None

CHART_FUNCS: Dict[str, callable] = {
    "Scatter": px.scatter,
    "Line": px.line,
    "Bar": px.bar,
    "Box": px.box,
    "Violin": px.violin,
    "Strip": px.strip,
    "Histogram": px.histogram,
    "ECDF": px.ecdf,
    "Parallel Coordinates": px.parallel_coordinates,
}
chart_type = st.selectbox("Chart type", list(CHART_FUNCS.keys()), index=0)
fig_func = CHART_FUNCS[chart_type]

plot_kw = dict(
    data_frame=filtered,
    x=x,
    y=y,
    template="plotly_white",
    title=f"{chart_type}: {y_display} vs {x_display}",
)
if color:
    plot_kw["color"] = color

fig = fig_func(**plot_kw)
fig.update_layout(font_family="Helvetica", title_font_size=20, legend_title_text=color_display if color else "")

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------------
# Download
# ------------------------------------------------------------------------------------
st.markdown("### Export chart")
fmt = st.selectbox("Format", ["png", "svg", "pdf"], index=0)

buf = io.BytesIO()
fig.write_image(buf, format=fmt, width=1600, height=900, scale=2)

st.download_button(
    label=f"💾 Download {fmt.upper()}",
    data=buf.getvalue(),
    file_name=f"flowsight_chart.{fmt}",
    mime="image/png" if fmt == "png" else "image/svg+xml" if fmt == "svg" else "application/pdf",
)

st.caption("Built with Streamlit · MLflow · Plotly · 2025")
