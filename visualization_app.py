from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml  # for manual meta.yaml parsing when APIs are missing

# ------------------------------------------------------------------------------------
# App config
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="MLflow Explorer", layout="wide")

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
# Experiment discovery
# ------------------------------------------------------------------------------------

def discover_experiments(root: str) -> Dict[str, str]:
    """Return mapping {experiment_name: experiment_id}. Works even if APIs are absent."""
    for attr in ("list_experiments", "search_experiments"):
        if hasattr(client, attr):
            try:
                exps = getattr(client, attr)()
                if exps:
                    return {e.name: e.experiment_id for e in exps}
            except Exception:
                pass  # fall through to folder scan

    if hasattr(mlflow, "search_experiments"):
        try:
            exps = mlflow.search_experiments(filter_string="")
            if exps:
                return {e.name: e.experiment_id for e in exps}
        except Exception:
            pass

    # Brute‑force: parse meta.yaml files inside the tracking directory
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
# Fetch epoch data for runs
# ------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading epoch metrics...")
def fetch_epoch_metrics(run_id: str, metric_keys: List[str]) -> Optional[pd.DataFrame]:
    """Fetch detailed metric history for a run including epochs/steps"""
    try:
        metrics_data = []
        for metric_key in metric_keys:
            history = client.get_metric_history(run_id, metric_key)
            for m in history:
                metrics_data.append({
                    "run_id": run_id,
                    "metric": metric_key,
                    "epoch": m.step,
                    "value": m.value,
                    "timestamp": m.timestamp
                })
        if metrics_data:
            return pd.DataFrame(metrics_data)
        return None
    except Exception as e:
        st.warning(f"Failed to fetch epoch metrics for run {run_id}: {e}")
        return None

# ------------------------------------------------------------------------------------
# Process run data
# ------------------------------------------------------------------------------------
# Get all runs
run_ids = df["run_id"].tolist()
filtered = df.copy()

metric_cols = [c for c in filtered.columns if c.startswith("metrics.")]
param_cols = [c for c in filtered.columns if c.startswith("params.")]
tag_cols = [c for c in filtered.columns if c.startswith("tags.")]

# Helper to strip prefixes for UI
nice = lambda col: col.split(".", 1)[1] if "." in col else col

base_map: Dict[str, str] = {nice(c): c for c in metric_cols + param_cols + tag_cols}
base_map["experiment_name"] = "experiment_name"

# Initialize tag_renames dictionary for column renaming
tag_renames = {}

# ------------------------------------------------------------------------------------
# Tag filtering and renaming
# ------------------------------------------------------------------------------------
st.sidebar.header("2 · Column Management")
with st.sidebar.expander("Column Filtering & Renaming", expanded=False):
    # Create separate lists for metrics, params, and tags for better organization
    metric_options = [nice(c) for c in metric_cols]
    param_options = [nice(c) for c in param_cols]
    tag_options = [nice(c) for c in tag_cols]
    
    # Tabs for metrics, params, and tags
    tab1, tab2, tab3 = st.tabs(["Metrics", "Parameters", "Tags"])
    
    # Metrics tab
    with tab1:
        selected_metrics_display = st.multiselect("Metrics to display", metric_options, default=metric_options)
        
        # Metric renaming
        if selected_metrics_display:
            st.subheader("Rename Metrics")
            cols = st.columns(2)
            for i, metric in enumerate(selected_metrics_display):
                original = f"metrics.{metric}"
                new_name = cols[i % 2].text_input(f"Rename {metric}", value=metric, key=f"rename_metric_{metric}")
                if new_name != metric:
                    tag_renames[original] = new_name
    
    # Parameters tab
    with tab2:
        selected_params_display = st.multiselect("Parameters to display", param_options, default=param_options)
        
        # Parameter renaming
        if selected_params_display:
            st.subheader("Rename Parameters")
            cols = st.columns(2)
            for i, param in enumerate(selected_params_display):
                original = f"params.{param}"
                new_name = cols[i % 2].text_input(f"Rename {param}", value=param, key=f"rename_param_{param}")
                if new_name != param:
                    tag_renames[original] = new_name
    
    # Tags tab
    with tab3:
        selected_tags = st.multiselect("Tags to display", tag_options, default=[])
        
        # Tag renaming
        if selected_tags:
            st.subheader("Rename Tags")
            cols = st.columns(2)
            for i, tag in enumerate(selected_tags):
                original = f"tags.{tag}"
                new_name = cols[i % 2].text_input(f"Rename {tag}", value=tag, key=f"rename_tag_{tag}")
                if new_name != tag:
                    tag_renames[original] = new_name
                    
    # Combine all selected columns
    selected_metrics_cols = [f"metrics.{m}" for m in selected_metrics_display]
    selected_params_cols = [f"params.{p}" for p in selected_params_display]
    selected_tags_cols = [f"tags.{t}" for t in selected_tags]

# ------------------------------------------------------------------------------------
# Parameter value renaming
# ------------------------------------------------------------------------------------
st.sidebar.header("3 · Parameter Value Renaming")
with st.sidebar.expander("Rename Parameter Values", expanded=False):
    st.info("Select parameters and rename their values (e.g., rename '32' to '32px')")
    
    # Select parameters to rename values for
    param_options = [nice(p) for p in param_cols if not p.endswith("run_timestamp")]
    
    # Allow selecting multiple parameters
    selected_params = st.multiselect("Parameters to rename values", options=param_options)
    
    # Parameter value renaming
    param_value_renames = {}
    
    # Show rename options for each selected parameter
    for selected_param in selected_params:
        param_col = f"params.{selected_param}"
        unique_values = filtered[param_col].unique().tolist()
        
        st.subheader(f"Rename values for {selected_param}")
        cols = st.columns(2)
        for i, value in enumerate(unique_values):
            new_value = cols[i % 2].text_input(
                f"Rename '{value}'", 
                value=value, 
                key=f"rename_val_{selected_param}_{value}"
            )
            if new_value != value:
                param_value_renames[(param_col, value)] = new_value

# ------------------------------------------------------------------------------------
# Param filtering for plots
# ------------------------------------------------------------------------------------
st.sidebar.header("4 · Parameter Filtering")
param_filters = {}
with st.sidebar.expander("Filter by Parameters", expanded=False):
    for param in [nice(p) for p in param_cols]:
        # Skip run_timestamp
        if param == "run_timestamp":
            continue
            
        param_values = filtered[f"params.{param}"].unique().tolist()
        if len(param_values) > 1:  # Only show params with multiple values
            selected_values = st.multiselect(
                f"Filter by {param}", 
                options=param_values,
                default=param_values
            )
            if selected_values and len(selected_values) < len(param_values):
                param_filters[f"params.{param}"] = selected_values

# Apply parameter filters
for param, values in param_filters.items():
    filtered = filtered[filtered[param].isin(values)]

# ------------------------------------------------------------------------------------
# Plot Style and Figure Size
# ------------------------------------------------------------------------------------
st.sidebar.header("5 · Plot Customization")
with st.sidebar.expander("Plot Style & Size", expanded=False):
    plot_template = st.selectbox(
        "Plot Template",
        options=["seaborn", "plotly_white", "plotly", "plotly_dark", "ggplot2", "simple_white"],
        index=0  # Default to seaborn
    )
    
    # Seaborn-specific color palettes (only show if seaborn template is selected)
    if plot_template == "seaborn":
        color_palette = st.selectbox(
            "Color Palette",
            options=["default", "deep", "muted", "pastel", "bright", "dark", "colorblind", 
                    "viridis", "plasma", "inferno", "magma", "cividis", "rocket",
                    "Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"],
            index=0
        )
    
    col1, col2 = st.columns(2)
    fig_width = col1.number_input("Figure Width (px)", min_value=600, max_value=2000, value=1200)
    fig_height = col2.number_input("Figure Height (px)", min_value=400, max_value=1200, value=700)

# ------------------------------------------------------------------------------------
# Main UI – DataFrame explorer
# ------------------------------------------------------------------------------------
st.title("MLflow result visualizer")

st.subheader("DataFrame explorer")
with st.expander("Show DataFrame", expanded=False):
    # Apply tag renaming for display
    display_df = filtered.copy()
    
    # Apply tag renaming
    for old_name, new_name in tag_renames.items():
        if old_name in display_df.columns:
            display_df = display_df.rename(columns={old_name: new_name})
    
    # Apply parameter value renaming
    for (param_col, old_value), new_value in param_value_renames.items():
        if param_col in display_df.columns:
            display_df[param_col] = display_df[param_col].replace(old_value, new_value)
    
    # Filter columns based on selections
    cols_to_show = selected_metrics_cols + selected_params_cols + selected_tags_cols + ["experiment_name", "run_id"]
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    
    if cols_to_show:
        st.dataframe(display_df[cols_to_show], hide_index=True)
    else:
        st.dataframe(display_df, hide_index=True)

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
    template=plot_template,
    title=f"{chart_type}: {y_display} vs {x_display}",
    width=fig_width,
    height=fig_height
)

# Apply seaborn color palette if selected
if plot_template == "seaborn" and color_palette != "default":
    plot_kw["color_discrete_sequence"] = px.colors.sequential.__getattribute__(color_palette) \
        if hasattr(px.colors.sequential, color_palette) \
        else px.colors.qualitative.__getattribute__(color_palette) \
        if hasattr(px.colors.qualitative, color_palette) \
        else None

# Apply parameter value renaming for display if any parameter values are renamed
if param_value_renames:
    # Create a copy of the dataframe with renamed values for display only
    plot_df = filtered.copy()
    for (p_col, old_value), new_value in param_value_renames.items():
        if p_col in [x, y, color]:
            plot_df[p_col] = plot_df[p_col].replace(old_value, new_value)
    plot_kw["data_frame"] = plot_df

if color:
    plot_kw["color"] = color

fig = fig_func(**plot_kw)
fig.update_layout(font_family="Helvetica", title_font_size=20, legend_title_text=color_display if color else "")

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------------
# Epoch-based metrics visualization
# ------------------------------------------------------------------------------------
st.subheader("Epoch-based Metrics Visualization")

# Get all metric keys (without 'metrics.' prefix)
metric_keys = [nice(m) for m in metric_cols]

# Select metrics to visualize
selected_metrics = st.multiselect(
    "Select metrics to visualize by epoch",
    options=metric_keys,
    default=metric_keys[:1] if metric_keys else []
)

if selected_metrics:
    # Add aggregation options
    aggregation_method = st.radio(
        "Aggregation method",
        options=["Max value", "Min value", "Average", "Median", "All values"],
        index=0,  # Default to max
        horizontal=True
    )
    
    # Fetch epoch data for all selected runs
    all_epoch_data = []
    for run_id in run_ids:
        run_metrics = [f"metrics.{m}" for m in selected_metrics]
        # Get run info for filtering and display
        run_info = filtered[filtered["run_id"] == run_id].iloc[0]
        
        # Check if this run should be included based on param filters
        include_run = True
        for param, values in param_filters.items():
            if run_info[param] not in values:
                include_run = False
                break
                
        if include_run:
            epoch_df = fetch_epoch_metrics(run_id, selected_metrics)
            if epoch_df is not None:
                # Add run parameters as columns for filtering/grouping
                for p in param_cols:
                    param_name = nice(p)
                    param_value = run_info[p]
                    
                    # Apply parameter value renaming if applicable
                    for (p_col, old_val), new_val in param_value_renames.items():
                        if p_col == f"params.{param_name}" and old_val == param_value:
                            param_value = new_val
                            break
                    
                    epoch_df[param_name] = param_value
                
                # Add experiment name
                epoch_df["experiment_name"] = run_info["experiment_name"]
                
                all_epoch_data.append(epoch_df)
    
    if all_epoch_data:
        epoch_df = pd.concat(all_epoch_data)
        
        # Group by control
        group_by = st.selectbox(
            "Group lines by:", 
            options=["run_id", "experiment_name"] + [nice(p) for p in param_cols],
            index=0
        )
        
        # Create epoch-based visualization
        fig = go.Figure()
        
        # Apply aggregation if not showing all values
        if aggregation_method != "All values":
            aggregated_data = []
            
            for metric in selected_metrics:
                # Filter data for this metric
                metric_data = epoch_df[epoch_df["metric"] == metric]
                
                # Group by epoch and the selected grouping variable
                grouped = metric_data.groupby(["epoch", group_by])
                
                # Apply the selected aggregation
                if aggregation_method == "Max value":
                    agg_data = grouped["value"].max().reset_index()
                elif aggregation_method == "Min value":
                    agg_data = grouped["value"].min().reset_index()
                elif aggregation_method == "Average":
                    agg_data = grouped["value"].mean().reset_index()
                elif aggregation_method == "Median":
                    agg_data = grouped["value"].median().reset_index()
                
                # Add metric name back
                agg_data["metric"] = metric
                aggregated_data.append(agg_data)
            
            if aggregated_data:
                # Replace the original data with aggregated data
                epoch_df = pd.concat(aggregated_data)
        
        # Group data for visualization
        for metric in selected_metrics:
            metric_data = epoch_df[epoch_df["metric"] == metric]
            
            # Group by the selected parameter
            for group_val, group_data in metric_data.groupby(group_by):
                # Sort by epoch
                group_data = group_data.sort_values("epoch")
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=group_data["epoch"],
                    y=group_data["value"],
                    mode="lines+markers",
                    name=f"{metric} - {group_val}",
                    hovertemplate="Epoch: %{x}<br>Value: %{y}<br>Group: " + str(group_val)
                ))
        
        # Apply seaborn color palette if selected
        if plot_template == "seaborn" and color_palette != "default":
            if hasattr(px.colors.sequential, color_palette):
                colors = px.colors.sequential.__getattribute__(color_palette)
            elif hasattr(px.colors.qualitative, color_palette):
                colors = px.colors.qualitative.__getattribute__(color_palette)
            else:
                colors = None
                
            if colors:
                for i, trace in enumerate(fig.data):
                    fig.data[i].line.color = colors[i % len(colors)]
        
        fig.update_layout(
            title=f"Metrics by Epoch ({aggregation_method})" if aggregation_method != "All values" else "Metrics by Epoch",
            xaxis_title="Epoch",
            yaxis_title="Value",
            template=plot_template,
            legend_title=f"Metric - {group_by}",
            width=fig_width,
            height=fig_height
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No epoch data available for the selected runs and metrics.")

# ------------------------------------------------------------------------------------
# Download
# ------------------------------------------------------------------------------------
st.markdown("### Export chart")
fmt = st.selectbox("Format", ["png", "svg", "pdf"], index=0)

buf = io.BytesIO()
fig.write_image(buf, format=fmt, width=fig_width, height=fig_height, scale=2)

st.download_button(
    label=f"💾 Download {fmt.upper()}",
    data=buf.getvalue(),
    file_name=f"flowsight_chart.{fmt}",
    mime="image/png" if fmt == "png" else "image/svg+xml" if fmt == "svg" else "application/pdf",
)

st.caption("Built with Streamlit · MLflow · Plotly · 2025")