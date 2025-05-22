from __future__ import annotations

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ------------------------------------------------------------------------------------
# Logging Configuration - Console Only
# ------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------
# App config
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="MLflow Explorer (Brute Force)", layout="wide")

# ------------------------------------------------------------------------------------
# Performance monitoring decorator
# ------------------------------------------------------------------------------------
def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

# ------------------------------------------------------------------------------------
# Brute Force MLflow Reader
# ------------------------------------------------------------------------------------
class BruteForceMLflowReader:
    """Direct file system reader for MLflow data"""
    
    def __init__(self, tracking_dir: str):
        self.tracking_dir = Path(tracking_dir)
        logger.info(f"Initialized BruteForceMLflowReader with tracking_dir: {tracking_dir}")
        
    @log_performance
    def list_experiments(self) -> Dict[str, Dict[str, Any]]:
        """List all experiments by reading meta.yaml files directly"""
        experiments = {}
        
        try:
            for exp_dir in self.tracking_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.name.isdigit():
                    exp_id = exp_dir.name
                    meta_path = exp_dir / "meta.yaml"
                    
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                meta = yaml.safe_load(f) or {}
                            
                            experiments[exp_id] = {
                                'experiment_id': exp_id,
                                'name': meta.get('name', f'Experiment {exp_id}'),
                                'artifact_location': meta.get('artifact_location', ''),
                                'lifecycle_stage': meta.get('lifecycle_stage', 'active'),
                                'creation_time': meta.get('creation_time', 0),
                                'last_update_time': meta.get('last_update_time', 0)
                            }
                            logger.debug(f"Loaded experiment {exp_id}: {meta.get('name')}")
                        except Exception as e:
                            logger.warning(f"Failed to read meta.yaml for experiment {exp_id}: {e}")
                            experiments[exp_id] = {
                                'experiment_id': exp_id,
                                'name': f'Experiment {exp_id}',
                                'lifecycle_stage': 'active'
                            }
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            
        logger.info(f"Found {len(experiments)} experiments")
        return experiments
    
    @log_performance
    def load_run(self, exp_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a single run's data"""
        run_path = self.tracking_dir / exp_id / run_id
        
        if not run_path.exists():
            logger.warning(f"Run path does not exist: {run_path}")
            return None
            
        run_data = {
            'run_id': run_id,
            'experiment_id': exp_id,
            'status': 'FINISHED',
            'start_time': None,
            'end_time': None,
            'metrics': {},
            'params': {},
            'tags': {}
        }
        
        try:
            # Load meta.yaml for run metadata
            meta_path = run_path / "meta.yaml"
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = yaml.safe_load(f) or {}
                run_data['status'] = meta.get('status', 'FINISHED')
                run_data['start_time'] = meta.get('start_time')
                run_data['end_time'] = meta.get('end_time')
                run_data['lifecycle_stage'] = meta.get('lifecycle_stage', 'active')
            
            # Load params
            params_path = run_path / "params"
            if params_path.exists():
                for param_file in params_path.iterdir():
                    if param_file.is_file():
                        try:
                            param_name = param_file.name
                            with open(param_file, 'r', encoding='utf-8') as f:
                                param_value = f.read().strip()
                            run_data['params'][param_name] = param_value
                        except Exception as e:
                            logger.debug(f"Failed to read param {param_file}: {e}")
            
            # Load metrics (final values)
            metrics_path = run_path / "metrics"
            if metrics_path.exists():
                for metric_file in metrics_path.iterdir():
                    if metric_file.is_file():
                        try:
                            metric_name = metric_file.name
                            with open(metric_file, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                if lines:
                                    # Get the last value
                                    last_line = lines[-1].strip()
                                    parts = last_line.split()
                                    if len(parts) >= 3:
                                        value = float(parts[1])
                                        run_data['metrics'][metric_name] = value
                        except Exception as e:
                            logger.debug(f"Failed to read metric {metric_file}: {e}")
            
            # Load tags
            tags_path = run_path / "tags"
            if tags_path.exists():
                for tag_file in tags_path.iterdir():
                    if tag_file.is_file():
                        try:
                            tag_name = tag_file.name
                            with open(tag_file, 'r', encoding='utf-8') as f:
                                tag_value = f.read().strip()
                            run_data['tags'][tag_name] = tag_value
                        except Exception as e:
                            logger.debug(f"Failed to read tag {tag_file}: {e}")
            
            return run_data
            
        except Exception as e:
            logger.error(f"Error loading run {run_id}: {e}")
            return None
    
    @log_performance
    def load_runs_parallel(self, experiment_ids: List[str], max_workers: int = 10) -> pd.DataFrame:
        """Load all runs from multiple experiments in parallel"""
        all_runs = []
        total_runs = 0
        futures_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit all run loading tasks
            for exp_id in experiment_ids:
                exp_path = self.tracking_dir / exp_id
                if exp_path.exists():
                    for run_dir in exp_path.iterdir():
                        if run_dir.is_dir() and len(run_dir.name) == 32:  # MLflow run IDs are 32 chars
                            future = executor.submit(self.load_run, exp_id, run_dir.name)
                            futures.append(future)
                            futures_count += 1
                else:
                    logger.warning(f"Experiment path does not exist: {exp_path}")
            
            # Handle case where no runs found
            if not futures:
                logger.warning("No run directories found in selected experiments")
                return pd.DataFrame(columns=['run_id', 'experiment_id', 'status', 'start_time', 'end_time'])
            
            # Collect results with progress bar
            progress_bar = st.progress(0)
            completed = 0
            
            for future in as_completed(futures):
                completed += 1
                progress_bar.progress(completed / len(futures))
                
                try:
                    run_data = future.result()
                    if run_data:
                        all_runs.append(run_data)
                        total_runs += 1
                except Exception as e:
                    logger.error(f"Failed to load run: {e}")
            
            progress_bar.empty()
        
        logger.info(f"Loaded {total_runs} runs from {len(experiment_ids)} experiments (checked {futures_count} directories)")
        
        # Convert to DataFrame
        df = self._runs_to_dataframe(all_runs)
        return df
    
    def _runs_to_dataframe(self, runs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert run dictionaries to a DataFrame matching MLflow format"""
        if not runs:
            # Return empty DataFrame with expected columns
            logger.warning("No runs to convert to DataFrame")
            return pd.DataFrame(columns=['run_id', 'experiment_id', 'status', 'start_time', 'end_time'])
        
        rows = []
        
        for run in runs:
            row = {
                'run_id': run['run_id'],
                'experiment_id': run['experiment_id'],
                'status': run.get('status', 'FINISHED'),
                'start_time': run.get('start_time'),
                'end_time': run.get('end_time')
            }
            
            # Add metrics with 'metrics.' prefix
            for metric_name, value in run.get('metrics', {}).items():
                row[f'metrics.{metric_name}'] = value
            
            # Add params with 'params.' prefix
            for param_name, value in run.get('params', {}).items():
                row[f'params.{param_name}'] = value
            
            # Add tags with 'tags.' prefix
            for tag_name, value in run.get('tags', {}).items():
                row[f'tags.{tag_name}'] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ensure essential columns exist
        for col in ['run_id', 'experiment_id', 'status', 'start_time', 'end_time']:
            if col not in df.columns:
                df[col] = None
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    @log_performance
    def load_metric_history(self, exp_id: str, run_id: str, metric_name: str) -> Optional[pd.DataFrame]:
        """Load full history of a metric including all epochs/steps"""
        metric_path = self.tracking_dir / exp_id / run_id / "metrics" / metric_name
        
        if not metric_path.exists():
            logger.warning(f"Metric file not found: {metric_path}")
            return None
        
        try:
            data = []
            with open(metric_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        timestamp = int(parts[0])
                        value = float(parts[1])
                        step = int(parts[2])
                        data.append({
                            'timestamp': timestamp,
                            'value': value,
                            'step': step,
                            'epoch': step,  # Using step as epoch
                            'metric': metric_name,
                            'run_id': run_id,
                            'experiment_id': exp_id
                        })
            
            if data:
                df = pd.DataFrame(data)
                logger.debug(f"Loaded {len(df)} entries for metric {metric_name} in run {run_id}")
                return df
            return None
            
        except Exception as e:
            logger.error(f"Error loading metric history for {metric_name}: {e}")
            return None
    
    @log_performance
    def load_metric_histories_batch(self, run_metrics: List[Tuple[str, str, str]], 
                                   max_workers: int = 10) -> pd.DataFrame:
        """Load multiple metric histories in parallel
        
        Args:
            run_metrics: List of (exp_id, run_id, metric_name) tuples
        """
        all_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.load_metric_history, exp_id, run_id, metric): 
                (exp_id, run_id, metric) 
                for exp_id, run_id, metric in run_metrics
            }
            
            for future in as_completed(futures):
                exp_id, run_id, metric = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_data.append(df)
                except Exception as e:
                    logger.error(f"Failed to load metric history: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total metric history entries")
            return combined
        
        return pd.DataFrame()

# ------------------------------------------------------------------------------------
# Sidebar – data source
# ------------------------------------------------------------------------------------
st.sidebar.header("1 · Data source")
tracking_dir = st.sidebar.text_input("MLflow tracking directory", value="./mlruns")

if not os.path.isdir(tracking_dir):
    st.sidebar.error("⚠️ Folder not found. Enter a valid `mlruns` path.")
    st.stop()

# Initialize reader
reader = BruteForceMLflowReader(tracking_dir)

# Load experiments
experiments_dict = reader.list_experiments()
experiments = {exp['name']: exp_id for exp_id, exp in experiments_dict.items()}

if not experiments:
    st.error("No experiments found in this tracking directory.")
    st.stop()

# Experiment selection
exp_names = st.sidebar.multiselect(
    "Select experiment(s)", 
    options=list(experiments.keys()), 
    default=list(experiments.keys())[:5]  # Default to first 5
)

if not exp_names:
    st.stop()

exp_ids = [experiments[n] for n in exp_names]

# ------------------------------------------------------------------------------------
# Load runs with caching
# ------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading runs...", ttl=300)  # 5 min TTL
def fetch_runs_cached(exp_ids: List[str], tracking_dir: str, experiments_map: Dict[str, str]) -> pd.DataFrame:
    """Cached function to load runs"""
    reader = BruteForceMLflowReader(tracking_dir)
    df = reader.load_runs_parallel(exp_ids, max_workers=20)
    
    # Check if DataFrame has required columns
    if not df.empty and 'experiment_id' in df.columns:
        # Add experiment names - use the passed experiments_map
        reverse_map = {v: k for k, v in experiments_map.items()}
        df['experiment_name'] = df['experiment_id'].map(reverse_map)
        
        # Fill any missing experiment names
        missing_names = df['experiment_name'].isna()
        if missing_names.any():
            logger.warning(f"Found {missing_names.sum()} runs with unmapped experiment IDs")
            df.loc[missing_names, 'experiment_name'] = df.loc[missing_names, 'experiment_id'].apply(
                lambda x: f"Experiment {x}"
            )
    else:
        logger.warning("DataFrame is empty or missing experiment_id column")
        # Ensure experiment_name column exists even if empty
        df['experiment_name'] = None
    
    return df

# Load runs
with st.spinner("Loading runs..."):
    start_time = time.time()
    df = fetch_runs_cached(exp_ids, tracking_dir, experiments)  # Pass experiments dict
    load_time = time.time() - start_time
    logger.info(f"Loaded {len(df)} runs in {load_time:.2f}s")
    
    # Debug info
    if df.empty:
        st.warning("No runs found in selected experiments. Please check:")
        st.write("- Selected experiment IDs:", exp_ids)
        st.write("- Tracking directory exists:", Path(tracking_dir).exists())
        for exp_id in exp_ids:
            exp_path = Path(tracking_dir) / exp_id
            st.write(f"- Experiment {exp_id} path exists:", exp_path.exists())
            if exp_path.exists():
                run_dirs = [d for d in exp_path.iterdir() if d.is_dir() and len(d.name) == 32]
                st.write(f"  - Number of run directories: {len(run_dirs)}")
                if len(run_dirs) > 0:
                    st.write(f"  - Sample run IDs: {[d.name[:8] + '...' for d in run_dirs[:3]]}")

if df.empty:
    st.warning("Selected experiments contain no runs.")
    st.stop()

# Removed loading stats as requested

# ------------------------------------------------------------------------------------
# Process columns
# ------------------------------------------------------------------------------------
metric_cols = sorted([c for c in df.columns if c.startswith("metrics.")])
param_cols = sorted([c for c in df.columns if c.startswith("params.")])
tag_cols = sorted([c for c in df.columns if c.startswith("tags.")])

logger.info(f"Found {len(metric_cols)} metrics, {len(param_cols)} params, {len(tag_cols)} tags")

# Helper to strip prefixes
nice = lambda col: col.split(".", 1)[1] if "." in col else col

# Initialize for later use
filtered = df.copy()
tag_renames = {}
param_value_renames = {}

# ------------------------------------------------------------------------------------
# Column Management
# ------------------------------------------------------------------------------------
st.sidebar.header("2 · Column Management")
with st.sidebar.expander("Column Filtering & Renaming", expanded=False):
    # Create separate lists for metrics, params, and tags
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
        index=0
    )
    
    col1, col2 = st.columns(2)
    fig_width = col1.number_input("Figure Width (px)", min_value=600, max_value=2000, value=1200)
    fig_height = col2.number_input("Figure Height (px)", min_value=400, max_value=1200, value=700)

# ------------------------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------------------------
st.title("MLflow Result Visualizer (Brute Force)")

# DataFrame explorer
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

# Base mapping for chart builder
base_map: Dict[str, str] = {nice(c): c for c in metric_cols + param_cols + tag_cols}
base_map["experiment_name"] = "experiment_name"

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

# Apply parameter value renaming for display if any
if param_value_renames:
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
# Epoch-based metrics visualization (FIXED)
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
        options=["Best run (max final value)", "Best run (min final value)", "Best run (avg value)", 
                 "Max value per epoch", "Min value per epoch", "Average per epoch", "All runs"],
        index=0,
        horizontal=True,
        help="'Best run' methods show all epochs from the run with the best final/average metric value per group"
    )
    
    # Cached function for epoch data
    @st.cache_data(show_spinner="Loading epoch metrics...", ttl=300)
    def fetch_epoch_metrics_batch(run_info: List[Tuple[str, str]], metric_keys: List[str], 
                                tracking_dir: str) -> pd.DataFrame:
        """Batch load epoch metrics for multiple runs"""
        reader = BruteForceMLflowReader(tracking_dir)
        
        # Prepare list of (exp_id, run_id, metric) tuples
        run_metrics = []
        for exp_id, run_id in run_info:
            for metric in metric_keys:
                run_metrics.append((exp_id, run_id, metric))
        
        logger.info(f"Loading {len(run_metrics)} metric histories...")
        return reader.load_metric_histories_batch(run_metrics, max_workers=20)
    
    # Get run info for selected (filtered) runs only
    run_info = [(row['experiment_id'], row['run_id']) for _, row in filtered.iterrows()]
    
    # Fetch epoch data
    epoch_df = fetch_epoch_metrics_batch(run_info, selected_metrics, tracking_dir)
    
    if not epoch_df.empty:
        # IMPORTANT: Merge with filtered dataframe to get parameters and experiment names
        # This ensures we only show data for the filtered runs
        epoch_df = epoch_df.merge(
            filtered[['run_id', 'experiment_id', 'experiment_name'] + param_cols],
            on=['run_id', 'experiment_id'],
            how='inner'  # Inner join ensures we only keep epochs for filtered runs
        )
        
        # Apply parameter value renaming to epoch data
        for (param_col, old_value), new_value in param_value_renames.items():
            if param_col in epoch_df.columns:
                epoch_df[param_col] = epoch_df[param_col].replace(old_value, new_value)
        
        # Group by control
        group_options = ["run_id", "experiment_name"] + [nice(p) for p in param_cols]
        group_by = st.selectbox(
            "Group lines by:", 
            options=group_options,
            index=1  # Default to experiment_name
        )
        
        # Convert group_by to actual column name if it's a param
        if group_by in [nice(p) for p in param_cols]:
            group_by_col = f"params.{group_by}"
        else:
            group_by_col = group_by
        
        # Create epoch-based visualization
        fig = go.Figure()
        
        # Track which groups we're displaying for debug info
        displayed_groups = []
        
        # Use epoch_df as the base plot data
        plot_data = epoch_df
        
        # Apply aggregation based on method
        if "Best run" in aggregation_method:
            # For "best run" methods, find the best run per group and show all its epochs
            for metric in selected_metrics:
                metric_data = plot_data[plot_data['metric'] == metric]
                
                for group_val, group_data in metric_data.groupby(group_by_col):
                    # Find the best run in this group
                    try:
                        if "max final" in aggregation_method:
                            # Get final value for each run
                            final_values = group_data.groupby('run_id')['value'].last()
                            if final_values.empty:
                                continue
                            best_run_id = final_values.idxmax()
                        elif "min final" in aggregation_method:
                            final_values = group_data.groupby('run_id')['value'].last()
                            if final_values.empty:
                                continue
                            best_run_id = final_values.idxmin()
                        elif "avg value" in aggregation_method:
                            # Get average value for each run
                            avg_values = group_data.groupby('run_id')['value'].mean()
                            if avg_values.empty:
                                continue
                            best_run_id = avg_values.idxmax()
                        
                        # Get all epochs for the best run
                        best_run_data = group_data[group_data['run_id'] == best_run_id].sort_values('epoch')
                        
                        if best_run_data.empty:
                            continue
                    except Exception as e:
                        logger.error(f"Error finding best run for metric {metric}: {e}")
                        continue
                    
                    # Add trace for this group's best run
                    trace_name = f"{metric} - {group_val}"
                    fig.add_trace(go.Scatter(
                        x=best_run_data['epoch'],
                        y=best_run_data['value'],
                        mode='lines+markers',
                        name=trace_name,
                        hovertemplate=(
                            f"Metric: {metric}<br>"
                            f"Group: {group_val}<br>"
                            f"Run: {best_run_id[:8]}...<br>"
                            "Epoch: %{x}<br>"
                            "Value: %{y}<extra></extra>"
                        )
                    ))
                    
                    # Track displayed group info
                    displayed_groups.append({
                        'Metric': metric,
                        'Group': str(group_val),
                        'Best Run ID': best_run_id[:8] + '...',
                        'Final Value': best_run_data['value'].iloc[-1] if len(best_run_data) > 0 else None,
                        'Num Epochs': len(best_run_data)
                    })
        
        elif aggregation_method in ["Max value per epoch", "Min value per epoch", "Average per epoch"]:
            # Original per-epoch aggregation logic
            for metric in selected_metrics:
                metric_data = plot_data[plot_data['metric'] == metric]
                
                for group_val, group_data in metric_data.groupby(group_by_col):
                    # Group by epoch and apply aggregation
                    if aggregation_method == "Max value per epoch":
                        agg_data = group_data.groupby('epoch')['value'].max().reset_index()
                    elif aggregation_method == "Min value per epoch":
                        agg_data = group_data.groupby('epoch')['value'].min().reset_index()
                    elif aggregation_method == "Average per epoch":
                        agg_data = group_data.groupby('epoch')['value'].mean().reset_index()
                    
                    agg_data = agg_data.sort_values('epoch')
                    
                    trace_name = f"{metric} - {group_val}"
                    fig.add_trace(go.Scatter(
                        x=agg_data['epoch'],
                        y=agg_data['value'],
                        mode='lines+markers',
                        name=trace_name,
                        hovertemplate="Epoch: %{x}<br>Value: %{y}<br>Group: " + str(group_val) + "<extra></extra>"
                    ))
                    
                    # Track displayed group info
                    num_runs = group_data['run_id'].nunique()
                    displayed_groups.append({
                        'Metric': metric,
                        'Group': str(group_val),
                        'Aggregation': aggregation_method,
                        'Num Runs': num_runs,
                        'Final Value': agg_data['value'].iloc[-1] if len(agg_data) > 0 else None
                    })
        
        else:  # "All runs"
            # Show all runs without aggregation
            for metric in selected_metrics:
                metric_data = plot_data[plot_data['metric'] == metric]
                
                for group_val, group_data in metric_data.groupby(group_by_col):
                    # Plot each run separately
                    for run_id, run_data in group_data.groupby('run_id'):
                        run_data = run_data.sort_values('epoch')
                        
                        fig.add_trace(go.Scatter(
                            x=run_data['epoch'],
                            y=run_data['value'],
                            mode='lines+markers',
                            name=f"{metric} - {group_val} - {run_id[:8]}",
                            hovertemplate=(
                                f"Run: {run_id[:8]}...<br>"
                                "Epoch: %{x}<br>"
                                "Value: %{y}<extra></extra>"
                            )
                        ))
                    
                    num_runs = group_data['run_id'].nunique()
                    if 'Metric' not in [g for g in displayed_groups if g.get('Group') == str(group_val) and g.get('Metric') == metric]:
                        displayed_groups.append({
                            'Metric': metric,
                            'Group': str(group_val),
                            'Num Runs': num_runs
                        })
        
        # Update layout
        title = f"Metrics by Epoch ({aggregation_method})"
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Value",
            template=plot_template,
            legend_title=f"Metric - {group_by}",
            width=fig_width,
            height=fig_height,
            font_family="Helvetica"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Debug info with group table
        with st.expander("Debug: Displayed Groups", expanded=False):
            if displayed_groups:
                # Remove duplicates and create DataFrame
                unique_groups = []
                seen = set()
                for g in displayed_groups:
                    key = (g.get('Metric'), g.get('Group'))
                    if key not in seen:
                        seen.add(key)
                        unique_groups.append(g)
                
                groups_df = pd.DataFrame(unique_groups)
                st.dataframe(groups_df, hide_index=True)
                
                st.write(f"Total groups displayed: {len(unique_groups)}")
                st.write(f"Total epoch entries in data: {len(epoch_df)}")
                st.write(f"Unique runs in epoch data: {epoch_df['run_id'].nunique()}")
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
    file_name=f"mlflow_chart.{fmt}",
    mime="image/png" if fmt == "png" else "image/svg+xml" if fmt == "svg" else "application/pdf",
)

# ------------------------------------------------------------------------------------
# Cache management
# ------------------------------------------------------------------------------------
with st.sidebar.expander("⚡ Performance", expanded=False):
    st.markdown("""
    **Cache management:**
    - Data is cached for 5 minutes
    - Click below to clear cache
    """)
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        logger.info("Cache cleared by user")

st.caption("Built with Streamlit · Plotly · 2025")