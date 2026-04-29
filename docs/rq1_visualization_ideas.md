## RQ1 Visualization Ideas Handoff

This document proposes visualization styles for RQ1 results with multiple metrics:

- CKA
- PWCCA
- k-NN overlap
- null distribution summaries (noise floor, p-values)

The goal is to help another developer build clear, modern, and interactive outputs beyond basic static Matplotlib plots.

---

## 1) What we need to visualize

For each modality pair and layer, we will have:

- observed metric value
- null summary (`null_mean`, `null_std`)
- p-value
- optional effect-size values (`delta_vs_null_mean`, z-score)

Across the full run, we also have:

- multiple metrics
- multiple modality pairs
- many layers
- potentially many scenes or subsets

So the visualization challenge is multi-dimensional.

---

## 2) Core figure set (minimum useful package)

### A) Per-metric layer curves

For each metric:

- x-axis: layer index
- y-axis: metric value
- one line per modality pair
- optional confidence band from null (`null_mean +/- null_std`)

Why:

- fast global view of where convergence emerges in depth.

### B) Significance heatmap

- rows: modality pairs
- columns: layers
- values: `-log10(p_value)` or significance mask

Why:

- compact view of where signal is strong.

### C) Pair comparison radar or grouped bars (at selected layers)

- selected early/middle/late layers
- compare all metrics for one pair, or all pairs for one metric

Why:

- easier interpretation for presentations.

### D) Null vs observed distribution panels

For selected `(pair, layer, metric)`:

- histogram/KDE of null draws
- vertical line for observed value

Why:

- makes hypothesis testing intuitive.

---

## 3) Recommended tooling (beyond Matplotlib)

### Notebook-first stack

- `plotly.express` / `plotly.graph_objects`
- `seaborn` for quick static checks
- `ipywidgets` for interactive controls in notebooks

### Browser dashboard stack

- `streamlit` for fastest delivery
- `dash` for more controlled, app-like interactions
- optional `panel`/`bokeh` if preferred

### Animation options

- Plotly animation frames for:
  - metric evolution by layer
  - pair switching over time-like axis
- Matplotlib `FuncAnimation` only for export GIF/MP4 if needed

---

## 4) Interactive UX ideas

### A) Linked controls

Controls:

- metric selector
- modality pair selector
- layer range slider
- significance threshold slider
- null summary toggle

Expected behavior:

- all plots update together from the same filters.

### B) Linked brushing / cross-filtering

If user clicks one heatmap cell `(pair, layer)`:

- update distribution panel (null vs observed)
- update line charts with highlighted point
- update table with exact values

### C) Scene subset filters

If runs are stratified by scene subsets/splits:

- add split selector
- compare curves side by side

---

## 5) Suggested data model for visualization

Prepare one tidy dataframe per run, then cache:

Main table columns:

- `run_id`
- `metric`
- `pair`
- `layer`
- `value`
- `p_value`
- `null_mean`
- `null_std`
- `delta_vs_null_mean`
- optional `z_score`

Optional null draws table (for distribution panels):

- `run_id`
- `metric`
- `pair`
- `layer`
- `draw_id`
- `null_value`

This makes notebook and dashboard code much cleaner.

---

## 6) Visual language recommendations

- Keep one consistent color palette per modality pair.
- Keep one consistent line style per metric.
- Show significance in a separate channel (marker fill, opacity, or annotation), not by changing base line color too much.
- Always label axes with metric names and layer index clearly.
- Include tooltips with exact numeric values in interactive views.

---

## 7) Concrete implementation options

### Option 1: Notebook report (fastest)

Create:

- `notebooks/rq1_visual_report.ipynb`

Sections:

1. load metrics and null artifacts
2. build tidy tables
3. plot core figure set
4. interactive controls with `ipywidgets`

Best for:

- quick research iteration.

### Option 2: Streamlit dashboard (best handoff UX)

Create:

- `apps/rq1_dashboard.py` (or `scripts/rq1_dashboard.py`)

Pages:

- overview (curves + heatmap)
- significance explorer
- null distribution explorer
- run comparison

Best for:

- sharing results with team members quickly in browser.

### Option 3: Hybrid

- notebook for exploration
- streamlit for communication

This is often the most practical path.

---

## 8) Animation ideas (optional but useful)

### A) Layer sweep animation

- animate layer from 0 to last
- show pair-wise metric values evolving

### B) Pair sweep animation

- fix metric and sweep through modality pairs
- highlight changing significance map

### C) Null stabilization animation

- show how null histogram converges as draws increase

These animations are excellent for explaining methods in presentations.

---

## 9) Priority roadmap for next developer

1. Build a clean tidy dataframe export from benchmark outputs.
2. Implement static core figure set first.
3. Add interactive notebook controls.
4. Add browser dashboard (Streamlit recommended).
5. Add one or two animations for communication.

---

## 10) Definition of done for RQ1 visualization

Visualization is considered complete when:

- all three metrics can be explored by pair and layer
- significance is visible and interpretable
- null vs observed can be inspected interactively
- outputs can be viewed both in notebook and browser
- figures are presentation-ready and reproducible

