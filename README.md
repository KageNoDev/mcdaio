# mcdaio

A small Python toolkit and Streamlit GUI for **Multi-Criteria Decision Analysis (MCDA)**
with built-in sensitivity analysis and interactive visualizations.

Supported methods:

- **MABAC** - Multi-Attributive Border Approximation area Comparison
- **TOPSIS** - Technique for Order of Preference by Similarity to Ideal Solution
- **VIKOR** - VIseKriterijumska Optimizacija I Kompromisno Resenje
- **ARAS** - Additive Ratio Assessment

The GUI is built with Streamlit, so it runs in any modern browser with a clean,
responsive interface.

Additional features:

- interactive Plotly charts of the results,
- leave-one-out sensitivity analysis on the criteria,
- a side-by-side comparison of all methods,
- light and dark mode,
- CSV export of the rankings.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/KageNoDev/mcdaio.git
cd mcdaio
pip install -e .
```

Requires **Python 3.9 or later**. All runtime dependencies (numpy, pandas,
scipy, plotly, streamlit) are pulled in automatically.

For development extras (tests, docs):

```bash
pip install -e ".[dev]"
```

---

## Usage

After installation, launch the GUI with the installed console script:

```bash
mcdaio
```

or directly via Streamlit:

```bash
streamlit run mcdaio/app.py
```

The application opens in your browser at:

```
http://localhost:8501
```

---

## Input format

- **Decision matrix:** a CSV/TXT file of numeric values, one alternative per row,
  one criterion per column. You can also paste it directly into the GUI.
- **Weights:** comma-separated, e.g. `0.2,0.3,0.5`. If empty, equal weights are used.
- **Criterion types:** comma-separated binary values, e.g. `1,0,1`
  - `1` → benefit (higher is better)
  - `0` → cost (lower is better)

Example matrix:

```
25000,7.5,3
30000,8.0,4
27000,7.0,5
```

with weights `0.4,0.35,0.25` and types `0,1,1`.

---

## Results

The application returns:

- a results table for the selected MCDA method,
- a ranking of the alternatives,
- a bar chart of the scores,
- (optional) a ranking-stability chart from the sensitivity analysis,
- (optional) a Spearman comparison against the other methods.

You can download the results as a **CSV file** directly from the GUI.

---

## Using mcdaio as a library

```python
import numpy as np
from mcdaio import MABAC, sensitivity_analysis, spearman_corr

matrix = np.array([
    [25000, 7.5, 3],
    [30000, 8.0, 4],
    [27000, 7.0, 5],
])
weights = np.array([0.4, 0.35, 0.25])
types = np.array([0, 1, 1])  # 0 = cost, 1 = benefit

scores, ranks = MABAC(matrix, weights, types).run()

rankings, corrs = sensitivity_analysis(ranks, matrix, weights, types, MABAC)
```

---

## Project layout

```
mcdaio/
├── mcdaio/                # Library package
│   ├── __init__.py        # Public API
│   ├── methods.py         # MABAC, TOPSIS, VIKOR, ARAS
│   ├── normalization.py   # Min-max normalization helper
│   ├── metrics.py         # Spearman + weighted rank correlation
│   ├── analysis.py        # Sensitivity analysis
│   ├── app.py             # Streamlit GUI
│   └── cli.py             # `mcdaio` console-script entry point
├── tests/                 # pytest suite
├── docs/                  # Sphinx documentation
├── pyproject.toml         # Build / packaging configuration
└── README.md
```

---

## Tests

```bash
pytest
```

## Documentation

```bash
cd docs
make html
```

The built HTML lives in `docs/build/html/index.html`.

---

## License

MIT License.

---

## Author

Created by **Laura Białobrzewska** - feel free to contribute or fork.
