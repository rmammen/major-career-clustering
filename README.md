# Major → Career: Clustering College Majors by Alignment

**INST414 - Data Science Techniques | Assignment 4**

## Research Question

> *To what extent do college majors align with their intended occupational fields, and can unsupervised clustering reveal latent archetypes of career alignment and misalignment across disciplines?*

**Stakeholder:** University academic advisors and career counselors  
**Method:** K-Means Clustering (k=4) with Euclidean distance on standardized features  
**Dataset:** FiveThirtyEight College Majors (ACS 2010–2012, ~173 majors)

---

## Project Structure

```
major-career-clustering/
├── major_career_clustering.py   # Main analysis script
├── medium_post.md               # Full Medium post draft
├── data/
│   └── recent_grads_sample.csv  # Local fallback dataset (65 majors)
├── figures/
│   ├── fig1_k_selection.png     # Elbow + Silhouette plots
│   ├── fig2_pca_clusters.png    # PCA scatter of 4 clusters
│   ├── fig3_cluster_heatmap.png # Feature profile heatmap
│   ├── fig4_silhouette.png      # Silhouette analysis
│   └── fig5_category_validation.png  # Cluster vs. Major category
├── cluster_summary.csv          # Output: cluster means table
└── README.md
```

---

## Setup & Running

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/major-career-clustering.git
cd major-career-clustering
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the analysis
```bash
python major_career_clustering.py
```

The script will:
- Attempt to load live data from FiveThirtyEight GitHub
- Fall back to `data/recent_grads_sample.csv` if network is unavailable
- Generate all 5 figures in the `figures/` directory
- Print cluster characterizations and save `cluster_summary.csv`

---

## Data Source

**FiveThirtyEight College Majors Dataset**  
URL: `https://github.com/fivethirtyeight/data/tree/master/college-majors`  
File: `recent-grads.csv`  
Original source: U.S. Census Bureau, American Community Survey (ACS) 2010–2012 Public Use Microdata Series  
Compiled by: Ben Casselman for FiveThirtyEight

### Key Fields Used

| Field | Description |
|---|---|
| `Median` | Median annual earnings |
| `Unemployment_rate` | Share unemployed and seeking work |
| `Full_time_year_round` | Graduates in stable full-time work |
| `College_jobs` | Jobs requiring a college degree |
| `Low_wage_jobs` | Service/retail/precarious employment |
| `Total` | Total graduates in sample |
| `Major_category` | Broad disciplinary grouping |

### Engineered Features

```python
df["college_job_rate"] = df["College_jobs"] / df["Total"]  # alignment signal
df["low_wage_rate"]    = df["Low_wage_jobs"] / df["Total"] # misalignment signal
df["fulltime_rate"]    = df["Full_time_year_round"] / df["Total"]
```

---

## Methods

### Similarity Metric
**Euclidean distance** on 5 standardized features:
- `Median`, `Unemployment_rate`, `fulltime_rate`, `college_job_rate`, `low_wage_rate`

Standardization via `StandardScaler` (z-score) prevents `Median` salary from dominating distance calculations.

### K Selection
- **Elbow Method:** Inertia vs. k for k=2..10 → clear inflection at k=4
- **Silhouette Score:** Peaks/stabilizes around k=4 (~0.40)
- **Interpretability:** k=4 maps to four theoretically coherent career archetypes

### Algorithm
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)
```

---

## Cluster Summary

| Cluster | Label | n | Avg Median | Avg College Job Rate | Avg Low-Wage Rate |
|---|---|---|---|---|---|
| 0 | Credentialed Pipelines | 18 | $57,250 | 76% | 8.8% |
| 1 | Mixed Outcomes / Mid-Range | 34 | $40,235 | 51% | 14.3% |
| 2 | Elite Technical Fields | 5 | $76,000 | 78% | 14.2% |
| 3 | Misaligned / Precarious | 8 | $34,125 | 34% | 22.9% |

### Example Majors Per Cluster

**Cluster 0 — Credentialed Pipelines**
- Nursing, Chemical Engineering, Biomedical Engineering, Applied Mathematics, Accounting

**Cluster 1 — Mixed Outcomes**
- Psychology, Biology, Journalism, Economics, Communications, History

**Cluster 2 — Elite Technical**
- Petroleum Engineering, Nuclear Engineering, Naval Architecture & Marine Engineering

**Cluster 3 — Misaligned / Precarious**
- Fine Arts, Music, Visual & Performing Arts, Studio Arts, Environmental Science

---

## Validation

- **Silhouette analysis** confirms meaningful cluster separation (avg score ~0.40)
- **Cross-tabulation** with `Major_category` shows Engineering concentrating in Clusters 0 & 2, Arts in Cluster 3 — consistent with domain expectations
- Missing value check: no nulls in feature columns after loading
- Spot-checked 5 majors against original FiveThirtyEight article for data accuracy

---

## Limitations

- ACS data from 2010–2012; labor market conditions have shifted since
- Graduate school pathways are not captured (affects Biology, Psychology)
- Median earnings obscure within-major variance
- Race, gender, and socioeconomic factors are absent from the analysis
- K-Means assumes spherical clusters; agglomerative clustering may reveal finer structure
