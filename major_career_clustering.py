import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings("ignore")

# Create output directories if they don't exist
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif"
})

URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/"
    "master/college-majors/recent-grads.csv"
)

print("Loading data from FiveThirtyEight GitHub...")
try:
    df = pd.read_csv(URL)
    print(f"Loaded from URL: {df.shape[0]} majors, {df.shape[1]} columns")
except Exception:
    # Fallback to local sample (included in repo as data/recent_grads_sample.csv)
    print("Network unavailable — loading local sample data (data/recent_grads_sample.csv)")
    df = pd.read_csv("data/recent_grads_sample.csv")
    print(f"Loaded local sample: {df.shape[0]} majors")
print(df.head())

# inspect data quality before modeling
print("\n--- Missing Values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Drop rows with missing values in our features of interest
df = df.dropna(subset=[
    "Median", "Unemployment_rate", "Full_time_year_round",
    "College_jobs", "Non_college_jobs", "Low_wage_jobs", "Total"
])
print(f"\nRows after dropping nulls: {len(df)}")

# feature engineering 

df["college_job_rate"] = df["College_jobs"] / df["Total"]
df["low_wage_rate"]    = df["Low_wage_jobs"] / df["Total"]
df["fulltime_rate"]    = df["Full_time_year_round"] / df["Total"]

FEATURES = [
    "Median",            # central tendency of earnings
    "Unemployment_rate", # labor market failure signal
    "fulltime_rate",     # stability of employment
    "college_job_rate",  # career ALIGNMENT (core variable)
    "low_wage_rate"      # career MISALIGNMENT (core variable)
]

X_raw = df[FEATURES].copy()
print(f"\nFeature matrix shape: {X_raw.shape}")
print(X_raw.describe().round(3))

# standardize features for K-means

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Elbow method + silhouette analysis to select k
print("\nRunning elbow + silhouette analysis for k = 2..10 ...")

inertias   = []
sil_scores = []
k_range    = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow + silhouette side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(k_range), inertias, "o-", color="#2563EB", linewidth=2)
ax1.axvline(x=4, color="#EF4444", linestyle="--", alpha=0.7, label="k=4 selected")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.set_title("Elbow Method — Selecting k")
ax1.legend()

ax2.plot(list(k_range), sil_scores, "s-", color="#10B981", linewidth=2)
ax2.axvline(x=4, color="#EF4444", linestyle="--", alpha=0.7, label="k=4 selected")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score — Cluster Cohesion")
ax2.legend()

plt.tight_layout()
plt.savefig("figures/fig1_k_selection.png", bbox_inches="tight")
plt.show()
print("Saved: figures/fig1_k_selection.png")

# final model with k=4 based on elbow + silhouette analysis
K = 4
km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

print(f"\nFinal model: KMeans(k={K})")
print(f"Inertia:          {km_final.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, df['cluster']):.4f}")
print(f"\nCluster sizes:\n{df['cluster'].value_counts().sort_index()}")

# visualize clusters using PCA projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

explained = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

CLUSTER_COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]
CLUSTER_LABELS = {
    0: "Cluster 0: Credentialed Pipelines",
    1: "Cluster 1: High-Earning STEM",
    2: "Cluster 2: Vocational / Applied",
    3: "Cluster 3: Misaligned / Precarious"
}

fig, ax = plt.subplots(figsize=(11, 7))
for c in range(K):
    mask = df["cluster"] == c
    ax.scatter(
        df.loc[mask, "pca1"], df.loc[mask, "pca2"],
        c=CLUSTER_COLORS[c], label=CLUSTER_LABELS[c],
        alpha=0.8, s=60, edgecolors="white", linewidths=0.4
    )

# Annotate a few notable majors
ANNOTATE = [
    "Computer Science", "Nursing", "Fine Arts",
    "Psychology", "Biology", "Accounting",
    "Philosophy", "Electrical Engineering", "Social Work"
]
for _, row in df[df["Major"].isin(ANNOTATE)].iterrows():
    ax.annotate(
        row["Major"], (row["pca1"], row["pca2"]),
        fontsize=7, alpha=0.85,
        xytext=(5, 3), textcoords="offset points"
    )

ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance explained)")
ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance explained)")
ax.set_title(
    "K-Means Clusters of College Majors by Career Alignment\n"
    "(PCA projection of 5 standardized features)",
    fontsize=12, fontweight="bold"
)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig2_pca_clusters.png", bbox_inches="tight")
plt.show()
print("Saved: figures/fig2_pca_clusters.png")

# cluster characterization: feature means, top majors, silhouette analysis, category validation
print("\n" + "="*65)
print("CLUSTER CHARACTERIZATION")
print("="*65)

cluster_summary = df.groupby("cluster")[FEATURES + ["college_job_rate", "low_wage_rate"]].mean().round(3)
cluster_summary.index = [CLUSTER_LABELS[i].split(":")[1].strip() for i in cluster_summary.index]
print(cluster_summary.to_string())

# Print top 5 majors per cluster by college_job_rate
for c in range(K):
    subset = df[df["cluster"] == c].sort_values("college_job_rate", ascending=False)
    print(f"\n--- {CLUSTER_LABELS[c]} ---")
    print("Top 5 by college_job_rate:")
    for _, row in subset.head(5).iterrows():
        print(f"  {row['Major']:<40} college_job_rate={row['college_job_rate']:.2f}  "
              f"Median=${row['Median']:,.0f}  Unemp={row['Unemployment_rate']:.1%}")
    print("Bottom 3 by college_job_rate (misaligned):")
    for _, row in subset.tail(3).iterrows():
        print(f"  {row['Major']:<40} college_job_rate={row['college_job_rate']:.2f}  "
              f"low_wage_rate={row['low_wage_rate']:.2f}")

# cluster profile heatmap (z-scored means)
fig, ax = plt.subplots(figsize=(10, 4))
heatmap_data = df.groupby("cluster")[FEATURES].mean()
heatmap_scaled = pd.DataFrame(
    StandardScaler().fit_transform(heatmap_data),
    columns=FEATURES,
    index=[f"Cluster {i}" for i in range(K)]
)
sns.heatmap(
    heatmap_scaled, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, linewidths=0.5, ax=ax
)
ax.set_title("Cluster Feature Profiles (Z-scored means)", fontweight="bold")
ax.set_xticklabels(
    ["Median\nSalary", "Unemployment\nRate", "Full-Time\nRate",
     "College\nJob Rate", "Low-Wage\nRate"],
    rotation=0, fontsize=9
)
plt.tight_layout()
plt.savefig("figures/fig3_cluster_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: figures/fig3_cluster_heatmap.png")

# silhouette analysis for final model
fig, ax = plt.subplots(figsize=(8, 5))
sil_vals = silhouette_samples(X_scaled, df["cluster"])
y_lower = 10
for c in range(K):
    c_sil = np.sort(sil_vals[df["cluster"] == c])
    size_c = len(c_sil)
    y_upper = y_lower + size_c
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                     facecolor=CLUSTER_COLORS[c], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_c, f"C{c}", fontsize=9)
    y_lower = y_upper + 10

avg_sil = silhouette_score(X_scaled, df["cluster"])
ax.axvline(x=avg_sil, color="red", linestyle="--", label=f"Avg = {avg_sil:.3f}")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title("Silhouette Analysis for K-Means (k=4)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig4_silhouette.png", bbox_inches="tight")
plt.show()
print("Saved: figures/fig4_silhouette.png")

# category validation: compare cluster assignments to Major_category distribution

crosstab = pd.crosstab(df["Major_category"], df["cluster"])
crosstab.columns = [f"C{i}" for i in range(K)]

fig, ax = plt.subplots(figsize=(10, 7))
crosstab.plot(kind="bar", stacked=True, ax=ax,
              color=CLUSTER_COLORS, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Major Category")
ax.set_ylabel("Number of Majors")
ax.set_title("Cluster Distribution by Major Category\n(Validation Check)",
             fontweight="bold")
ax.legend([CLUSTER_LABELS[i].split(":")[1].strip() for i in range(K)],
          loc="upper right", fontsize=8)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig5_category_validation.png", bbox_inches="tight")
plt.show()
print("Saved: figures/fig5_category_validation.png")

# ── 13. SUMMARY TABLE ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL CLUSTER SUMMARY TABLE")
print("="*65)
summary = df.groupby("cluster").agg(
    n_majors=("Major", "count"),
    avg_median_salary=("Median", "mean"),
    avg_unemployment=("Unemployment_rate", "mean"),
    avg_college_job_rate=("college_job_rate", "mean"),
    avg_low_wage_rate=("low_wage_rate", "mean")
).round(3)
summary.index = [CLUSTER_LABELS[i] for i in summary.index]
print(summary.to_string())
summary.to_csv("cluster_summary.csv")
print("\nSaved: cluster_summary.csv")

print("\n✓ Analysis complete. All figures saved to /figures/")
