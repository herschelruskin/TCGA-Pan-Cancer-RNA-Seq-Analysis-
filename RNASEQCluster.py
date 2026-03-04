#!/usr/bin/env python3
"""
Clustering on UCI 'gene expression cancer RNA-Seq' dataset (id=401).

Dataset: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
"""

import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix  
import itertools 


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


OUTPUT_DIR = "outputs"

N_TOP_GENES = 3000

N_PCA_COMPONENTS = 50

K_RANGE = range(2, 11)

DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 10


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def select_top_variance_features(X: pd.DataFrame, n_top: int) -> pd.DataFrame:
    """
    Keep the n_top numeric columns with highest variance.
    """
    X_num = X.select_dtypes(include=[np.number])

    if n_top >= X_num.shape[1]:
        return X_num

    variances = X_num.var(axis=0)
    top_cols = variances.sort_values(ascending=False).head(n_top).index
    return X_num[top_cols]


def preprocess_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    1. log1p transform to compress large RNA-Seq values.
    2. StandardScaler to zero mean, unit variance per feature.
    """
    X_arr = X.to_numpy(dtype=float)
    X_log = np.log1p(X_arr)  # log(1 + x), safe for zeros

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled, scaler


def apply_pca(X_scaled: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, algo_name: str) -> dict:
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        print(f"[WARN] {algo_name}: only {len(unique_labels)} cluster(s); metrics undefined.")
        return {
            "n_clusters": len(unique_labels),
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return {
        "n_clusters": len(unique_labels),
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }


def run_kmeans_suite(X: np.ndarray, k_range) -> pd.DataFrame:
    rows = []
    for k in k_range:
        print(f"[INFO] KMeans k={k}")
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        m = evaluate_clustering(X, labels, f"KMeans(k={k})")
        m["k"] = k
        rows.append(m)
    df = pd.DataFrame(rows).set_index("k")
    return df


def run_agglo_suite(X: np.ndarray, k_range) -> pd.DataFrame:
    rows = []
    for k in k_range:
        print(f"[INFO] Agglomerative k={k}")
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ac.fit_predict(X)
        m = evaluate_clustering(X, labels, f"Agglomerative(k={k})")
        m["k"] = k
        rows.append(m)
    df = pd.DataFrame(rows).set_index("k")
    return df


def run_dbscan_once(X: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, dict]:
    print(f"[INFO] DBSCAN eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    m = evaluate_clustering(X, labels, "DBSCAN")
    m["eps"] = eps
    m["min_samples"] = min_samples
    return labels, m

def plot_dbscan_sweep(df: pd.DataFrame, output_dir: str):
    # Plot: eps vs number of clusters
    plt.figure()
    plt.plot(df["eps"], df["n_clusters"], marker="o")
    plt.xlabel("eps")
    plt.ylabel("Number of clusters")
    plt.title("DBSCAN Sweep: eps vs n_clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dbscan_sweep_n_clusters.png"), dpi=200)
    plt.close()

    # Plot: eps vs silhouette
    plt.figure()
    plt.plot(df["eps"], df["silhouette"], marker="o")
    plt.xlabel("eps")
    plt.ylabel("Silhouette score")
    plt.title("DBSCAN Sweep: eps vs silhouette")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dbscan_sweep_silhouette.png"), dpi=200)
    plt.close()


def plot_metric_vs_k(df: pd.DataFrame, algo_name: str, metric: str, output_dir: str) -> None:
    plt.figure()
    df[metric].plot(marker="o")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.title(f"{algo_name}: {metric} vs k")
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{algo_name}_{metric}_vs_k.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[INFO] Saved {fname}")


def plot_pca_scatter(X_pca: np.ndarray, labels: np.ndarray, title: str,
                     output_dir: str, prefix: str) -> None:
    if X_pca.shape[1] < 2:
        print("[WARN] PCA has <2 components; skipping scatter.")
        return

    plt.figure()
    uniq = np.unique(labels)
    for lab in uniq:
        mask = labels == lab
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.7, label=str(lab))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{prefix}_pca_scatter.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[INFO] Saved {fname}")

def dbscan_sweep(X: np.ndarray, eps_values, min_samples):
    """
    Run DBSCAN for a range of eps values.
    Returns a DataFrame with:
        - eps
        - n_clusters
        - silhouette (if valid)
    """
    results = []

    for eps in eps_values:
        print(f"[SWEEP] DBSCAN eps={eps}, min_samples={min_samples}")
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        unique = np.unique(labels)

        # Default metrics
        sil = np.nan

        # Silhouette only valid if >= 2 clusters
        if len(unique) > 1:
            try:
                sil = silhouette_score(X, labels)
            except:
                sil = np.nan

        results.append({
            "eps": eps,
            "n_clusters": len(unique),
            "silhouette": sil
        })

    return pd.DataFrame(results)

def plot_pca_scatter_true_labels(X_pca: np.ndarray, y_true: pd.Series,
                                 output_dir: str, prefix: str = "true_labels"):
    """
    2D PCA scatter colored by true labels (tumor types).
    """
    if X_pca.shape[1] < 2:
        print("[WARN] PCA has <2 components; skipping true-label scatter.")
        return

    plt.figure()
    labels = y_true.astype(str).values
    unique = np.unique(labels)
    for lab in unique:
        mask = labels == lab
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.7, label=str(lab))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (PC1 vs PC2) colored by true cancer labels")
    plt.legend(title="Cancer type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{prefix}_pca_scatter.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[INFO] Saved {fname}")

def plot_confusion_heatmap(y_true, y_pred, output_dir: str,
                           title: str = "True label vs cluster (KMeans)",
                           fname: str = "kmeans_confusion_matrix.png"):
    """
    Plot a confusion-matrix-like heatmap of true labels vs cluster IDs.
    Uses raw counts.
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels_true)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im)
    plt.xlabel("Cluster ID")
    plt.ylabel("True label")
    plt.title(title)

    plt.yticks(ticks=range(len(labels_true)), labels=labels_true)
    plt.xticks(ticks=range(len(labels_pred)), labels=labels_pred)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")

def plot_model_comparison(best_kmeans_sil, best_agglo_sil,
                          ari_kmeans, ari_agglo,
                          output_dir: str,
                          fname: str = "model_comparison.png"):
    """
    Simple bar chart comparing silhouette and ARI for
    best KMeans and best Agglomerative solutions.
    """
    models = ["KMeans", "Agglomerative"]

    sil_values = [best_kmeans_sil, best_agglo_sil]
    ari_values = [ari_kmeans, ari_agglo]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, sil_values, width, label="Silhouette")
    plt.bar(x + width/2, ari_values, width, label="ARI")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Comparison of best clustering solutions")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def main():
    ensure_output_dir(OUTPUT_DIR)
    print("[INFO] Loading local CSV files (data.csv, labels.csv)...")

    DATA_CSV = r"C:\Users\hersc\Downloads\gene+expression+cancer+rna+seq\TCGA-PANCAN-HiSeq-801x20531\TCGA-PANCAN-HiSeq-801x20531\data.csv"
    LABELS_CSV = r"C:\Users\hersc\Downloads\gene+expression+cancer+rna+seq\TCGA-PANCAN-HiSeq-801x20531\TCGA-PANCAN-HiSeq-801x20531\labels.csv"

    X_raw = pd.read_csv(DATA_CSV)
    print(f"[INFO] Raw data shape: {X_raw.shape}")

    sample_ids = X_raw.iloc[:, 0].values

    X = X_raw.iloc[:, 1:]
    print(f"[INFO] Feature matrix shape (numeric genes only): {X.shape}")

    labels_df = pd.read_csv(LABELS_CSV)
    print(f"[INFO] Raw labels shape: {labels_df.shape}")

    y_series = labels_df.iloc[:, 1]   

    print(f"[INFO] Selecting top {N_TOP_GENES} high-variance genes...")
    X_top = select_top_variance_features(X, N_TOP_GENES)
    print(f"[INFO] Shape after gene filtering: {X_top.shape}")

    print("[INFO] Preprocessing features (log1p + StandardScaler)...")
    X_scaled, scaler = preprocess_features(X_top)
    print(f"[INFO] Scaled shape: {X_scaled.shape}")

    print(f"[INFO] Applying PCA with {N_PCA_COMPONENTS} components...")
    X_pca, pca = apply_pca(X_scaled, N_PCA_COMPONENTS)
    print(f"[INFO] PCA-reduced shape: {X_pca.shape}")
    print("[INFO] First 10 explained variance ratios:")
    print(pca.explained_variance_ratio_[:10])

    if y_series is not None:
        plot_pca_scatter_true_labels(X_pca, y_series, OUTPUT_DIR)

    kmeans_metrics = run_kmeans_suite(X_pca, K_RANGE)
    kmeans_metrics.to_csv(os.path.join(OUTPUT_DIR, "kmeans_metrics.csv"))
    print("[INFO] Saved kmeans_metrics.csv")

    for metric in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
        plot_metric_vs_k(kmeans_metrics, "KMeans", metric, OUTPUT_DIR)

    best_k_kmeans = kmeans_metrics["silhouette"].idxmax()
    print(f"[INFO] Best KMeans k (by silhouette): {best_k_kmeans}")
    best_km = KMeans(n_clusters=best_k_kmeans, random_state=42, n_init="auto")
    km_labels = best_km.fit_predict(X_pca)
    plot_pca_scatter(X_pca, km_labels, f"KMeans (k={best_k_kmeans})", OUTPUT_DIR, "kmeans_best")
    if y_series is not None:
        plot_confusion_heatmap(y_series, km_labels, OUTPUT_DIR)


    agglo_metrics = run_agglo_suite(X_pca, K_RANGE)
    agglo_metrics.to_csv(os.path.join(OUTPUT_DIR, "agglo_metrics.csv"))
    print("[INFO] Saved agglo_metrics.csv")

    for metric in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
        plot_metric_vs_k(agglo_metrics, "Agglomerative", metric, OUTPUT_DIR)

    best_k_agglo = agglo_metrics["silhouette"].idxmax()
    print(f"[INFO] Best Agglomerative k (by silhouette): {best_k_agglo}")
    best_ag = AgglomerativeClustering(n_clusters=best_k_agglo, linkage="ward")
    ag_labels = best_ag.fit_predict(X_pca)
    plot_pca_scatter(X_pca, ag_labels, f"Agglomerative (k={best_k_agglo})", OUTPUT_DIR, "agglo_best")

    db_labels, db_metrics = run_dbscan_once(X_pca, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    db_metrics_df = pd.DataFrame([db_metrics])
    db_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_metrics.csv"), index=False)
    print("[INFO] Saved dbscan_metrics.csv")
    plot_pca_scatter(X_pca, db_labels, "DBSCAN", OUTPUT_DIR, "dbscan")

    eps_values = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10]
    print("[INFO] Running DBSCAN sweep...")
    sweep_df = dbscan_sweep(X_pca, eps_values, DBSCAN_MIN_SAMPLES)
    sweep_df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_sweep_results.csv"), index=False)
    plot_dbscan_sweep(sweep_df, OUTPUT_DIR)
    print("[INFO] Saved DBSCAN sweep plots + results.")

    result_df = pd.DataFrame()
    result_df["sample_id"] = sample_ids
    result_df["kmeans_k"] = best_k_kmeans
    result_df["kmeans_cluster"] = km_labels
    result_df["agglo_k"] = best_k_agglo
    result_df["agglo_cluster"] = ag_labels
    result_df["dbscan_eps"] = DBSCAN_EPS
    result_df["dbscan_min_samples"] = DBSCAN_MIN_SAMPLES
    result_df["dbscan_label"] = db_labels


    if y_series is not None:
        result_df["true_label"] = y_series.values
        ari_kmeans = adjusted_rand_score(y_series, km_labels)
        ari_agglo = adjusted_rand_score(y_series, ag_labels)
        print(f"[INFO] Adjusted Rand Index (KMeans best k): {ari_kmeans:.3f}")
        print(f"[INFO] Adjusted Rand Index (Agglomerative best k): {ari_agglo:.3f}")
    else:
        ari_kmeans = np.nan
        ari_agglo = np.nan

    out_csv = os.path.join(OUTPUT_DIR, "cluster_assignments.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved cluster_assignments.csv")

    print("\n[SUMMARY]")
    print("KMeans metrics (best k by silhouette):")
    print(kmeans_metrics.loc[best_k_kmeans])
    print("\nAgglomerative metrics (best k by silhouette):")
    print(agglo_metrics.loc[best_k_agglo])
    print("\nDBSCAN metrics:")
    print(db_metrics_df.iloc[0])
    if y_series is not None:
        print(f"\nAdjusted Rand Index wrt tumor labels:")
        print(f"KMeans (k={best_k_kmeans}): {ari_kmeans:.3f}")
        print(f"Agglomerative (k={best_k_agglo}): {ari_agglo:.3f}")
    if y_series is not None:
        best_kmeans_sil = kmeans_metrics.loc[best_k_kmeans, "silhouette"]
        best_agglo_sil = agglo_metrics.loc[best_k_agglo, "silhouette"]
        plot_model_comparison(best_kmeans_sil, best_agglo_sil,
                              ari_kmeans, ari_agglo,
                              OUTPUT_DIR)



if __name__ == "__main__":
    main()
