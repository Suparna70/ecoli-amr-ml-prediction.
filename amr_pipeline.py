"""
Predicting antibiotic resistance in Escherichia coli from gene presence/absence data
====================================================================================
End-to-end pipeline (single file):
  1. Load public dataset (Moradigaravand et al., 2018 derivative CSVs)
  2. Build feature matrix + labels for each antibiotic
  3. Feature selection (|P_R - P_S| score, then chi-square + odds ratio filter)
  4. Train/test split (70:30, stratified) + 3 models with 5-fold CV
  5. Evaluate on held-out test set (AUC, sensitivity, specificity, accuracy, F1)
  6. Save results tables and publication-style figures

Run (Colab):   !python amr_pipeline.py
Run (local):   python amr_pipeline.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURATION  (edit this block only)
# ----------------------------------------------------------------------------
BASE = ("https://raw.githubusercontent.com/Lucy-Moctezuma/"
        "ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli/main/Datasets/")
METADATA_URL = BASE + "Metadata.csv"
GENES_URL = BASE + "AccessoryGene.csv"

ID_COL = "Isolate"                                   # isolate ID column in metadata
ANTIBIOTICS = {"CIP": "Ciprofloxacin",               # column name -> display name
               "AMP": "Ampicillin",
               "GEN": "Gentamicin"}
TOP_N_GENES = 100          # genes kept after the |P_R - P_S| ranking
P_THRESHOLD = 0.05         # chi-square significance cutoff
OR_THRESHOLD = 2.0         # odds-ratio cutoff
TEST_SIZE = 0.30           # 70:30 split
CV_FOLDS = 5
RANDOM_STATE = 42

# True  = feature selection uses TRAINING data only (no data leakage; recommended)
# False = selection on all data before the split (reproduces the first poster numbers)
SELECT_ON_TRAIN_ONLY = True

OUT_DIR = "results"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# PLOT STYLE (journal-like, colour-blind-safe Okabe-Ito palette)
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight",
})
C_SUS, C_RES = "#0072B2", "#D55E00"
MODEL_COLORS = {"Logistic Regression": "#0072B2",
                "Random Forest": "#009E73",
                "Gradient Boosting": "#E69F00"}


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_data():
    metadata = pd.read_csv(METADATA_URL)
    genes = pd.read_csv(GENES_URL)
    print(f"Metadata: {metadata.shape} | Gene matrix: {genes.shape}")
    return metadata, genes


def build_xy(metadata, genes, antibiotic):
    """Merge gene matrix with one antibiotic's labels (R=1, S=0, missing dropped)."""
    gene_id = genes.columns[0]
    merged = metadata[[ID_COL, antibiotic]].merge(
        genes, left_on=ID_COL, right_on=gene_id)
    merged = merged.dropna(subset=[antibiotic])
    y = merged[antibiotic].map({"R": 1, "S": 0})
    X = merged.drop(columns=[ID_COL, antibiotic, gene_id], errors="ignore")
    keep = y.notna()
    return X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. FEATURE SELECTION
# ----------------------------------------------------------------------------
def gene_stats(gene, X, y):
    """Chi-square p-value and odds ratio for one gene vs. resistance."""
    table = pd.crosstab(X[gene], y)
    if table.shape != (2, 2):
        return np.nan, np.nan
    _, p, _, _ = chi2_contingency(table)
    a, b = table.loc[1, 1], table.loc[1, 0]
    c, d = table.loc[0, 1], table.loc[0, 0]
    odds_ratio = (a * d) / (b * c) if (b * c) != 0 else np.inf
    return p, odds_ratio


def select_features(X, y):
    """Rank by |P_R - P_S|, keep top N, then keep genes with p<0.05 and OR>2."""
    score = (X[y == 1].mean() - X[y == 0].mean()).abs()
    top = score.sort_values(ascending=False).head(TOP_N_GENES).index
    stats = pd.DataFrame([(g, *gene_stats(g, X, y)) for g in top],
                         columns=["gene", "p_value", "odds_ratio"])
    sig = stats[(stats.p_value < P_THRESHOLD) & (stats.odds_ratio > OR_THRESHOLD)]
    genes = sig["gene"].tolist() or list(top)      # fallback: never return empty
    return genes, stats


# ----------------------------------------------------------------------------
# 3. MODELS + EVALUATION
# ----------------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(model, X_te, y_te):
    """Test-set metrics at the Youden-index optimal threshold."""
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    fpr, tpr, thr = roc_curve(y_te, proba)
    best = thr[np.argmax(tpr - fpr)]
    pred = (proba >= best).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {"AUC": auc, "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
            "Accuracy": accuracy_score(y_te, pred), "F1": f1_score(y_te, pred)}


def run_antibiotic(metadata, genes, code):
    name = ANTIBIOTICS[code]
    print(f"\n=== {name} ({code}) ===")
    X, y = build_xy(metadata, genes, code)
    print(f"n={len(y)} | resistant={int(y.sum())} | susceptible={int((y == 0).sum())}")

    if SELECT_ON_TRAIN_ONLY:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        feats, stats = select_features(X_tr, y_tr)          # training data only
    else:
        feats, stats = select_features(X, y)                # (leaky) all data
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[feats], y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        X_tr, X_te = X_tr[feats], X_te[feats]
    X_tr, X_te = X_tr[feats], X_te[feats]
    print(f"Selected {len(feats)} significant genes (of top {TOP_N_GENES})")

    rows, fitted, cv_table = [], {}, []
    for mname, model in get_models().items():
        cv = cross_val_score(model, X_tr, y_tr, cv=CV_FOLDS, scoring="roc_auc")
        model.fit(X_tr, y_tr)
        fitted[mname] = model
        m = evaluate(model, X_te, y_te)
        rows.append({"Antibiotic": code, "Model": mname, "CV_AUC_mean": cv.mean(),
                     "CV_AUC_std": cv.std(), **m})
        print(f"{mname:20s} CV AUC {cv.mean():.4f} +/- {cv.std():.4f} | "
              f"Test AUC {m['AUC']:.4f}")

    rf_imp = pd.Series(fitted["Random Forest"].feature_importances_,
                       index=feats).sort_values(ascending=False)
    return {"code": code, "name": name, "y": y, "X_te": X_te, "y_te": y_te,
            "models": fitted, "results": pd.DataFrame(rows),
            "importance": rf_imp, "gene_stats": stats}


# ----------------------------------------------------------------------------
# 4. FIGURES
# ----------------------------------------------------------------------------
def save(fig, stem):
    fig.savefig(f"{FIG_DIR}/{stem}.png")
    fig.savefig(f"{FIG_DIR}/{stem}.pdf")
    plt.close(fig)


def fig_class_distribution(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(7.2, 2.6))
    for ax, r, lab in zip(axes, runs, "ABC"):
        sus, res = int((r["y"] == 0).sum()), int((r["y"] == 1).sum())
        bars = ax.bar(["Susceptible", "Resistant"], [sus, res],
                      color=[C_SUS, C_RES], width=0.6)
        for b, v in zip(bars, [sus, res]):
            ax.text(b.get_x() + b.get_width() / 2, v + max(sus, res) * 0.02,
                    str(v), ha="center", fontsize=8)
        ax.set_title(r["name"])
        ax.set_ylabel("Isolates (n)" if ax is axes[0] else "")
        ax.set_ylim(0, max(sus, res) * 1.18)
        ax.text(-0.28, 1.05, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_class_distribution")


def fig_model_comparison(all_results):
    codes = list(ANTIBIOTICS)
    models = list(MODEL_COLORS)
    x, w = np.arange(len(codes)), 0.25
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, m in enumerate(models):
        sub = all_results[all_results.Model == m].set_index("Antibiotic").loc[codes]
        ax.bar(x + i * w, sub.CV_AUC_mean, w, yerr=sub.CV_AUC_std, capsize=2.5,
               color=MODEL_COLORS[m], label=m, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(codes)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("5-fold CV AUC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_model_comparison")


def fig_roc(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(7.2, 2.7))
    for ax, r, lab in zip(axes, runs, "ABC"):
        for mname, model in r["models"].items():
            p = model.predict_proba(r["X_te"])[:, 1]
            fpr, tpr, _ = roc_curve(r["y_te"], p)
            ax.plot(fpr, tpr, color=MODEL_COLORS[mname], lw=1.3,
                    label=f"{mname.split()[0]} ({roc_auc_score(r['y_te'], p):.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
        ax.set_title(r["name"])
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate" if ax is axes[0] else "")
        ax.legend(loc="lower right", fontsize=6.5)
        ax.set_aspect("equal")
        ax.text(-0.32, 1.08, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_roc_curves")


def fig_metrics_heatmap(all_results):
    cols = ["AUC", "Sensitivity", "Specificity", "Accuracy", "F1"]
    df = all_results.set_index(["Antibiotic", "Model"])[cols]
    labels = [f"{a} - {''.join(w[0] for w in m.split())}" for a, m in df.index]
    fig, ax = plt.subplots(figsize=(3.8, 5.2))
    im = ax.imshow(df.values, cmap="viridis", vmin=0.75, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < 0.9 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Score")
    fig.tight_layout()
    save(fig, "fig_metrics_heatmap")


def fig_feature_importance(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(7.5, 3.0))
    for ax, r, lab in zip(axes, runs, "ABC"):
        imp = r["importance"].head(10)
        ypos = np.arange(len(imp))
        ax.barh(ypos, imp.values, color="#4C4C9C", height=0.65)
        ax.set_yticks(ypos)
        ax.set_yticklabels(imp.index, fontsize=7.5)
        for t in ax.get_yticklabels():                      # italic gene names
            if not t.get_text().startswith("group_"):
                t.set_fontstyle("italic")
        ax.invert_yaxis()
        ax.set_xlabel("Feature importance")
        ax.set_title(r["name"])
        ax.text(-0.55, 1.08, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_feature_importance")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    metadata, genes = load_data()
    runs = [run_antibiotic(metadata, genes, code) for code in ANTIBIOTICS]

    all_results = pd.concat([r["results"] for r in runs], ignore_index=True)
    all_results.round(4).to_csv(f"{OUT_DIR}/model_performance.csv", index=False)

    imp_rows = []
    for r in runs:
        top = r["importance"].head(15)
        imp_rows += [{"Antibiotic": r["code"], "Rank": i + 1, "Gene": g, "Importance": v}
                     for i, (g, v) in enumerate(top.items())]
    pd.DataFrame(imp_rows).round(5).to_csv(f"{OUT_DIR}/top_genes.csv", index=False)

    fig_class_distribution(runs)
    fig_model_comparison(all_results)
    fig_roc(runs)
    fig_metrics_heatmap(all_results)
    fig_feature_importance(runs)

    print("\n" + "=" * 70)
    print(all_results.round(3).to_string(index=False))
    print(f"\nSaved tables to ./{OUT_DIR}/ and figures to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
