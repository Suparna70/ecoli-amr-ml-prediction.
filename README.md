# Predicting Antibiotic Resistance in *E. coli* from Gene Presence/Absence Data

Machine-learning pipeline that predicts resistance of *Escherichia coli* to three antibiotics
(ciprofloxacin, ampicillin, gentamicin) using only publicly available data, and checks whether
the genes the models rely on match the known resistance biology.

**Key finding:** all models reach high test AUC (~0.92-0.96), but only for ampicillin (*bla*) and
gentamicin (*aac3-VI*) does the top feature correspond to the true causal resistance gene. For
ciprofloxacin (resistance driven by *gyrA*/*parC* point mutations, invisible to gene
presence/absence data) the model relies on plasmid-maintenance genes as an indirect proxy.
**Accuracy is not the same as mechanistic correctness.**

## Pipeline

1. Load public dataset (1,936 *E. coli* isolates, 17,198 accessory genes)
2. Build feature matrix and labels per antibiotic (R = 1, S = 0)
3. Feature selection: top 100 genes by |P_R - P_S|, then chi-square (p < 0.05) and odds ratio (OR > 2)
4. Stratified 70:30 train/test split; Logistic Regression, Random Forest, Gradient Boosting (5-fold CV)
5. Evaluate on held-out test set: AUC, sensitivity, specificity, accuracy, F1 (Youden-index threshold)
6. Save result tables and publication-style figures

Feature selection is done **inside the training set only** (no data leakage). Set
`SELECT_ON_TRAIN_ONLY = False` in the config block to reproduce the earlier all-data variant.

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/ecoli-amr-ml-prediction.git
cd ecoli-amr-ml-prediction
pip install -r requirements.txt
python amr_pipeline.py
```

Google Colab: upload `amr_pipeline.py`, then run `!pip install -r requirements.txt` and `!python amr_pipeline.py`.

## Outputs

| Folder | Contents |
|---|---|
| `results/model_performance.csv` | CV AUC and test metrics for every model x antibiotic |
| `results/top_genes.csv` | Top 15 Random Forest features per antibiotic |
| `figures/` | Class distribution, model comparison, ROC curves, metrics heatmap, feature importance (PNG + PDF) |

## Data and credit

Gene presence/absence matrix and susceptibility phenotypes derive from:
Moradigaravand D. et al. (2018) *Prediction of antibiotic resistance in Escherichia coli from
large-scale pan-genome data.* PLoS Comput Biol 14(12): e1006258. The CSV copies are loaded directly
from a public tutorial repository (see `METADATA_URL` / `GENES_URL` in the script). The framework
was inspired by Wan et al. (2026) *Front. Microbiol.* 17:1842717.

## Limitations

Single public cohort, no external validation, binary R/S labels only, and gene presence/absence
cannot capture point mutations. Planned next steps: add SNP features (*gyrA*/*parC*), more annotated
isolates, and independent validation.

## Author

Suparna Mondal - Department of Biotechnology, Chandigarh University

## License

MIT (add a `LICENSE` file - choose "MIT" in GitHub's *Add file > Create new file > LICENSE* template).
