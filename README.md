# NS5007 — Week 5 lab: Dimensionality reduction (PCA & MDS)

Python lab for **NS5007 Human and Artificial Intelligence** (MSc Neuroscience).
Run [`week5_pca_mds.ipynb`](week5_pca_mds.ipynb) directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qihongl/NS5007-week5/main/week5_pca_mds.ipynb)

No setup needed — part II of the lab fetches its packaged fMRI data (~15 MB,
`data/`) from this repo automatically.

## Contents

| file | what it is |
|---|---|
| `week5_pca_mds.ipynb` | the lab: PCA & classical MDS from scratch → Kriegeskorte-style IT RDMs → Schapiro-style community clustering in real fMRI |
| `data/week5_schapiro_patterns.npz` | item × voxel fMRI patterns, 20 subjects × 4 regions (left IFG/anterior insula, left occipito-temporal, hippocampus, early visual control) |
| `data/week5_schapiro_tmap.npz` + `_clusters.json` | group whole-brain searchlight t-map (lag-matched statistic) + top cluster masks |
| `data/week5_schapiro_shifts.npz` | shift-predictor null distributions per subject/region |
| `make_week5_derived_data.py` | exact offline pipeline that generates the packaged data from raw OpenNeuro ds001621 BOLD |
| `REPRODUCTION_NOTES.md` | what reproduced, what didn't, and why (preprocessing sensitivity, lag-matching control) |

## Data provenance

All fMRI data derive from [OpenNeuro ds001621](https://openneuro.org/datasets/ds001621)
(Schapiro, Rogers, Cordova, Turk-Browne & Botvinick, 2013, *Nat Neurosci*,
Experiment 3 — temporal community structure). Offline preprocessing per run:
despiking, rigid-body motion correction, cosine drift removal (0.01 Hz), affine
EPI→MNI normalization, then per-item pattern extraction following the paper's
Methods (z-scored timecourses sampled +2 TR after onset, Hamiltonian-path
trials ≥4 steps into a block, 3×3×3 searchlights, Fisher-z within-minus-between
statistic with path-distance matching). See `REPRODUCTION_NOTES.md`.

If you use the data in your own work, cite the original paper and dataset.
