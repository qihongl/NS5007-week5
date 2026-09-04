# NS5007 — lab notebooks

Python labs for **NS5007 Human and Artificial Intelligence** (MSc Neuroscience).

All lab notebooks — open directly in Colab:

| notebook | open in Colab |
|---|---|
| [`pca_mds.ipynb`](pca_mds.ipynb) — PCA & MDS on neural data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qihongl/NS5007/blob/main/pca_mds.ipynb) |
| [`basic_nn.ipynb`](basic_nn.ipynb) — neural networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qihongl/NS5007/blob/main/basic_nn.ipynb) |
| [`ensemble_methods_lab.ipynb`](ensemble_methods_lab.ipynb) — ensembles & random forests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qihongl/NS5007/blob/main/ensemble_methods_lab.ipynb) |

No setup needed — the fMRI lab fetches its packaged data (~15 MB, `data/`) from this
repo automatically, and the Kriegeskorte 92-image supplement is cached in
`data/kriegeskorte92/`.

## Contents

| file | what it is |
|---|---|
| `pca_mds.ipynb` | the lab: PCA & classical MDS from scratch → Kriegeskorte-style IT RDMs → Schapiro-style community clustering in real fMRI |
| `basic_nn.ipynb` | neural networks: one neuron → depth on the moons → digit classification |
| `ensemble_methods_lab.ipynb` | ensembles: majority votes, Condorcet/bagging simulations, random forests, label-noise stress test |
| `data/kriegeskorte92/` | 92-image RDMs (human + monkey IT, model RDMs), © Kriegeskorte et al. 2008, mirrored from the [rsatoolbox](https://github.com/rsagroup/rsatoolbox) repo |
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
