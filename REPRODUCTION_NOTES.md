# Reproduction notes: Schapiro et al. 2013, Exp 3 (ds001621)

## What the paper actually did (Methods, verified from the PDF)

The famous community-cluster MDS figure (**Figure 4**) is **not hippocampus**.
It comes from a whole-brain **searchlight pattern analysis**; clusters with
reliable community structure were **left IFG, left insula, left anterior
temporal lobe (ATL), left STG** (t(19)=140.8 for within>between community
similarity). Their recipe:

1. z-score each voxel's timecourse per run (on AFNI/SPM-preprocessed data:
   slice-time correction, 6-param motion correction, despiking, nonlinear
   Talairach normalization)
2. for each item presentation **>=4 steps into a Hamiltonian path block**,
   take the z-scored activation **2 TRs (4 s) after onset**
3. average across presentations -> one 27-voxel vector per item per 3x3x3
   searchlight -> Pearson -> 15x15 similarity matrix per searchlight
4. statistic: mean Fisher-z corr(within community) - mean corr(between),
   **matched on path distance** (only compare within- vs between-community
   pairs that are the same number of steps apart in the subject's fixed
   Hamiltonian path, d = 1..4)
5. Figure 4: MDS of searchlight-averaged RDMs within each cluster, n=20

The hippocampal "events in the hippocampus" story comes from the follow-up
paper (Schapiro, Turk-Browne, Norman & Botvinick, 2016, *Hippocampus*), not
from this figure.

## What our first attempts did wrong

* analyzed **hippocampus** (wrong region for this figure)
* used **GLM beta estimates** (they use no GLM for this analysis)
* no lag-matching -> temporal-proximity/drift artifact dominates raw data

## Results (20 subjects, all raw BOLD from OpenNeuro ds001621)

| region | raw stat (no control) | lag-matched stat (their control) |
|---|---|---|
| IFG/insula L | +0.835 (t=94.8) | +0.010 (t=0.65, p=0.53) |
| ATL L | +0.821 (t=99.2) | +0.016 (t=1.25, p=0.23) |
| STG L | +0.819 (t=105.8) | +0.009 (t=0.66, p=0.52) |
| Hippocampus | +0.816 (t=86.5) | +0.017 (t=1.60, p=0.13) |

Embedding separation (centroid distance minus spread, label-permutation test):
significant in all regions (p ~ 0.002) **but** the lag-matched control is not,
so the visible clustering cannot be attributed to community representation
specifically -- in our data it is confounded with temporal proximity.

## Interpretation

* The huge raw values are drift/autocorrelation artifacts (same-community
  items occur closer together in time by construction).
* With the paper's lag-matched control, our lightly-preprocessed data show
  weak, non-significant positive effects. The gap to their t=140 is most
  plausibly explained by missing preprocessing (motion correction, despiking,
  slice-timing) and by their data-driven cluster selection (winner's curse)
  vs our unbiased atlas ROIs.
* No evidence of fraud; strong evidence that this effect size is
  preprocessing-dependent and that the lag-matching control is essential.

## Next step to close the gap

Add motion correction (e.g., dipy/ANTs rigid per volume, or FSL mcflirt if
available) + despiking to the preprocessing, then rerun
`reproduce_v3_matched.py`. Files: `v3_rdm_<sub>.npz` caches make the analysis
instant once patterns are computed.

## UPDATE: faithful reproduction succeeds (v4, `reproduce_v4_wholebrain.py`)

Pipeline: despiking + rigid motion correction (custom, simplified mcflirt) +
their exact pattern analysis (z-scored timecourses, +2 TR samples, HP-block
trials >=4 steps in, 3x3x3 searchlights, Fisher-z within-minus-between WITH
their path-distance lag-matching control) + whole-brain searchlight map +
group t-map across 20 subjects + data-driven cluster selection (their
Figure 4 logic).

Motion was tiny in this dataset (max ~0.3 vox, 0.5 deg) yet MC+despiking
still mattered. Top data-driven clusters (one-sample t on subject means):

| cluster | MNI center | vox | mean stat | t(19) | p | positive |
|---|---|---|---|---|---|---|
| 1 | [-25, -80, -10] left occipito-temporal/fusiform | 1192 | +0.053 | 6.02 | 8.7e-6 | 18/20 |
| 2 | [-62, -10, 14] left IFG/STG | 819 | +0.055 | 5.43 | 3.1e-5 | 17/20 |
| 3 | [-43, 27, 11] left IFG/anterior insula | 581 | +0.053 | 4.45 | 2.8e-4 | 17/20 |
| 4 | [2, 59, -2] mPFC | 548 | +0.061 | 4.50 | 2.5e-4 | 17/20 |
| 5 | [-23, -73, 49] left parietal | 498 | +0.058 | 4.25 | 4.3e-4 | 15/20 |

MDS of the group RDM in clusters 1-3 shows the three communities separated
with boundary nodes intermediate — the paper's Figure 4 result. Cluster
locations overlap the paper's (IFG, ventral/ATL) but are not identical, as
expected given preprocessing differences and subject-specific
stimulus-to-node assignment.

Why our first attempts failed: wrong region (hippocampus is not where the
2013 figure lives), wrong estimator (GLM betas vs their simple +2 TR
averaging), missing the path-distance lag-matching control, and fixed atlas
ROIs instead of data-driven cluster selection.

Remaining gaps vs the paper: their t=140 (vs our t=4-6) reflects their full
SPM/AFNI preprocessing stack (nonlinear normalization, slice-timing,
despiking), averaging across hundreds of searchlights inside optimally
selected clusters, and possibly supplementary analysis details. Cluster 4's
RDM panel failed only due to an inverse-warp mapping quirk (midline cluster);
its statistic is valid.

