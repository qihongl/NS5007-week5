#!/usr/bin/env python3
"""Generate the packaged data for the NS5007 week-5 lab (PCA & MDS on the
Schapiro et al. 2013 temporal-community-structure fMRI experiment).

Inputs (produced locally by reproduce_v4_wholebrain.py / preprocess_mc.py /
reproduce_v3_matched.py from OpenNeuro ds001621, raw BOLD):
  * motion-corrected + despiked masked timecourses  scaleup_data/mc/*.npz
  * the group t-map from the whole-brain searchlight  out/v4_tmap.npy

Outputs (into the course folder, data/):
  week5_schapiro_patterns.npz
      - item x voxel pattern matrices (15 x <=512 voxels, float32) for
        20 subjects x 4 regions:
          IFG_insula_L   left IFG/anterior insula  (paper Table 1 cluster,
                                                      MNI ~ [-43, 27, 11])
          OccTemp_L      left occipito-temporal    (our strongest cluster,
                                                      MNI ~ [-25, -80, -10])
          Hippocampus    bilateral hippocampus     (control: no pattern
                                                      effect reported in 2013)
          VisualControl  intracalcarine (V1)       (stimulus-driven control)
        voxels are z-scored across items per voxel (as in the paper's recipe)
      - per-subject Hamiltonian path positions pathpos__<sid> (15 ints):
          path position (1..15) of each node A1..C5 in that subject's fixed
          Hamiltonian path -- needed for the lag-matched control statistic
      - node metadata (community codes, boundary flags) and region metadata
  week5_schapiro_tmap.npz
      - group one-sample t-map across 20 subjects of the whole-brain
        searchlight statistic (lag-matched), on the MNI 1 mm grid
      - the five largest t>3 cluster masks, their MNI peak coordinates and
        one-sample statistics

Provenance: raw data https://openneuro.org/ds001621 (Schapiro, Rogers,
Cordova, Turk-Browne & Botvinick, 2013, Nat Neurosci, Experiment 3).
Preprocessing applied offline: despiking, 6-parameter rigid motion
correction, ANTs affine EPI->MNI normalization. Deviations from the paper's
AFNI/SPM stack are documented in the lab notebook.
"""
import os, sys
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from nilearn.datasets import load_mni152_template, fetch_atlas_harvard_oxford

TPL = load_mni152_template(resolution=1)
NSHIFT = 12

BASE = '/tmp/wk5build'
DATA = f'{BASE}/scaleup_data'
OUT = f'{BASE}/out'
COURSE = ('/Users/qlu/Library/Mobile Documents/com~apple~CloudDocs/'
          'My teaching/2026-NS5007')
DEST = f'{COURSE}/data'
os.makedirs(DEST, exist_ok=True)

RUNS = ['run-02', 'run-03', 'run-04', 'run-05']
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 21)]
NODES = [f'{c}{i}' for c in 'ABC' for i in range(1, 6)]
MAX_VOX = 512

def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)
import time

# ---- reuse the item-matrix builder from the whole-brain script
sys.path.insert(0, BASE)
import reproduce_v4_wholebrain as V4
import reproduce_v3_matched as M3

def lag_masks_for(sid):
    ev0 = pd.read_csv(f'{DATA}/{sid}_{RUNS[0]}_events.tsv', sep='\t')
    items0 = ev0['item'].str.strip().tolist()
    _, blocks = M3.segment_hp_blocks(items0)
    path = items0[blocks[0]:blocks[0]+15]
    pathpos = np.array([path.index(nd) + 1 for nd in NODES])
    masks = []
    for d in (1, 2, 3, 4):
        w = np.zeros((15, 15), bool); b = np.zeros((15, 15), bool)
        for a, u in enumerate(NODES):
            for b_, v in enumerate(NODES):
                if a < b_ and abs(pathpos[a] - pathpos[b_]) == d:
                    (w if u[0] == v[0] else b)[a, b_] = True
        masks.append((w, b))
    return pathpos, masks

def stats_from_patterns(P, pathpos, masks):
    """P: (15, V) z-scored patterns. Returns lag-matched & naive statistics."""
    M = P - P.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True); sd[sd == 0] = 1
    R = (M / sd) @ (M / sd).T / P.shape[1]
    Fz = np.arctanh(np.clip(R, -0.999, 0.999))
    Fz = (Fz + Fz.T) / 2
    iu = np.triu_indices(15, 1)
    comm = np.array(['ABC'.index(nd[0]) for nd in NODES])
    same = comm[iu[0]] == comm[iu[1]]
    naive = Fz[iu][same].mean() - Fz[iu][~same].mean()
    lags = []
    for w, b in masks:
        lags.append(Fz[w].mean() - Fz[b].mean())
    return float(np.mean(lags)), float(naive)

def load_subject_runs(sid):
    trials = M3.qualifying_trials(sid)
    runs_data, inter = [], None
    for run in RUNS:
        trs = [t for t in trials if t[0] == run]
        if not trs:
            continue
        mc = np.load(f'{DATA}/mc/{sid}_{run}_masked.npz')
        Zr = mc['Z'].astype(np.float32)
        TR = float(mc['TR'])
        idx_r = np.flatnonzero(mc['mask'])
        sd = Zr.std(axis=0); sd[sd == 0] = 1.0
        Zr = (Zr - Zr.mean(axis=0)) / sd
        m = np.zeros(mc['mask'].size, bool); m[idx_r] = True
        inter = m.copy() if inter is None else (inter & m)
        runs_data.append((Zr, idx_r, TR, trs))
    bidx = np.full(inter.size, -1, dtype=np.int64)
    bidx[np.flatnonzero(inter)] = np.arange(int(inter.sum()))
    return trials, runs_data, inter, bidx

def compute_shift_nulls(sid, mni_masks, nshift=NSHIFT):
    import ants
    trials, runs_data, inter, bidx = load_subject_runs(sid)
    zm0 = np.load(f'{DATA}/mc/{sid}_{RUNS[0]}_masked.npz')
    shape3 = tuple(int(v) for v in zm0['shape'])
    raw = nib.load(f'{DATA}/{sid}_{RUNS[0]}_bold.nii.gz')
    reg = M3_reg_cached(sid)
    fixed_nat = M3.to_ants(nib.Nifti1Image(np.zeros(shape3, np.uint8),
                                           raw.affine))
    pathpos, lagmasks = lag_masks_for(sid)
    out = {}
    for rname, mni_mask in mni_masks.items():
        m = mni_mask
        if rname in ('IFG_insula_L', 'OccTemp_L'):
            # small data-driven clusters: dilate so the inverse-warp still
            # covers anatomy across subjects (null computation only)
            m = ndimage.binary_dilation(m, iterations=3)
        w = ants.apply_transforms(fixed_nat, M3.to_ants(
            nib.Nifti1Image(m.astype(np.uint8), TPL.affine)),
            reg['invtransforms'])
        natmask = np.asarray(w.numpy()) > 0
        # per-run column positions for this region
        acc0 = {it: [] for it in NODES}
        accs = [{it: [] for it in NODES} for _ in range(nshift)]
        shift_sets = []
        for (Zr, idx_r, TR, trs) in runs_data:
            sd = Zr.std(axis=0); sd[sd == 0] = 1.0
            Zz = (Zr - Zr.mean(axis=0)) / sd
            pos = np.searchsorted(
                idx_r, np.flatnonzero(natmask.reshape(-1) & inter))
            shift_sets.append([np.random.default_rng(60000 + s).integers(
                40, 270, size=len(pos)) for s in range(nshift)])
            for (r_, item, ei, onset, pos_i) in trs:
                fr = int(round(onset / TR)) + 2
                fr %= Zz.shape[0]
                acc0[item].append(Zz[fr][pos])
                for s in range(nshift):
                    sh = shift_sets[-1][s]
                    accs[s][item].append(Zz[(fr - sh) % Zz.shape[0], pos])
        def stat(acc):
            P = np.stack([np.mean(np.stack(acc[it]), axis=0) for it in NODES])
            return stats_from_patterns(P, pathpos, lagmasks)[0]
        out[rname] = (stat(acc0),
                      np.array([stat(accs[s]) for s in range(nshift)],
                               dtype=np.float32))
    return out

def main():
    from nilearn.image import resample_to_img
    import ants
    from scipy import ndimage

    # ---------------- region masks in MNI space (1 mm template grid)
    tpl = TPL
    sub = fetch_atlas_harvard_oxford('sub-maxprob-thr0-1mm')
    cor = fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
    def atlas_sel(atlas, keys, left_only=False):
        lab = np.asarray(atlas.maps.dataobj)
        ids = [i for i, l in enumerate(atlas.labels)
               if any(k.lower() in str(l).lower() for k in keys)]
        m = np.isin(lab, ids)
        if left_only:
            xs = (np.arange(m.shape[0])[:, None, None] * atlas.maps.affine[0, 0]
                  + atlas.maps.affine[0, 3])
            m = m & (xs < 0)
        return m
    cluster_masks = {}
    tmap = np.load(f'{OUT}/v4_tmap.npy')
    lab, ncl = ndimage.label(tmap > 3.0)
    sizes = ndimage.sum(tmap > 3.0, lab, range(1, ncl + 1))
    # cluster 1 = biggest, cluster 3 = 3rd biggest (as ranked in v4: by size)
    order = np.argsort(sizes)[::-1]
    big1 = order[0] + 1     # occipito-temporal  [-25,-80,-10]
    big3 = order[2] + 1     # IFG/anterior insula [-43, 27, 11]
    cluster_masks['OccTemp_L'] = (lab == big1)
    cluster_masks['IFG_insula_L'] = (lab == big3)
    mni_regions = {
        'IFG_insula_L': cluster_masks['IFG_insula_L'],
        'OccTemp_L': cluster_masks['OccTemp_L'],
        'Hippocampus': atlas_sel(sub, ['Hippocampus']),
        'VisualControl': atlas_sel(cor, ['Intracalcarine Cortex']),
    }

    patterns = {}
    meta_regions = {}
    rng = np.random.default_rng(0)
    _items = {}
    def get_items(sid):
        """item matrix (Vbrain x 15) on the subject's common mask + helpers."""
        if sid not in _items:
            item_mat, common, common_idx = V4.subject_items(sid)
            zm0 = np.load(f'{DATA}/mc/{sid}_{RUNS[0]}_masked.npz')
            shape3 = tuple(int(v) for v in zm0['shape'])
            raw = nib.load(f'{DATA}/{sid}_{RUNS[0]}_bold.nii.gz')
            mask_flat = np.zeros(common.size, bool)
            mask_flat[np.flatnonzero(common)] = True
            _items[sid] = (item_mat, mask_flat, shape3, raw.affine)
        return _items[sid]

    fixed_mni = M3.to_ants(tpl)
    for rname, mni_mask in mni_regions.items():
        mni_mask_flat = mni_mask.reshape(-1) > 0
        for i, sid in enumerate(SUBJECTS):
            item_mat, mask_flat, shape3, affine = get_items(sid)
            reg = M3_reg_cached(sid)
            # warp the 15 item maps from native space onto the MNI grid
            Pmni = np.zeros((15, int(mni_mask_flat.sum())), np.float32)
            cand = np.flatnonzero(mni_mask_flat)
            for it in range(15):
                vol = np.zeros(shape3, np.float32)
                vol.reshape(-1)[mask_flat] = item_mat[:, it]
                w = ants.apply_transforms(fixed_mni, M3.to_ants(
                    nib.Nifti1Image(vol, affine)), reg['fwdtransforms'])
                Pmni[it] = np.asarray(w.numpy(), dtype=np.float32).reshape(-1)[cand]
            # drop empty voxels (outside the warped item maps' support)
            good = Pmni.std(axis=0) > 0
            cols = cand[good]
            P = Pmni[:, good]
            if P.shape[1] > MAX_VOX:
                keep = np.sort(rng.choice(P.shape[1], MAX_VOX, replace=False))
                P, cols = P[:, keep], cols[keep]
            # re-standardize each voxel across items after resampling
            sd = P.std(axis=1, keepdims=True); sd[sd == 0] = 1
            P = (P - P.mean(axis=1, keepdims=True)) / sd
            patterns[f'{rname}__{sid}'] = P.astype(np.float32)
        meta_regions[rname] = dict(
            n_subjects=len(SUBJECTS), n_voxels=int(P.shape[1]),
            note=('data-driven searchlight cluster' if rname in
                  ('IFG_insula_L', 'OccTemp_L') else 'anatomical ROI (Harvard-Oxford)'))
        log(f'{rname}: packaged for {len(SUBJECTS)} subjects '
            f'({P.shape[1]} voxels each)')

    meta = {}
    for i, sid in enumerate(SUBJECTS):
        pp, masks = lag_masks_for(sid)
        meta[f'pathpos__{sid}'] = pp.astype(np.int16)
    patterns['subjects'] = np.array(SUBJECTS)
    patterns['nodes'] = np.array(NODES)
    patterns['community'] = np.array([nd[0] for nd in NODES])
    patterns['boundary'] = np.array([nd[1] in '15' for nd in NODES])
    patterns['regions'] = np.array(list(mni_regions))
    import json
    patterns['region_info'] = np.array(json.dumps(meta_regions))
    np.savez_compressed(f'{DEST}/week5_schapiro_patterns.npz', **patterns)

    # ---------------- shift-predictor nulls (the decisive control)
    # For each subject & region: the lag-matched statistic computed after
    # applying a per-voxel circular time-shift to the (detrended,
    # motion-corrected) timecourses. A shift destroys item-time
    # correspondence while preserving temporal autocorrelation, so the null
    # distribution quantifies how much "community structure" pure
    # time-structure can produce. NSHIFT draws per subject.


    from multiprocessing import get_context
    mni_masks = dict(mni_regions)
    with get_context('spawn').Pool(min(6, os.cpu_count() or 1)) as pool:
        per_subj = pool.starmap(compute_shift_nulls,
                                [(sid, mni_masks, NSHIFT) for sid in SUBJECTS])
    for i, sid in enumerate(SUBJECTS):
        for r in mni_regions:
            obs, nulls = per_subj[i][r]
            meta[f'obsnat__{r}'] = (
                meta[f'obsnat__{r}'] if f'obsnat__{r}' in meta
                else np.zeros(len(SUBJECTS), np.float32))
            meta[f'obsnat__{r}'][i] = obs
            if f'shiftnull__{r}' not in meta:
                meta[f'shiftnull__{r}'] = np.zeros((len(SUBJECTS), NSHIFT),
                                                   np.float32)
            meta[f'shiftnull__{r}'][i] = nulls
        log(f'  {sid}: shift-predictor nulls done')
    for r in mni_regions:
        meta[f'shiftnull__{r}'] = meta[f'shiftnull__{r}']
    np.savez_compressed(f'{DEST}/week5_schapiro_shifts.npz', **meta)
    log('wrote', f'{DEST}/week5_schapiro_shifts.npz',
        f'({os.path.getsize(f"{DEST}/week5_schapiro_shifts.npz")/1e3:.0f} KB)')


    # ---------------- t-map package
    lab2, ncl2 = ndimage.label(tmap > 3.0)
    sizes2 = ndimage.sum(tmap > 3.0, lab2, range(1, ncl2 + 1))
    order2 = np.argsort(sizes2)[::-1][:5]
    cluster_out = {}
    info = []
    for rank, k in enumerate(order2, 1):
        cmask = (lab2 == k + 1)
        cluster_out[f'cluster{rank}_mask'] = cmask
        vox = np.array(np.nonzero(cmask)).T
        peak = (tpl.affine[:3, :3] @ vox[np.argmax(tmap[cmask])].T + tpl.affine[:3, 3])
        info.append(dict(cluster=rank, n_vox=int(cmask.sum()),
                         peak_mni=[round(float(v), 1) for v in peak]))
    np.savez_compressed(f'{DEST}/week5_schapiro_tmap.npz',
                        tmap=tmap.astype(np.float32),
                        affine=tpl.affine,
                        **cluster_out)
    with open(f'{DEST}/week5_schapiro_tmap_clusters.json', 'w') as f:
        json.dump(info, f, indent=1)
    log('wrote', DEST)
    for fn in ('week5_schapiro_patterns.npz', 'week5_schapiro_tmap.npz'):
        p = f'{DEST}/{fn}'
        log(f'  {fn}: {os.path.getsize(p)/1e6:.1f} MB')

_MREG = {}
def M3_reg_cached(sid):
    import ants
    from nilearn.image import mean_img
    if sid in _MREG:
        return _MREG[sid]
    bolds = {r: f'{DATA}/{sid}_{r}_bold.nii.gz' for r in RUNS}
    ref_img = mean_img([mean_img(bolds[r]) for r in RUNS])
    fixed = M3.to_ants(load_mni152_template(resolution=1))
    moving = M3.to_ants(ref_img)
    _MREG[sid] = ants.registration(fixed=fixed, moving=moving,
                                   type_of_transform='Affine')
    return _MREG[sid]

if __name__ == '__main__':
    import json
    main()
