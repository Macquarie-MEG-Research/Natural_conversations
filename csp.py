'''
CSP is computed by mne-bids-pipeline (plots are shown in the report, and decoding
scores are saved in an excel file), but the classifiers themselves are not saved.

Here we write a script that gives the same results as the pipeline (so we can
extract the CSP output for individual subjects), then project to source space.

General steps:

1) optionally add additional continuous (from raw) projections to epochs and inverse
2) band-pass filter the epochs with epochs.filter
3) mne.decoding.Scaler to deal with channel types
4) sklearn PCA to reduce rank to that of the data
5) mne.decoding.CSP
6) Logistic Regression
7) Project patterns to source space
8) Save the results
9) Plot the results

We filter with a transition bandwidth of 2 Hz, and use a minimum phase filter to
keep things causial (avoid artifact leakage backward in time), but needs
https://github.com/mne-tools/mne-python/pull/12507 (merged 2024/03/19) to
filter properly.

TODO:
- Fix MNE-Python bug with get_coef
'''

from pathlib import Path

import h5io
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps, cm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import mne
from mne.decoding import CSP, UnsupervisedSpatialFilter, Scaler, LinearModel, get_coef

import config  # our config.py file


# adjust mne options to fix rendering issues (only needed in Linux / WSL)
mne.viz.set_3d_options(
    antialias=False, depth_peeling=False, smooth_shading=False, multi_samples=1,
)

csp_freqs = config.decoding_csp_freqs
n_components = 4
random_state = 42
n_splits = 5
n_proj = 4
n_exclude = 0  # must be zero unless on special branch that supports it
ch_type = None  # "eeg"  # None means all
whiten = True  # default is True
rerun = False  # force re-run / overwrite of existing files

# Construct the time bins
time_bins = np.array(config.decoding_csp_times)
assert time_bins.ndim == 1
time_bins = np.c_[time_bins[:-1], time_bins[1:]]
del config

subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
            '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24',
            '25', '26', '27'] # excluding subj 12 & 20

#path = Path(__file__).parents[1] / "Natural_Conversations_study" / "analysis" / 'natural-conversations-bids' / 'derivatives' / 'mne-bids-pipeline' / 'sub-' + sub / 'meg'
#epochs_fname = path / 'sub-' + sub + '_task-conversation_proc-clean_epo.fif'
path = Path('/mnt/d/Work/analysis_ME206/processing/bids/all/')
#epochs_fname = path + 'sub-' + sub + '_task-conversation_proc-clean_epo.fif'
#inv_fname = path + 'sub-' + sub + '_task-conversation_inv.fif'
save_path = Path('/mnt/d/Work/analysis_ME206/results/bids/CSP/')
fig_path = save_path / "figures"
subjects_dir = Path('/mnt/d/Work/analysis_ME206/processing/mri/') # only for plotting stc
'''
analysis_path = deriv_path = Path(__file__).parents[1] / "Natural_Conversations_study" / "analysis"
analysis_path = '/mnt/d/Work/analysis_ME206/processing/bids/all/'
deriv_path = analysis_path / "natural-conversations-bids" / "derivatives"
fig_path = analysis_path / "figures"
'''

use_subjects = subjects  # run all of them (could use e.g. subjects[2:3] just to run 03)
fs_vertices = [
    s["vertno"] for s in mne.read_source_spaces(
        subjects_dir / "fsaverage" / "bem" / "fsaverage-oct6-src.fif"
    )
]
n_vertices = sum(len(v) for v in fs_vertices)

# %%
# Loop over subjects to compute decoding scores and source space projections

title = f"N={len(use_subjects)} subjects, {n_components} components"
extra = ""
if n_proj or n_exclude or ch_type or not whiten:
    extra += "_proc"
    if n_proj:
        extra += f"-{n_proj}proj"
        title += f", {n_proj} proj"
    if n_exclude:
        extra += f"-{n_exclude}excl"
        title += f", first {n_exclude} excluded"
    if ch_type:
        extra += f"-{ch_type}"
        title += f", {ch_type} only"
    if not whiten:
        extra += "-nowhiten"
        title += ", no whitening"
        if not ch_type:
            raise RuntimeError("Must whiten when ch_type is None")
for si, sub in enumerate(use_subjects):  # just 03 for now
    #path = deriv_path / 'mne-bids-pipeline' / f'sub-{sub}' / 'meg'
    epochs_fname = path / f'sub-{sub}_task-conversation_proc-clean_epo.fif'
    fwd_fname = path / f'sub-{sub}_task-conversation_fwd.fif'
    cov_fname = path / f'sub-{sub}_task-rest_proc-clean_cov.fif'
    inv_fname = path / f'sub-{sub}_task-conversation_inv.fif'
    out_fname = save_path / f'sub-{sub}_task-conversation_decoding{extra}_csp.h5'
    proj_fname = path / f'sub-{sub}_task-conversation_proc-proj_proj.fif'
    if out_fname.exists() and not rerun:
        continue

    print(f"Processing sub-{sub} ...")

    # Read data
    epochs = mne.read_epochs(epochs_fname).load_data()

    if n_proj:
        if rerun or not proj_fname.exists():
            print(f"  Loading raw data ...")
            import time
            t0 = time.time()
            raw = mne.concatenate_raws([
                mne.io.read_raw_fif(path / f"sub-{sub}_task-conversation_run-{run:02d}_proc-clean_raw.fif").load_data().resample(100, method="polyphase")
                for run in range(1, 6)  # 1 through 5 are conversation/repetition
            ])
            raw.filter(2, None, l_trans_bandwidth=1)  # we know it's broadband
            reject = dict(mag=5e-12, eeg=500e-6)
            proj = mne.compute_proj_raw(
                raw, n_mag=10, n_grad=0, n_eeg=10, reject=reject, verbose=True,
            )
            assert len(proj) == 20
            mne.write_proj(proj_fname, proj, overwrite=True)
        all_proj = mne.read_proj(proj_fname)
        proj = list()
        for ii, kind in enumerate(("MEG", "EEG")):
            these_proj = all_proj[10 * ii:10 * ii + n_proj]
            tot_exp = 100 * sum(p["explained_var"] for p in these_proj)
            print(f"  {kind} {n_proj=} raw exp var: {tot_exp:0.1f}%")
            proj.extend(these_proj)
        del all_proj
        epochs.add_proj(proj).apply_proj()

    # only select the conditions we are interested in
    epochs = epochs[['conversation', 'repetition']].pick(["meg", "eeg"], exclude="bads")
    if ch_type:
        epochs.pick(ch_type)
    assert epochs.info["bads"] == []  # should have picked good only
    epochs.equalize_event_counts()
    labels = epochs.events[:, 2] # conversation=2, repetition=4
    ranks = mne.compute_rank(inst=epochs, tol=1e-3, tol_kind="relative")
    rank = sum(ranks.values())
    print(f"  Ranks={ranks} (total={rank})")
    scaler = Scaler(epochs.info)
    pca = UnsupervisedSpatialFilter(PCA(rank, whiten=whiten), average=False)
    kwargs = dict()
    if n_exclude:
        kwargs["n_exclude"] = n_exclude
    csp = CSP(
        n_components=n_components,
        reg=0.1,
        log=False,
        **kwargs,
    )
    lr = LinearModel(LogisticRegression(solver="liblinear", random_state=random_state))
    steps = [("scaler", scaler), ("PCA", pca), ("CSP", csp), ("LR", lr)]
    clf = Pipeline(steps)
    if n_proj or ch_type:
        # Recreate inverse taking into account additional projections
        cov = mne.read_cov(cov_fname)
        fwd = mne.read_forward_solution(fwd_fname)
        inv = mne.minimum_norm.make_inverse_operator(
            epochs.info, fwd, cov, loose=0.2, depth=0.8, rank=ranks,
        )
    else:
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    assert inv["src"][0]["subject_his_id"] == "fsaverage"
    for si, s in enumerate(inv["src"]):
        assert si in range(2)
        np.testing.assert_array_equal(fs_vertices[si], s["vertno"])

    # Loop over frequency bands x time bins
    sub_stc_data = np.zeros((len(csp_freqs), len(time_bins), n_vertices, n_components))
    sub_scores = np.zeros((len(csp_freqs), len(time_bins), n_splits))
    for bi, (band, (fmin, fmax)) in enumerate(csp_freqs.items()):
        # 0) band-pass filtering the epochs to get the relevant freq band
        epochs_filt = epochs.copy().filter(
            fmin, fmax, l_trans_bandwidth=2., h_trans_bandwidth=2., verbose="error",
            phase="minimum",
        )
        for ti, (tmin, tmax) in enumerate(time_bins):
            # Crop data to the time window of interest
            if tmax is not None:  # avoid warnings about outside the interval
                tmax = min(tmax, epochs_filt.times[-1])

            # Get the data for all time points
            X = epochs_filt.copy().crop(tmin, tmax).get_data(copy=False)

            # Calculate the decoding scores
            cv = StratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True,
            )
            sub_scores[bi, ti] = cross_val_score(
                clf, X, labels, cv=cv, verbose=True, scoring="roc_auc",
            )
            print(
                f"  {band.ljust(5)} {tmin} - {tmax}s: "
                f"{np.mean(sub_scores[bi, ti]):0.2f}"
            )

            # project CSP patterns to source space
            clf.fit(X, labels)
            # In theory we should be able to extract the coef from the classifier:
            # coef = get_coef(clf, "patterns_", inverse_transform=True, verbose=True)
            # https://github.com/mne-tools/mne-python/issues/12502
            coef = csp.patterns_[n_exclude:n_exclude + n_components]
            assert coef.shape == (n_components, pca.estimator.n_components_), coef.shape
            coef = pca.estimator.inverse_transform(coef)
            assert coef.shape == (n_components, len(epochs.ch_names)), coef.shape
            coef = scaler.inverse_transform(coef.T[np.newaxis])[0]
            assert coef.shape == (len(epochs.ch_names), n_components), coef.shape
            evoked = mne.EvokedArray(coef, epochs.info, tmin=0, nave=len(epochs) // 2)
            stc = mne.minimum_norm.apply_inverse(evoked, inv, 1.0 / 9.0, "dSPM")
            assert stc.data.min() >= 0, stc.data.min()  # loose should do this already
            #if sub == "03" and fmin == 14 and tmin == -1.5:  # sub_scores[bi, ti].mean() > 0.9:
            #    brain = stc.plot(
            #        hemi="split", views=("lat", "med"), initial_time=0.,
            #        subjects_dir=subjects_dir, time_viewer=True,
            #    )
            #    raise RuntimeError
            sub_stc_data[bi, ti] = stc.data

    # Save the results
    h5io.write_hdf5(
        out_fname,
        {"stc_data": sub_stc_data, "scores": sub_scores},
        overwrite=True,
    )
    del sub_stc_data, sub_scores

# %%
# Plot the results

stc_data = np.zeros(
    (len(use_subjects), len(csp_freqs), len(time_bins), n_vertices, n_components),
)
scores = np.zeros(
    (len(use_subjects), len(csp_freqs), len(time_bins), n_splits),
)
for si, sub in enumerate(use_subjects):
    #path = deriv_path / 'mne-bids-pipeline' / f'sub-{sub}' / 'meg'
    dec_fname = save_path / f'sub-{sub}_task-conversation_decoding{extra}_csp.h5'
    data = h5io.read_hdf5(dec_fname)
    stc_data[si] = data["stc_data"]
    scores[si] = data["scores"]

# Binarize absolute value of STC coefficients: keep top 10th percentile of weights
# across vertices (-2), then sum across components (-1) and subjects (0)
data = (stc_data >= np.percentile(stc_data, 90, axis=-2, keepdims=True)).sum(-1).sum(0)
assert data.shape == (len(csp_freqs), len(time_bins), n_vertices)
data.shape = (-1, n_vertices)
stc = mne.SourceEstimate(
    data.T, vertices=fs_vertices, tmin=0, tstep=1., subject="fsaverage",
)
assert data.min() == 0
del data

fig, axes = plt.subplots(
    len(csp_freqs), len(time_bins), figsize=(12, 8), layout="constrained",
    squeeze=False,
)
fig.suptitle(title)
brain = stc.plot(
    hemi="split", views=("lat", "med"), initial_time=0., subjects_dir=subjects_dir,
    background="w", size=(800, 600), time_viewer=False, colormap="viridis",
    clim=dict(kind="value", lims=[0, 1, 2]), smoothing_steps=5, colorbar=False,
    transparent=True,
)


def clean_brain(brain_img):
    """Remove borders of a brain image and make transparent."""
    bg = (brain_img == brain_img[0, 0]).all(-1)
    brain_img = brain_img[(~bg).any(axis=-1)]
    brain_img = brain_img[:, (~bg).any(axis=0)]
    alpha = 255 * np.ones(brain_img.shape[:-1], np.uint8)
    x, y = np.where((brain_img == 255).all(-1))
    alpha[x, y] = 0
    return np.concatenate((brain_img, alpha[..., np.newaxis]), -1)


for bi, (band, (fmin, fmax)) in enumerate(csp_freqs.items()):
    vmax = np.max(stc.data[:, bi * len(time_bins):(bi + 1) * len(time_bins)])
    assert vmax in range(len(use_subjects) * n_components + 1), vmax
    brain.update_lut(fmin=-0.5, fmid=vmax / 2., fmax=vmax + 0.5)

    cmap = colormaps.get_cmap("viridis")
    cmaplist = np.array([cmap(i / vmax) for i in range(vmax + 1)])
    w = np.linspace(0, 1, vmax // 2, endpoint=False)
    cmaplist[:vmax // 2] = (  # take first half of points and alpha them in with mid gray
        w[:, np.newaxis] * cmaplist[:vmax // 2] +
        (1 - w[:, np.newaxis]) * np.array([0.5, 0.5, 0.5, 0])
    )
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))
    del cmaplist, w

    norm = colors.BoundaryNorm(np.arange(0, vmax + 2) - 0.5, vmax + 1, clip=True)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ticks = np.arange(0, vmax + 1)
    ticks = ticks[np.round(np.linspace(0, vmax, min(5, vmax + 1))).astype(int)]
    cb = fig.colorbar(
        sm, ax=axes[bi], orientation="vertical", label="subject x component",
        ticks=ticks, aspect=10, shrink=0.8,
    )
    cb.ax.patch.set_color('0.5')

    for ti, (tmin, tmax) in enumerate(time_bins):
        ax = axes[bi, ti]
        if bi == 0:
            ax.set_title(f"{tmin} - {tmax}s")
        if ti == 0:
            ax.set_ylabel(f"{fmin} - {fmax} Hz")
        ax.set(xticks=[], yticks=[], aspect="equal")
        brain.set_time_point(bi * len(time_bins) + ti)
        ax.imshow(clean_brain(brain.screenshot()))
        for key in ax.spines:
            ax.spines[key].set_visible(False)
        ax.text(
            0.5, 0.5, f"{scores.mean(-1).mean(0)[bi, ti]:0.2f}",
            transform=ax.transAxes, ha="center", va="center",
        )

brain.close()
del brain
fig_path.mkdir(exist_ok=True)
fig.savefig(fig_path / f"decoding{extra}_csp.png")
