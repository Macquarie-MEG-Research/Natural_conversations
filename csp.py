"""
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
"""

from pathlib import Path
import platform
import textwrap

import h5io
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps, cm
import PIL
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
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
csp_freqs['broadband'] = [4, 45] # add broadband decoding
n_components = 4
random_state = 42
n_splits = 5
n_proj = 4
n_exclude = 0  # must be zero unless on special branch that supports it
ch_type = None  # "eeg"  # None means all
whiten = True  # default is True
rerun = False  # force re-run / overwrite of existing files
src_type = 'surf' # surf or vol
mode = 'stat_map' # stat_map or glass_brain, vol source space plotting mode
randomize = False  # False or nonzero int, randomize the trial labels
decode = "interviewer"  # "participant" or "interviewer" turns, or "bada"

assert src_type in ("vol", "surf"), src_type
assert decode in ("participant", "interviewer", "bada"), decode
mode_extra = f"_{mode[0:5]}" if src_type == "vol" and mode == "glass_brain" else ""

plot_classification = True
plot_indiv = False
plot_correlations = True

# Construct the time bins
time_bins = np.array(config.decoding_csp_times)
assert time_bins.ndim == 1
time_bins = np.c_[time_bins[:-1], time_bins[1:]]
subjects = config.subjects
if subjects == "all":
    subjects = [f"{sub:02d}" for sub in range(1, 33)]
subjects = [subject for subject in subjects if subject not in config.exclude_subjects]
del config

use_subjects = subjects  # run all of them (could use e.g. subjects[2:3] just to run 03)
del subjects

data_path = deriv_path = Path(__file__).parents[1] / "Natural_Conversations_study" / "data"
analysis_path = deriv_path = Path(__file__).parents[1] / "Natural_Conversations_study" / "analysis"
if platform.system() == 'Windows':
    data_path = Path("D:/Work/analysis_ME206/data")
    analysis_path = Path("D:/Work/analysis_ME206/Natural_Conversations_study/analysis")

deriv_path = analysis_path / "natural-conversations-bids" / "derivatives"
fig_path = analysis_path / "figures" / "CSP-decoding"
if src_type == 'vol':
    fig_path = analysis_path / "figures" / "CSP-decoding-vol"
    results_path = analysis_path / "results" / "CSP-decoding-vol"
cop_path = analysis_path / "figures"
subjects_dir = deriv_path / "freesurfer" / "subjects"
if src_type == 'vol':
    src_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-vol-5-src.fif"
    bem_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
else:
    src_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-oct6-src.fif"
fs_vertices = [
    s["vertno"] for s in mne.read_source_spaces(
        src_fname
    )
]
n_vertices = sum(len(v) for v in fs_vertices)

# %%
# Loop over subjects to compute decoding scores and source space projections

title = f"N={len(use_subjects)} subjects, {n_components} components"
extra = ""
if n_proj or n_exclude or ch_type or not whiten or src_type != "surf" or randomize or decode != "participant":  # noqa: E501
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
    if src_type == 'vol':
        extra += "-vol"
        title += ", vol src"
    if randomize:
        extra += f"-rand{randomize}"
        title += f", random labels (seed {randomize})"
    if decode != "participant":
        extra += f"-{decode}"
        title += f", {decode}"
for si, sub in enumerate(use_subjects):  # just 03 for now
    path = deriv_path / 'mne-bids-pipeline' / f'sub-{sub}' / 'meg'
    epochs_fname = path / f'sub-{sub}_task-conversation_proc-clean_epo.fif'
    trans_fname = path / f'sub-{sub}_task-conversation_trans.fif'
    fwd_fname = path / f'sub-{sub}_task-conversation_fwd.fif'
    cov_fname = path / f'sub-{sub}_task-rest_proc-clean_cov.fif'
    inv_fname = path / f'sub-{sub}_task-conversation_inv.fif'
    out_fname = path / f'sub-{sub}_task-conversation_decoding{extra}_csp.h5'
    proj_fname = path / f'sub-{sub}_task-conversation_proc-proj_proj.fif'
    if out_fname.exists() and not rerun:
        continue

    print(f"Processing sub-{sub} ...")

    # Read data
    epochs = mne.read_epochs(epochs_fname).load_data()

    if n_proj:
        if rerun or not proj_fname.exists():
            print(f"  Loading raw data ...")
            # Using Raw
            raw = mne.concatenate_raws([
                mne.io.read_raw_fif(path / f"sub-{sub}_task-conversation_run-{run:02d}_proc-clean_raw.fif").load_data().resample(100, method="polyphase")
                for run in range(1, 6)  # 1 through 5 are conversation/repetition
            ])
            raw.filter(2, None, l_trans_bandwidth=1)  # we know it's broadband
            reject = dict(mag=5e-12, eeg=500e-6)
            proj = mne.compute_proj_raw(
                raw, n_mag=10, n_grad=0, n_eeg=10, reject=reject, verbose=True,
            )
            # Could use Epochs
            # proj_epochs = epochs.copy().filter(3, None, l_trans_bandwidth=2).crop(0, None)
            # proj = mne.compute_proj_epochs(
            #     proj_epochs, n_mag=10, n_grad=0, n_eeg=10, verbose=True,
            # )
            # Could use Evoked
            # proj = mne.compute_proj_evoked(
            #     proj_epochs.average(), n_mag=10, n_grad=0, n_eeg=10, verbose=True,
            # )
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
    if decode in ("participant", "interviewer"):
        conds = [f"{decode}_conversation", f"{decode}_repetition"]
    else:
        conds = ["ba", "da"]
    ids = [v for cond in conds for k, v in epochs.event_id.items() if k == cond]
    assert all(cond in epochs.event_id for cond in conds), (conds, list(epochs.event_id))
    if decode == "participant":  # based on alphebetical order
        assert ids == [5, 6], ids
    elif decode == "interviewer":
        assert ids == [3, 4], ids
    else:
        assert ids == [1, 2], ids
    epochs = epochs[conds].pick(["meg", "eeg"], exclude="bads")
    if ch_type:
        epochs.pick(ch_type)
    assert epochs.info["bads"] == []  # should have picked good only
    epochs.equalize_event_counts()
    labels = epochs.events[:, 2] # conversation=2, repetition=4
    assert np.isin(labels, ids).all(), np.unique(labels)
    if randomize:
        # Ensure that the randomization gets exactly half wrong (to within one)
        orig = labels.copy()
        n_cond = (labels == ids[0]).sum()
        want = [0] * 7
        want[ids[0]] = n_cond
        want[ids[1]] = n_cond
        assert list(np.bincount(labels, minlength=len(want))) == want
        n_switch = n_cond // 2
        rng = np.random.RandomState(randomize)
        idx_0 = np.where(labels == ids[0])[0]
        idx_1 = np.where(labels == ids[1])[0]
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)
        labels[idx_0[:n_switch]] = ids[1]
        labels[idx_1[:n_switch]] = ids[0]
        assert list(np.bincount(labels, minlength=len(want))) == want
        np.testing.assert_allclose((orig == labels).mean(), 0.5, atol=1.5 / n_cond)
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
    if n_proj or ch_type or src_type == 'vol':
        # Recreate inverse taking into account additional projections
        cov = mne.read_cov(cov_fname)
        if src_type == 'vol':
            src = mne.read_source_spaces(src_fname)
            fwd = mne.make_forward_solution(
                epochs.info,
                trans=trans_fname,
                src=src,
                bem=bem_fname,
                meg=True,
                eeg=True,
            )
            inv = mne.minimum_norm.make_inverse_operator(
                epochs.info, fwd, cov, rank=ranks,
            )
        else: # surface source space
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
                clf, X, labels, cv=cv, verbose=True, scoring="roc_auc", error_score="raise",
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
    path = deriv_path / 'mne-bids-pipeline' / f'sub-{sub}' / 'meg'
    dec_fname = path / f'sub-{sub}_task-conversation_decoding{extra}_csp.h5'
    data = h5io.read_hdf5(dec_fname)
    stc_data[si] = data["stc_data"]
    scores[si] = data["scores"]

# Binarize absolute value of STC coefficients: keep top 10th percentile of weights
# across vertices (-2), then sum across components (-1) and subjects (0)
subj_data = (stc_data >= np.percentile(stc_data, 90, axis=-2, keepdims=True)).sum(-1)
data_ = subj_data.sum(0)  # subjects
assert data_.shape == (len(csp_freqs), len(time_bins), n_vertices)


def clean_brain(brain_img):
    """Remove borders of a brain image and make transparent."""
    bg = (brain_img == brain_img[0, 0]).all(-1)
    brain_img = brain_img[(~bg).any(axis=-1)]
    brain_img = brain_img[:, (~bg).any(axis=0)]
    alpha = 255 * np.ones(brain_img.shape[:-1], np.uint8)
    x, y = np.where((brain_img == 255).all(-1))
    alpha[x, y] = 0
    return np.concatenate((brain_img, alpha[..., np.newaxis]), -1)


brain_kwargs = dict(
    hemi="split", views=("lat", "med"), time_viewer=False, background="w",
    size=(800, 600), smoothing_steps=5, colorbar=False, subjects_dir=subjects_dir,
)
if plot_classification or plot_indiv:
    all_datas = dict()
    if plot_classification:
        all_datas[""] = (data_, scores.mean(-1).mean(0))
    if plot_indiv:
        for si, subj in enumerate(use_subjects):
            all_datas[subj] = (subj_data[si], scores.mean(-1)[si])
    for subj_key, (this_data, this_scores) in all_datas.items():
        assert this_scores.shape == (len(csp_freqs), len(time_bins)), this_scores.shape
        fig, axes = plt.subplots(
            len(csp_freqs), len(time_bins), figsize=(12, 8), layout="constrained",
            squeeze=False,
        )
        fig.suptitle(title + (f", G{subj_key}" if subj_key else ""))
        assert this_data.shape == (len(csp_freqs), len(time_bins), n_vertices), this_data.shape  # noqa: E501
        if src_type == 'vol':
            stc = mne.VolSourceEstimate(
                this_data.reshape(-1, n_vertices).T,
                vertices=fs_vertices, tmin=0, tstep=1., subject="fsaverage",
            )
            # save the GA stc - this can be exported as Nifti, then loaded into other software (e.g. xjview) to extract anatomical labels
            if subj_key == "":                
                results_path.mkdir(exist_ok=True)
                stc.save(f'{results_path}/decoding{extra}_csp_GA') 
            src = mne.read_source_spaces(src_fname)
            # for vol src, the plot function returns a matplotlib Figure -
            # we can't update the clim & time point for this once plotted, so do the actual plotting later
        else:
            stc = mne.SourceEstimate(
                this_data.reshape(-1, n_vertices).T,
                vertices=fs_vertices, tmin=0, tstep=1., subject="fsaverage",
            )
            brain = stc.plot(
                initial_time=0., transparent=True,
                colormap="viridis", clim=dict(kind="value", lims=[0, 1, 2]),
                **brain_kwargs,
            )

        for bi, (band, (fmin, fmax)) in enumerate(csp_freqs.items()):
            if subj_key == "":
                vmax = np.max(stc.data[:, bi * len(time_bins):(bi + 1) * len(time_bins)])
                fmid = vmax / 2.
            else:
                vmax = n_components
                fmid = 1.
            assert vmax in range(len(use_subjects) * n_components + 1), vmax
            if src_type != 'vol':
                brain.update_lut(fmin=-0.5, fmid=fmid, fmax=vmax + 0.5)

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
                #print(ti, tmin, tmax)
                ax = axes[bi, ti]
                if bi == 0:
                    ax.set_title(f"{tmin} - {tmax}s")
                if ti == 0:
                    ax.set_ylabel(f"{fmin} - {fmax} Hz")
                ax.set(xticks=[], yticks=[], aspect="equal")
                # plot now and add as subplot
                if src_type == 'vol':
                    brain = stc.plot(src=src,
                        subject='fsaverage', subjects_dir=subjects_dir, verbose=True,
                        mode=mode,
                        initial_time=bi * len(time_bins) + ti, # idx 0-11, corresponding to the 4 freq bands * 3 time bins
                        #colorbar=False, # need to show colorbar in order to set clim
                        colormap="viridis",
                        clim=dict(kind="value", lims=[0, fmid, vmax + 0.5]),
                    )
                    brain.canvas.draw()
                    img = PIL.Image.frombytes('RGB',
                        brain.canvas.get_width_height(), brain.canvas.tostring_rgb()) # Note: brain.canvas.buffer_rgba() doesn't work
                    width, height = img.size
                    ax.imshow(img.crop((0, 50, width-50, height/2))) # tmp hack to remove the trace at bottom & make the img bigger
                    plt.close(brain)
                else:
                    brain.set_time_point(bi * len(time_bins) + ti)
                    ax.imshow(clean_brain(brain.screenshot()))
                for key in ax.spines:
                    ax.spines[key].set_visible(False)
                if src_type == 'vol':
                    vert_pos = 0
                else:
                    vert_pos = 0.5
                ax.text(
                    0.5, vert_pos, f"{this_scores[bi, ti]:0.2f}",
                    transform=ax.transAxes, ha="center", va="center",
                )

        if src_type != 'vol':
            brain.close()
        del brain
        fig_path.mkdir(exist_ok=True)
        subj_extra = f"_G{subj_key}" if subj_key else ""
        fig.savefig(fig_path / f"decoding{extra}{mode_extra}_csp{subj_extra}.png")
        if key:
            plt.close(fig=fig)

# %%
# Individual subject maps


# %%
# Correlations with copresence

co = pd.read_excel(data_path / "Copresence_questionnaire_(values_only).xlsx")
co.drop([
    "StartDate", "EndDate", "Status", "IPAddress", "Progress", "Duration (in seconds)",
    "Finished", "RecordedDate", "ResponseId", "RecipientLastName",
    "RecipientFirstName", "RecipientEmail", "ExternalReference", "LocationLatitude",
    "LocationLongitude", "DistributionChannel", "UserLanguage", "SessionDate",
    "SessionTime", "Gender",
], axis=1, inplace=True)
order = [list(co["ParticipantID"]).index(f"G{subj}") for subj in use_subjects]
co = co.reindex(order)
co.set_index("ParticipantID", inplace=True)
np.testing.assert_array_equal(co.index, [f"G{subj}" for subj in use_subjects])
co_kinds = list(co.columns)
co_values = np.array(co, float)
del co
if decode == "participant":
    toi = (-1.0, -0.5)
else:
    assert decode in ("interviewer", "bada")
    toi = (0, 0.5)
tidx = np.where((time_bins == toi).all(-1))[0]
assert len(tidx) == 1, tidx
tidx = tidx[0]
# Augment the array with the scores in the time of interest
# co_kinds += [f"{band} {toi[0]} {toi[1]}" for band in csp_freqs]
# co_values = np.c_[co_values, scores[:, :, tidx].mean(-1)]

indep = np.eye(len(use_subjects))
indep -= indep.mean(0)

if plot_correlations:
    # First characterize the questionnaire
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    im = axes[0].imshow(np.corrcoef(co_values.T) ** 2, vmin=0, vmax=1, cmap="magma")
    axes[0].set(xlabel="Age/Question/Score", xticks=[])
    ls = dict(Components="-", Uncorrelated="--")
    cs = dict(Components="C0", Uncorrelated="0.7")
    ms = dict(Components="o", Uncorrelated="none")
    use_u = use_s = use_v = None
    for which, vals in dict(Components=co_values, Uncorrelated=indep).items():
        c2 = vals - vals.mean(0)
        c2 /= np.linalg.norm(c2, axis=0)
        u, s, v = np.linalg.svd(c2, full_matrices=False)
        s **= 2
        s /= s.sum() / 100
        if which == "Components":
            use_u, use_s, use_v = u, s, v
        axes[1].plot(np.arange(1, len(s) + 1), np.cumsum(s), label=which,
                     color=cs[which], linestyle=ls[which], marker=ms[which])
        if which == "Components":
            df_out = pd.DataFrame(data=np.c_[s, v], columns=["var"] + list(co_kinds))
            df_out.to_csv(cop_path / "copresence_svd.csv", index=False)

    axes[1].legend(loc="lower right")
    axes[1].set(xlabel="Component", ylabel="Cumulative var exp (%)")
    fig.colorbar(im, ax=axes, label="R²", location="left", shrink=0.8)
    fig.savefig(cop_path / f"copresence_svd.png")

    # Next correlate with some questions of interest
    corr_types = ["_apriori", "_svd"]
    for corr_extra in corr_types:
        if corr_extra == "_apriori":
            qois = {  # Eventually we could pull from the sheet directly
                1: "Involvement - He/she was intensely involved in our conversation.",
                4: "Involvement - He/she was interested in talking.",
                9: "Social_distance - He/she made our conversation seem intimate.",
                16: "Composure - He/she felt very relaxed talking with me.",
                21: "Attraction - He/she created a sense of closeness between us.",
                34: "Task_orientation - He/she was open to my ideas.",
                38: "Depth_Similarity - He/she made me feel we had a lot in common.",
                45: "Trust_Receptivity - He/she was sincere.",
                46: "Trust_Receptivity - He/she was honest in communicating with me.",
                # 52: f"theta {toi[0]} {toi[1]}",
                # 53: f"alpha {toi[0]} {toi[1]}",
                # 54: f"beta {toi[0]} {toi[1]}",
                # 55: f"gamma {toi[0]} {toi[1]}",
            }
        elif corr_extra == "_svd":
            qois = {
                f"SVD{ii + 1}": (
                    "Left singular vector - Q"
                    + "•".join(f"{n}" for n in np.argsort(np.abs(use_v)[ii])[::-1][:10])
                    + f"… expvar={use_s[ii]:0.1f}%"
                )
                for ii in range(6)
            }
        fig, axes = plt.subplots(
            len(csp_freqs), len(qois), figsize=(2.5 * len(qois), 7),
            layout="constrained", squeeze=False,
        )
        fig.suptitle(title)
        cmap = "inferno"
        clim = [0.1, 0.3, 0.5] # [0.15, 0.35, 0.55]
        cmap_show = colormaps.get_cmap(cmap)
        cmaplist = np.array([cmap_show(i / 255) for i in range(256)])
        w = np.linspace(0, 1, 128, endpoint=False)
        cmaplist[:128] = (  # take first half of points and alpha them in with mid gray
            w[:, np.newaxis] * cmaplist[:128] +
            (1 - w[:, np.newaxis]) * np.array([0.5, 0.5, 0.5, 1])
        )
        cmap_show = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))
        for ci, (qi, this_title) in enumerate(qois.items()):
            if isinstance(qi, str):
                assert qi.startswith("SVD")
                this_a = use_u[:, int(qi[3:]) - 1]
            else:
                this_q = co_kinds[qi]
                this_a = co_values[:, qi]
                assert this_title.split()[0] in this_q, f"Title not found in {this_q}: {this_title}"
            assert this_a.shape == (len(use_subjects),), this_a.shape
            for bi, band in enumerate(csp_freqs):
                this_data = subj_data[:, bi, tidx, :]
                assert this_data.shape == (len(use_subjects), n_vertices), this_data.shape
                corrs = np.array([kendalltau(this_a, d).statistic for d in this_data.T])
                corrs[~np.isfinite(corrs)] = 0
                corrs = np.abs(corrs)
                ax = axes[bi, ci]
                if src_type == 'vol':
                    stc = mne.VolSourceEstimate(
                        corrs[:, np.newaxis],
                        vertices=fs_vertices, tmin=0, tstep=1., subject="fsaverage",
                    )
                    src = mne.read_source_spaces(src_fname)
                    brain = stc.plot(src=src,
                        subject='fsaverage', subjects_dir=subjects_dir, verbose=True,
                        mode=mode,
                        #colorbar=False,
                        colormap="inferno", clim=dict(kind="value", lims=clim),
                    )
                    brain.canvas.draw()
                    img = PIL.Image.frombytes('RGB',
                        brain.canvas.get_width_height(), brain.canvas.tostring_rgb()) # Note: brain.canvas.buffer_rgba() doesn't work
                    width, height = img.size
                    ax.imshow(img.crop((70, 50, width-105, height/2-20))) # tmp hack to remove the trace at bottom & make the img bigger
                    plt.close(brain)
                else:
                    stc = mne.SourceEstimate(corrs[:, np.newaxis], fs_vertices, 0, 1, "fsaverage")
                    brain = stc.plot(
                        colormap="inferno", clim=dict(kind="value", lims=clim),
                        **brain_kwargs,
                    )
                    ax.imshow(clean_brain(brain.screenshot()))
                    brain.close()
                ax.set(xticks=[], yticks=[], aspect="equal")
                for key in ax.spines:
                    ax.spines[key].set_visible(False)
                if bi == 0:
                    title_short = this_title.split("-")[0].strip()
                    desc = this_title.split("-", 1)[1].strip().rstrip(".")
                    desc = desc.lstrip("He/she").strip()
                    desc = "\n".join(textwrap.wrap(desc, 30))
                    ax.set_title(f"{qi}: {title_short}\n{desc}", fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"{band}: {scores[:, bi, tidx, :].mean():0.2f}")
        sm = cm.ScalarMappable(norm=colors.Normalize(clim[0], clim[2]), cmap=cmap_show)
        fig.colorbar(sm, ax=axes, label="Kendall's τ", location="bottom", shrink=0.1)
        fig.savefig(
            fig_path / f"copresence_correlations{corr_extra}{extra}{mode_extra}.png"
        )

# %%
