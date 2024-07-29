import os
import mne
import numpy as np
from scipy import stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from scipy.ndimage import label

# Define constants
SUBJECTS_DIR = "/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/freesurfer/subjects"
STC_DIR = "/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/LCMV_analysis"
SRC_FNAME = os.path.join(SUBJECTS_DIR, "fsaverage/bem/fsaverage-oct6-src.fif")
FREQUENCY_BANDS = ["broadband"]


def load_stcs(directory, band):
    """Load Source Time Courses (STCs) for a given frequency band and crop them."""
    stcs = []
    for filename in os.listdir(directory):
        if filename.endswith(
            f"{band}_lcmv_beamformer_participant_conversation_vs_participant_repetition_difference-stc-rh.stc"
        ):
            stc_path = os.path.join(directory, filename)
            stc = mne.read_source_estimate(stc_path)

            # Crop the STC
            cropped_stc = stc.crop(tmin=0.25, tmax=0.75)

            stcs.append(cropped_stc)
    return stcs


def process_frequency_band(band, src):
    """Process data for a single frequency band and return aggregated timeseries for significant clusters."""
    print(f"Processing {band} band")

    stcs = load_stcs(STC_DIR, band)
    if not stcs:
        print(f"No STCs found for {band} band")
        return None

    avg_stc = sum(stcs) / len(stcs)

    X = np.array([stc.data for stc in stcs])
    X = np.transpose(X, [0, 2, 1])

    t_threshold = stats.distributions.t.ppf(1 - 0.001, len(X) - 1)
    adjacency = mne.spatial_src_adjacency(src[:])

    cluster_stats = spatio_temporal_cluster_1samp_test(
        X,
        threshold=t_threshold,
        adjacency=adjacency,
        n_jobs=-1,
        buffer_size=None,
        n_permutations=50,
    )

    t_vals, clusters, cluster_pvals, H0 = cluster_stats

    # Find significant clusters
    good_cluster_inds = np.where(cluster_pvals < 0.05)[0]

    if len(good_cluster_inds) == 0:
        print("No significant clusters found.")
        return None

    # Create a 3D mask for significant clusters
    mask = np.zeros(t_vals.shape, dtype=bool)
    for cluster_ind in good_cluster_inds:
        mask[clusters[cluster_ind]] = True

    # Label connected components in the mask
    labeled_mask, num_clusters = label(mask)

    # Initialize a dictionary to store aggregated timeseries for each cluster
    cluster_timeseries = {}

    for i in range(1, num_clusters + 1):
        cluster_mask = labeled_mask == i
        cluster_data = t_vals[cluster_mask].reshape(-1, t_vals.shape[2])

        # Aggregate timeseries for this cluster (using mean, but you could use median or other methods)
        aggregated_timeseries = np.mean(cluster_data, axis=0)

        # Store the aggregated timeseries
        cluster_timeseries[f"cluster_{i}"] = aggregated_timeseries

    # Create a new STC with one timeseries per cluster
    n_clusters = len(cluster_timeseries)
    aggregated_data = np.zeros((n_clusters, t_vals.shape[2]))
    for i, timeseries in enumerate(cluster_timeseries.values()):
        aggregated_data[i, :] = timeseries

    # Use the first vertex of each hemisphere as placeholder locations
    vertices = [
        src[0]["vertno"][:n_clusters],
        src[1]["vertno"][:0],
    ]  # All in left hemisphere for simplicity

    aggregated_stc = mne.SourceEstimate(
        aggregated_data,
        vertices=vertices,
        tmin=avg_stc.tmin,
        tstep=avg_stc.tstep,
        subject=avg_stc.subject,
    )

    # Save aggregated STC
    aggregated_fname = os.path.join(
        STC_DIR, f"aggregated_clusters_{band}_lcmv_beamformer-stc"
    )
    aggregated_stc.save(aggregated_fname)
    print(f"Saved aggregated SourceEstimate to {aggregated_fname}")

    visualize_clusters(cluster_stats, stcs, src)

    return aggregated_stc, cluster_timeseries


def visualize_clusters(cluster_stats, stcs, src):
    """Visualize cluster results."""
    print("Visualizing clusters.")
    tmin = stcs[0].tmin * 100
    tstep = stcs[0].tstep * 100  # convert to milliseconds

    fsave_vertices = [s["vertno"] for s in src]
    stc_all_cluster_vis = summarize_clusters_stc(
        cluster_stats, tstep=tstep, vertices=fsave_vertices, subject="fsaverage"
    )
    plot_fmax = np.max(stc_all_cluster_vis.data) * 1.2
    brain = stc_all_cluster_vis.plot(
        hemi="both",
        views="lateral",
        brain_kwargs=dict(show=False),
        add_data_kwargs=dict(
            fmin=plot_fmax / 10,
            fmid=plot_fmax / 2,
            fmax=plot_fmax,
            scale_factor=0.0001,
            colorbar_kwargs=dict(label_font_size=10),
        ),
        subjects_dir=SUBJECTS_DIR,
        time_label="temporal extent (ms)",
        size=(800, 800),
    )


def plot_average_stc(avg_stc):
    """Plot the average Source Time Course."""
    plot_fmax = np.max(avg_stc.data) * 0.8
    brain = avg_stc.plot(
        subject="fsaverage",
        hemi="both",
        views="dorsal",
        initial_time=1.2,
        brain_kwargs=dict(show=False),
        add_data_kwargs=dict(
            fmin=plot_fmax / 10,
            fmid=plot_fmax / 2,
            fmax=plot_fmax,
            scale_factor=0.0001,
            colorbar_kwargs=dict(label_font_size=10),
        ),
    )


def main():
    src = mne.read_source_spaces(SRC_FNAME)
    aggregated_stcs = {}
    all_cluster_timeseries = {}
    for band in FREQUENCY_BANDS:
        result = process_frequency_band(band, src)
        if result is not None:
            aggregated_stc, cluster_timeseries = result
            aggregated_stcs[band] = aggregated_stc
            all_cluster_timeseries[band] = cluster_timeseries

    print("Analysis complete")
    return aggregated_stcs, all_cluster_timeseries


if __name__ == "__main__":
    aggregated_stcs, all_cluster_timeseries = main()


if __name__ == "__main__":
    main()
