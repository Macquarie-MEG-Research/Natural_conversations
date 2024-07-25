import os
import mne
import numpy as np
import h5py
from mne.beamformer import apply_lcmv_epochs, make_lcmv
from mne.cov import compute_covariance
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Configuration options - change as needed saving
# cropped data takes up a lot of space
EQUALIZE_EVENT_COUNTS = False
SAVE_CROPPED_DATA_H5 = False
DIAGNOSTIC_PLOTS = False


def setup_directories():
    data_dir = "/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Test_data/"
    output_dir = "/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Test_output_1"
    os.makedirs(output_dir, exist_ok=True)
    return data_dir, output_dir


def load_data(subject, data_dir):
    fwd_fname = os.path.join(
        data_dir, subject, "meg", f"{subject}_task-conversation_fwd.fif"
    )
    epochs_fname = os.path.join(
        data_dir, subject, "meg", f"{subject}_task-conversation_proc-clean_epo.fif"
    )
    noise_fname = os.path.join(
        data_dir, subject, "meg", f"{subject}_task-rest_proc-clean_raw.fif"
    )

    if not all(os.path.exists(f) for f in [fwd_fname, epochs_fname, noise_fname]):
        return None

    fwd = mne.read_forward_solution(fwd_fname)
    fwd_fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, use_cps=True
    )  # Convert to fixed orientation - necessary for LCMV ori = normal
    all_epochs = mne.read_epochs(epochs_fname)
    noise_raw = mne.io.read_raw_fif(noise_fname)
    return fwd_fixed, all_epochs, noise_raw


def average_stc_in_time(stc, window_size=0.1):

    sfreq = 1 / stc.tstep
    n_samples = int(window_size * sfreq)

    # Calculate the number of full windows
    n_windows = (
        len(stc.times) - 1
    ) // n_samples  # -1 to ensure we don't exceed the time range

    # Prepare the data for averaging
    data_to_average = stc.data[:, : n_windows * n_samples]

    # Reshape and average
    averaged_data = data_to_average.reshape(
        data_to_average.shape[0], n_windows, n_samples
    ).mean(axis=2)

    # Create new time array
    new_times = np.arange(n_windows) * window_size + stc.tmin + (window_size / 2)

    return mne.SourceEstimate(
        averaged_data,
        vertices=stc.vertices,
        tmin=new_times[0],
        tstep=window_size,
        subject=stc.subject,
    )


def is_subject_processed(subject, output_dir):
    # Check if all expected output files exist for the subject
    frequency_bands = ["broadband", "alpha", "beta", "gamma"]
    conditions = [
        "ba",
        "da",
        "interviewer_conversation",
        "interviewer_repetition",
        "participant_conversation",
        "participant_repetition",
    ]

    for band in frequency_bands:
        for condition in conditions:
            roi_fname = f"{subject}_task-{condition}_{band}_lcmv_beamformer_roi_time_courses.npy"
            stc_fname = (
                f"{subject}_task-{condition}_{band}_lcmv_beamformer_averaged-stc-lh.stc"
            )
            if not (
                os.path.exists(os.path.join(output_dir, roi_fname))
                and os.path.exists(os.path.join(output_dir, stc_fname))
            ):
                return False
    return True


def prepare_epochs(all_epochs, noise_raw):
    print(f"Initial all_epochs: {len(all_epochs)}")

    sfreq = all_epochs.info["sfreq"]
    ch_names = all_epochs.ch_names
    tmin, tmax = all_epochs.tmin, all_epochs.tmax
    n_samples = len(all_epochs.times)

    if noise_raw.info["sfreq"] != sfreq:
        noise_raw = noise_raw.resample(sfreq)

    print(f"Task epochs - tmin: {tmin}, tmax: {tmax}, n_samples: {n_samples}")

    noise_events = mne.make_fixed_length_events(noise_raw, duration=tmax - tmin)
    noise_epochs = mne.Epochs(
        noise_raw,
        noise_events,
        event_id={"noise": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
    )

    print(f"Initial noise_epochs: {len(noise_epochs)}")
    print(
        f"Noise epochs - tmin: {noise_epochs.tmin}, tmax: {noise_epochs.tmax}, n_samples: {len(noise_epochs.times)}"
    )

    noise_epochs = noise_epochs.copy().pick(ch_names)

    if len(all_epochs.times) > len(noise_epochs.times):
        print(
            f"Cropping task epochs from {len(all_epochs.times)} to {len(noise_epochs.times)} samples"
        )
        all_epochs = all_epochs.copy().crop(
            tmin=tmin, tmax=tmin + (len(noise_epochs.times) - 1) / sfreq
        )

    # Crop 200 ms from each end
    crop_time = 0.2
    all_epochs = all_epochs.copy().crop(
        tmin=all_epochs.tmin + crop_time, tmax=all_epochs.tmax - crop_time
    )
    noise_epochs = noise_epochs.copy().crop(
        tmin=noise_epochs.tmin + crop_time, tmax=noise_epochs.tmax - crop_time
    )

    print(
        f"After cropping - Task epochs tmin: {all_epochs.tmin}, tmax: {all_epochs.tmax}, n_samples: {len(all_epochs.times)}"
    )
    print(
        f"After cropping - Noise epochs tmin: {noise_epochs.tmin}, tmax: {noise_epochs.tmax}, n_samples: {len(noise_epochs.times)}"
    )

    all_epochs.metadata = None
    noise_epochs.metadata = None

    if not np.allclose(all_epochs.times, noise_epochs.times):
        raise ValueError("Time points still don't match after cropping")

    combined_epochs = mne.concatenate_epochs([all_epochs, noise_epochs])

    print(f"Final combined_epochs: {len(combined_epochs)}")
    print(f"Time points: {len(combined_epochs.times)}")
    print(f"Combined epochs tmin: {combined_epochs.tmin}, tmax: {combined_epochs.tmax}")
    if DIAGNOSTIC_PLOTS:
        # Diagnostic plots
        plt.figure(figsize=(15, 10))

        plt.subplot(311)
        plt.plot(combined_epochs.times, combined_epochs.get_data().mean(axis=(0, 1)))
        plt.title("Average of all channels and epochs")
        plt.xlabel("Time (s)")

        plt.subplot(312)
        plt.plot(combined_epochs.times, combined_epochs.get_data()[0, 0, :])
        plt.title("First channel of first epoch")
        plt.xlabel("Time (s)")

        plt.subplot(313)
        plt.imshow(
            combined_epochs.get_data().mean(axis=0),
            aspect="auto",
            extent=[
                combined_epochs.times[0],
                combined_epochs.times[-1],
                0,
                combined_epochs.get_data().shape[1],
            ],
        )
        plt.title("Heatmap of all channels (averaged across epochs)")
        plt.xlabel("Time (s)")
        plt.ylabel("Channels")

        plt.tight_layout()
        plt.show()

    if EQUALIZE_EVENT_COUNTS:
        combined_epochs.equalize_event_counts()
    return combined_epochs


def filter_data(epochs, fmin, fmax):
    """Filters epochs in the specified frequency band."""
    return epochs.copy().filter(fmin, fmax)


def create_lcmv_filter(epochs_filt, fwd, noise_epochs_filt):
    data_cov = compute_covariance(epochs_filt, tmin=None, tmax=None, method="empirical")
    noise_cov = compute_covariance(
        noise_epochs_filt, tmin=None, tmax=None, method="empirical"
    )
    filters_lcmv = make_lcmv(
        epochs_filt.info,
        fwd,
        data_cov=data_cov,
        noise_cov=noise_cov,
        reg=0.05,
        pick_ori="normal",
    )
    return filters_lcmv


def apply_lcmv_filter(epochs_filt, filters_lcmv):
    return apply_lcmv_epochs(epochs_filt, filters_lcmv, return_generator=True)


def stc_to_matrix(stc, parcellation):
    """Parcellate a SourceEstimate and return a matrix of ROI time courses."""
    roi_time_courses = [
        np.mean(stc.in_label(label).data, axis=0) for label in parcellation
    ]
    return np.array(roi_time_courses)


def load_parcellation():
    subjects_dir = os.environ.get(
        "SUBJECTS_DIR",
        "/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/freesurfer/subjects",
    )
    return mne.read_labels_from_annot(
        "fsaverage", parc="HCPMMP1", subjects_dir=subjects_dir, hemi="both"
    )


def compute_source_estimate(
    epochs_stcs, fwd, epochs, subject, output_dir, condition, band_name, parcellation
):
    """Computes source estimates, ROI time courses, and saves them.

    Args:
        epochs_stcs (list): List of SourceEstimate objects.
        fwd (dict): Forward solution dictionary.
        epochs (mne.Epochs): Evoked data object.
        subject (str): Subject name.
        output_dir (str): Output directory path.
        condition (str): Experimental condition.
        band_name (str): Band name.
        parcellation (mne.Parcellation): Parcellation object.

    Returns:
        tuple: A tuple containing the following elements:
            - averaged_stc (mne.SourceEstimate): The averaged source estimate.
            - roi_time_courses (np.ndarray): Original ROI time courses.
            - time_averaged_stc (mne.SourceEstimate): Time-averaged source estimate.
            - time_averaged_roi_time_courses (np.ndarray): Time-averaged ROI time courses.
    """

    n_sources = fwd["nsource"]
    n_times = len(epochs.times)
    averaged_data = np.zeros((n_sources, n_times), dtype=complex)
    all_data = []
    n_epochs = 0

    vertices_lh, vertices_rh = fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]

    with h5py.File(
        os.path.join(
            output_dir, f"{subject}_task-{condition}_{band_name}_epochs_stcs.h5"
        ),
        "w",
    ) as h5f:
        for i, stc in enumerate(epochs_stcs):
            analytic_signal = hilbert(stc.data, axis=1)
            averaged_data += analytic_signal
            all_data.append(analytic_signal)
            n_epochs += 1

            h5f.create_dataset(f"epoch_{i}", data=stc.data)

        h5f.attrs["subject"] = subject
        h5f.attrs["condition"] = condition
        h5f.attrs["band_name"] = band_name
        h5f.attrs["n_epochs"] = n_epochs

    if n_epochs == 0:
        raise ValueError("No epochs were processed")

    averaged_data /= n_epochs
    envelope = np.abs(averaged_data)

    averaged_stc = mne.SourceEstimate(
        envelope,
        vertices=[vertices_lh, vertices_rh],
        tmin=epochs.times[0],
        tstep=epochs.times[1] - epochs.times[0],
        subject="fsaverage",
    )

    time_averaged_stc = average_stc_in_time(averaged_stc, window_size=0.1)

    roi_time_courses = stc_to_matrix(averaged_stc, parcellation)
    time_averaged_roi_time_courses = stc_to_matrix(time_averaged_stc, parcellation)

    # Save outputs
    roi_fname = (
        f"{subject}_task-{condition}_{band_name}_lcmv_beamformer_roi_time_courses.npy"
    )
    np.save(os.path.join(output_dir, roi_fname), roi_time_courses)

    time_avg_roi_fname = f"{subject}_task-{condition}_{band_name}_lcmv_beamformer_time_averaged_roi_time_courses.npy"
    np.save(
        os.path.join(output_dir, time_avg_roi_fname), time_averaged_roi_time_courses
    )

    averaged_fname = (
        f"{subject}_task-{condition}_{band_name}_lcmv_beamformer_averaged-stc"
    )
    averaged_stc.save(os.path.join(output_dir, averaged_fname), overwrite=True)

    time_avg_fname = (
        f"{subject}_task-{condition}_{band_name}_lcmv_beamformer_time_averaged-stc"
    )
    time_averaged_stc.save(os.path.join(output_dir, time_avg_fname), overwrite=True)

    print(
        f"Saved original SourceEstimate to {os.path.join(output_dir, averaged_fname)}"
    )
    print(
        f"Saved time-averaged SourceEstimate to {os.path.join(output_dir, time_avg_fname)}"
    )
    print(f"Saved original ROI time courses to {os.path.join(output_dir, roi_fname)}")
    print(
        f"Saved time-averaged ROI time courses to {os.path.join(output_dir, time_avg_roi_fname)}"
    )

    return (
        averaged_stc,
        roi_time_courses,
        time_averaged_stc,
        time_averaged_roi_time_courses,
    )


def process_subject(subject, data_dir, output_dir):
    try:
        fwd, all_epochs, noise_raw = load_data(subject, data_dir)
        if fwd is None:
            print(f"Skipping {subject}: Missing required files")
            return

        print(f"All epochs info:")
        print(f"Number of epochs: {len(all_epochs)}")
        print(f"Time points: {len(all_epochs.times)}")
        print(f"tmin: {all_epochs.tmin}, tmax: {all_epochs.tmax}")

        combined_epochs = prepare_epochs(all_epochs, noise_raw)
        parcellation = load_parcellation()

        frequency_bands = {
            "broadband": (1, 40),
            "alpha": (8, 12),
            "beta": (13, 30),
            "gamma": (30, 40),
        }

        conditions = [
            "ba",
            "da",
            "interviewer_conversation",
            "interviewer_repetition",
            "participant_conversation",
            "participant_repetition",
        ]  # TODO see below re: removing localisers

        for band_name, (fmin, fmax) in frequency_bands.items():
            print(f"Processing {band_name} band ({fmin}-{fmax} Hz)")
            filtered_epochs = filter_data(combined_epochs, fmin, fmax)

            # Create a common filter using all conditions
            all_condition_epochs = mne.concatenate_epochs(
                [filtered_epochs[cond] for cond in conditions]
            )
            noise_epochs = filtered_epochs["noise"]
            common_filter = create_lcmv_filter(
                all_condition_epochs, fwd, noise_epochs
            )  # TODO don't include localisers? Mayby do a separate run for each type so that code doesn't become too complex?

            averaged_stc_dict = {}
            roi_time_courses_dict = {}
            time_averaged_stc_dict = {}
            time_averaged_roi_time_courses_dict = {}

            for condition in conditions:
                condition_epochs = filtered_epochs[condition]

                # Apply the common filter to each condition
                epochs_stcs = apply_lcmv_filter(condition_epochs, common_filter)

                (
                    averaged_stc,
                    roi_time_courses,
                    time_averaged_stc,
                    time_averaged_roi_time_courses,
                ) = compute_source_estimate(
                    epochs_stcs,
                    fwd,
                    condition_epochs,
                    subject,
                    output_dir,
                    condition,
                    band_name,
                    parcellation,
                )

                averaged_stc_dict[condition] = averaged_stc
                roi_time_courses_dict[condition] = roi_time_courses
                time_averaged_stc_dict[condition] = time_averaged_stc
                time_averaged_roi_time_courses_dict[condition] = (
                    time_averaged_roi_time_courses
                )

            diff_pairs = [
                ("interviewer_conversation", "interviewer_repetition"),
                ("participant_conversation", "participant_repetition"),
            ]

            for cond1, cond2 in diff_pairs:
                # Original difference
                diff_stc = averaged_stc_dict[cond1] - averaged_stc_dict[cond2]
                diff_fname = os.path.join(
                    output_dir,
                    f"{subject}_{band_name}_lcmv_beamformer_{cond1}_vs_{cond2}_difference-stc",
                )
                diff_stc.save(diff_fname, overwrite=True)

                # Time-averaged difference
                time_avg_diff_stc = (
                    time_averaged_stc_dict[cond1] - time_averaged_stc_dict[cond2]
                )
                time_avg_diff_fname = os.path.join(
                    output_dir,
                    f"{subject}_{band_name}_lcmv_beamformer_{cond1}_vs_{cond2}_time_averaged_difference-stc",
                )
                time_avg_diff_stc.save(time_avg_diff_fname, overwrite=True)

        print(f"Finished processing {subject}")
        update_progress(subject, output_dir)
    except Exception as e:
        print(f"Error processing {subject}: {str(e)}")
        import traceback

        traceback.print_exc()


def update_progress(subject, output_dir):
    progress_file = os.path.join(output_dir, "progress.txt")
    with open(progress_file, "a") as f:
        f.write(f"{subject}\n")


def get_processed_subjects(output_dir):
    progress_file = os.path.join(output_dir, "progress.txt")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(line.strip() for line in f)
    return set()


def main():
    data_dir, output_dir = setup_directories()
    subject_dirs = [
        d for d in os.listdir(data_dir) if d.startswith("sub-") and d[4:].isdigit()
    ]

    processed_subjects = get_processed_subjects(output_dir)

    for subject in subject_dirs:
        if subject in processed_subjects or is_subject_processed(subject, output_dir):
            print(f"Skipping {subject}: Already processed")
            continue
        process_subject(subject, data_dir, output_dir)
        # Removed: update_progress(subject, output_dir)

    print("All subjects processed")


if __name__ == "__main__":
    main()
