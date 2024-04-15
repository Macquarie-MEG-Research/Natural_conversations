"""MNE-BIDS-Pipeline configuration for natural-conversations-bids.

Run with:

    $ mne_bids_pipeline ./config.py

For options, see https://mne.tools/mne-bids-pipeline/stable/settings/general.html.
"""

from pathlib import Path

study_name = "natural-conversations"
bids_root = (
    Path(__file__).parent / ".." / "Natural_Conversations_study" / "analysis"
    / f'{study_name}-bids'
).resolve()
interactive = False
sessions = "all"
task = "conversation"
subjects = "all"  # ["01"]
# too many drops during conversation blocks (all epochs!)
exclude_subjects = ["20"]
runs = ["01", "02", "03", "04", "05", "06"]

ch_types = ["meg", "eeg"]
data_type = "meg"
eeg_reference = "average"

l_freq = 0.5
h_freq = 45.0
h_trans_bandwidth = 5
epochs_decim = 5
process_rest = True

regress_artifact = dict(
    picks="meg",
    picks_artifact=["MISC 001", "MISC 002", "MISC 003"],
)

spatial_filter = 'ssp'
n_proj_ecg = dict(n_mag=2, n_eeg=0)
n_proj_eog = dict(n_mag=1, n_eeg=1)

# Epoching
reject = {'eeg': 150e-6, 'mag': 5000e-15}
conditions = ["ba", "da", "conversation", "repetition"]
epochs_tmin = -1.5
epochs_tmax = 0.5
baseline = None

# Decoding
contrasts = [("conversation", "repetition")]
decoding_csp = True
decoding_csp_times = [-1.5, -1, -0.5, 0, 0.5]  # before and after
decoding_csp_freqs = {
    'theta': [4, 7],
    'alpha': [8, 13],
    'beta': [14, 30],
    'gamma': [31, 45],
}

# TFRs
time_frequency_freq_min = 1
time_frequency_freq_max = 50
time_frequency_baseline = (-1., 1.)
time_frequency_baseline_mode = "logratio"

# Source estimation
run_source_estimation = True
subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
use_template_mri = "fsaverage"
adjust_coreg = True  # use head-coord fiducials to align with fsaverage MRI
spacing = "oct6"
inverse_method = "dSPM"
noise_cov = "rest"  # rather than the baseline, which will have auditory in it
