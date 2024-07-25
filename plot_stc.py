import mne
import os

SUBJECTS_DIR = "/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/freesurfer/subjects"
STC_DIR = "/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Test_output_1/"
DATA_FILE = "sub-01_task-da_broadband_lcmv_beamformer_averaged-stc-lh.stc"


# Load the saved STC file (replace with your actual file path)
stc_file = os.path.join(STC_DIR, DATA_FILE)
stc = mne.read_source_estimate(stc_file)

stc.plot(
    subjects_dir=SUBJECTS_DIR,
    subject="fsaverage",
    initial_time=0.1,
    hemi="both",
    views="lateral",
    clim=dict(kind="value", lims=[3, 6, 9]),
)
