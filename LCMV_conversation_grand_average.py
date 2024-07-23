import os
import mne
import numpy as np
from glob import glob
import re

# Set the path to the folder containing the .stc files
stc_folder = "/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/LCMV_analysis"

# Create a subfolder for average STCs
average_folder = os.path.join(stc_folder, "average_stc")
os.makedirs(average_folder, exist_ok=True)


# Function to extract subject number and condition from filename
def parse_filename(filename):
    # Extract subject number
    match = re.search(r"sub-(\d+)", filename)
    if match:
        subject_number = int(match.group(1))
    else:
        raise ValueError(f"Cannot extract subject number from filename: {filename}")

    # Extract condition (everything after 'sub-XX_' and before '.stc')
    condition = filename.split("sub-")[1].split(".stc")[0].split("_", 1)[1]

    return subject_number, condition


# Get all .stc files
all_files = glob(os.path.join(stc_folder, "sub-*_*.stc"))

# Extract all unique conditions
conditions = list(set([parse_filename(f)[1] for f in all_files]))

# Process each condition
for condition in conditions:
    print(f"Processing condition: {condition}")

    # Get all .stc files for this condition
    condition_files = [f for f in all_files if parse_filename(f)[1] == condition]

    # Sort files by subject number
    condition_files.sort(key=lambda f: parse_filename(f)[0])

    # Read all STCs for this condition
    stcs = [mne.read_source_estimate(f) for f in condition_files]

    # Check if all STCs have the same number of time points
    n_times = [stc.data.shape[1] for stc in stcs]
    if len(set(n_times)) > 1:
        raise ValueError("Not all STCs have the same number of time points")

    # Stack the data from all subjects
    data = np.stack([stc.data for stc in stcs])

    # Calculate the average across subjects
    avg_data = np.mean(data, axis=0)

    # Create a new STC with the averaged data
    avg_stc = mne.SourceEstimate(
        data=avg_data,
        vertices=stcs[0].vertices,
        tmin=stcs[0].tmin,
        tstep=stcs[0].tstep,
        subject=stcs[0].subject,
    )

    # Save the averaged STC in the new subfolder
    avg_stc.save(os.path.join(average_folder, f"avg_{condition}"), overwrite=True)

print(
    "Averaging complete for all conditions. Results saved in 'average_stc' subfolder."
)
