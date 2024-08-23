# Run traditional source analysis (surface or volume-based) and save the STCs
# Compute and plot GAs
# Extract time courses from predefined ROIs

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import glob
import platform
from pathlib import Path

import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv

import config  # our config.py file


# specify source method etc
source_method = "dSPM" #"MNE"
src_type = 'surface' # 'vol' or 'surface'
spacing = "oct6" # for 'surface' source space only
if src_type != 'surface':
    spacing = ''

# specify how many SSP projectors to use for speech artifact removal
n_proj = 4  # 1 means: 1 for MEG & 1 for EEG
this_run = f"{n_proj}-proj"
#this_run = "ba+da__ba-da" # for the "ba+da" and "ba-da" sanity checks

# which conditions to compare in ROI analysis
comparison = "participant" # "interviewer" or "participant"
conds_ROI = [f"{comparison}_conversation", f"{comparison}_repetition"]

ch_type = None  # "meg" or "eeg"  # None means all
if ch_type:
    path_suffix = f"_{ch_type}-only"
else:
    path_suffix = ""

if src_type == 'vol':
    src_suffix = '-vl.stc'
    #if this_run == "ba+da__ba-da":
    #    src_suffix = '-vol.stc' # to read a previous version of saved results
elif src_type == 'surface':
    src_suffix = '-lh.stc'


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
#data_path = Path("/mnt/d/Work/analysis_ME206/data")
#analysis_path = Path("/mnt/d/Work/analysis_ME206/Natural_Conversations_study/analysis")

deriv_path = analysis_path / "natural-conversations-bids" / "derivatives"
source_results_dir = analysis_path / "results" / f"source-{source_method}-{src_type}{path_suffix}" / this_run
figures_dir = analysis_path / "figures" / f"source-{source_method}-{src_type}{path_suffix}" / this_run
figures_ROI_dir = figures_dir / f"{comparison}_ROI"
#figures_ROI_zscores_dir = figures_dir / f"{comparison}_ROI_zscores"
figures_ROI_zscores_dir = figures_dir / f"{comparison}_ROI_zscores-timeslices"
# create the folders if needed
source_results_dir.mkdir(parents=True, exist_ok=True)
figures_ROI_dir.mkdir(parents=True, exist_ok=True)
figures_ROI_zscores_dir.mkdir(parents=True, exist_ok=True)

subjects_dir = deriv_path / "freesurfer" / "subjects"
subject = 'fsaverage'
if src_type == 'vol':
    src_fname = subjects_dir / subject / "bem" / "fsaverage-vol-5-src.fif"
    bem_fname = subjects_dir / subject / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
else:
    src_fname = subjects_dir / subject / "bem" / "fsaverage-oct6-src.fif"

# adjust mne options to fix rendering issues (only needed in Linux / WSL)
mne.viz.set_3d_options(antialias = False, depth_peeling = False, 
                    smooth_shading = False, multi_samples = 1) 


# loop through the subjects we want to analyse
for sub in use_subjects:
    path = deriv_path / 'mne-bids-pipeline' / f'sub-{sub}' / 'meg'
    epochs_fname = path / f'sub-{sub}_task-conversation_proc-clean_epo.fif'
    trans_fname = path / f'sub-{sub}_task-conversation_trans.fif'
    fwd_fname = path / f'sub-{sub}_task-conversation_fwd.fif'
    cov_fname = path / f'sub-{sub}_task-rest_proc-clean_cov.fif'
    inv_fname = path / f'sub-{sub}_task-conversation_inv.fif'
    proj_fname = path / f'sub-{sub}_task-conversation_proc-proj_proj.fif'

    stc_filename_prefix = op.join(source_results_dir, f'sub-{sub}_task-conversation_proc')

    if Path(stc_filename_prefix + '_ba' + src_suffix).exists():
        continue

    # Read data
    epochs = mne.read_epochs(epochs_fname)

    # Apply SSP projectors for speech artifact removal
    if n_proj:
        all_proj = mne.read_proj(proj_fname)
        proj = list()
        for ii, kind in enumerate(("MEG", "EEG")):
            these_proj = all_proj[10*ii : 10*ii+n_proj]
            proj.extend(these_proj)
        del all_proj
        epochs.add_proj(proj).apply_proj()
    
    epochs = epochs.pick(["meg", "eeg"], exclude="bads")
    if ch_type:
        epochs.pick(ch_type)

    ranks = mne.compute_rank(inst=epochs, tol=1e-3, tol_kind="relative")
    rank = sum(ranks.values())
    print(f"  Ranks={ranks} (total={rank})")

    # TEMP - always recreate inv for now, until we change to free orientation in the pipeline
    if 1: #n_proj or ch_type or src_type == 'vol':
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
                epochs.info, fwd, cov, loose=1., depth=0.8, rank=ranks, # use free orientation, so that the surface- and volume-based results are more comparable
            )
    else: # load the default inv solution (surface source space; no SSP projectors for speech artifact removal)
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        
        
    # compute evoked (averaged over all conditions)
    #evoked_allconds = epochs.average()
    #evoked_allconds.plot_joint() # average ERF across all conds

    epochs.equalize_event_counts(['ba', 'da'])
    epochs.equalize_event_counts(['interviewer_conversation', 'interviewer_repetition'])
    epochs.equalize_event_counts(['participant_conversation', 'participant_repetition'])
    
    # compute evoked for each cond (ba, da, conversation, repetition)
    evokeds = []
    for cond in epochs.event_id:
        evokeds.append(epochs[cond].average())

    # compute source timecourses
    stcs = dict()

    for index, evoked in enumerate(evokeds):
        cond = evoked.comment

        snr = 3.
        lambda2 = 1. / snr ** 2
        stcs[cond], residual = apply_inverse(evoked, inv, lambda2,
                                    method=source_method, pick_ori=None,
                                    return_residual=True, verbose=True)

        # save the source estimates
        stcs[cond].save(stc_filename_prefix + '_' + cond, overwrite=True)

    if this_run == "ba+da__ba-da":
        # compute "ba+da" (i.e. combining the two conditions)
        evoked_ba = epochs['ba'].average()
        evoked_da = epochs['da'].average()
        evoked_combined = mne.combine_evoked([evoked_ba, evoked_da], weights='equal')
        stc_combined, residual = apply_inverse(evoked_combined, inv, lambda2,
                                        method=source_method, pick_ori=None,
                                        return_residual=True, verbose=True)
        stc_combined.save(f'{stc_filename_prefix}_ba+da')

        # compute "ba-da" (i.e. difference between the two conditions)
        evoked_subtracted = mne.combine_evoked([evoked_ba, evoked_da], weights=[1, -1])
        stc_subtracted, residual = apply_inverse(evoked_subtracted, inv, lambda2,
                                        method=source_method, pick_ori=None,
                                        return_residual=True, verbose=True)
        stc_subtracted.save(f'{stc_filename_prefix}_ba-da')

    
    # Plot the source timecourses
    '''
    for index, evoked in enumerate(evokeds):
        cond = evoked.comment

        hemi='both'
        #vertno_max, time_max = stcs[cond].get_peak(hemi=hemi, tmin=0.1, tmax=0.27)
        initial_time = 0.09 #time_max
        surfer_kwargs = dict(
            hemi=hemi, subjects_dir=subjects_dir,
            initial_time=initial_time, 
            time_unit='s', title=subject + ' - ' + cond,
            views=['lateral','medial'], #['caudal','ventral','lateral','medial'], 
            show_traces=False,
            smoothing_steps=10)
        brain = stcs[cond].plot(**surfer_kwargs)
        #brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, 
        #    color='blue', scale_factor=0.6, alpha=0.5)
        brain.save_image(op.join(figures_dir, "indi_subjects", subject + '-' + cond + '.png'))
        #brain.save_movie(op.join(figures_dir, "indi_subjects", subject + '-' + cond + '-both.mov'), 
        #    tmin=0, tmax=0.35, interpolation='linear',
        #    time_dilation=50, time_viewer=True)

    # close all figures before moving onto the next subject
    mne.viz.close_all_3d_figures()
    '''


# Compute the grand average

conds = ['ba','da','interviewer_conversation','interviewer_repetition','participant_conversation','participant_repetition']
if this_run == "ba+da__ba-da":
    conds = ['ba+da', 'ba-da']

# overwrite the paths here to use a previous version of results for participant-locked epochs
#source_results_dir = "/mnt/d/Work/analysis_ME206/results/bids/source/MNE_surface_4proj"
#figures_dir = "/mnt/d/Work/analysis_ME206/results/bids/source/MNE_surface_4proj/Figures_ROI"

GA_stcs = {}
for cond in conds:
    # find all the saved stc results
    stc_files = glob.glob(op.join(source_results_dir, 'sub*' + cond + src_suffix))
    # Note: for surface-based stcs, only need to supply the filename 
    # for one hemisphere (e.g. '-lh.stc'), and it will look for both
    # https://mne.tools/stable/generated/mne.read_source_estimate.html

    # initialise the sum array to correct size using first subject's stc
    stc = mne.read_source_estimate(stc_files[0])
    '''
    # read in the stc for each subsequent subject, add to the sum array
    for fname in stc_files[1:]:
        stc += mne.read_source_estimate(fname)
    # divide by number of files
    stc /= len(stc_files)
    '''
    
    # the above only seems to work for surface-based stcs,
    # so we are retaining the code below for now to handle vol-based stcs

    stcs_sum = stc.data # this contains lh & rh vertices combined together
    # there are also separate fields for the 2 hemis (stc.lh_data, stc.rh_data),
    # but their content cannot be set directly, so just use the combined one

    # read in the stc for each subsequent subject, add to the sum array
    for fname in stc_files[1:]:
        stc = mne.read_source_estimate(fname)
        stcs_sum = stcs_sum + stc.data
    # divide by number of files
    stc.data = stcs_sum / len(stc_files)

    # store in the GA struct
    GA_stcs[cond] = stc

    
    # Plot the GAs
    initial_time = 0.09

    # Depending on the src type, use diff types of plots
    if src_type == 'vol':
        src = mne.read_source_spaces(src_fname)
        fig = stc.plot(src=src, colormap="viridis", 
            subject='fsaverage', subjects_dir=subjects_dir, verbose=True,
            initial_time=initial_time)
        fig.savefig(figures_dir / f'GA_{cond}.png')
        plt.close(fig)
        # seems like we can't save movies for these volume-based stcs

    elif src_type == 'surface':  
        hemi='both'
        #vertno_max, time_max = stc.get_peak(hemi=hemi)
        #initial_time = time_max
        # to get auto clim for a particular time point, crop to a short interval around that time first
        # (otherwise clim will be based on the peak activity over the entire epoch)
        #stc = stc.crop(tmin=initial_time, tmax=initial_time+0.1)
        surfer_kwargs = dict(
            hemi=hemi, subject='fsaverage', subjects_dir=subjects_dir, 
            #clim=dict(kind='value', lims=[8, 12, 15]), 
            initial_time=initial_time, time_unit='s', 
            views=['lateral','medial'], show_traces=False,
            size=(800, 800), smoothing_steps=10)
        brain = stc.plot(**surfer_kwargs)
        #brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
        #            scale_factor=0.6, alpha=0.5)
        brain.save_image(figures_dir / f'GA_{str(initial_time)}s_{cond}.png')
        brain.save_movie(figures_dir / f'GA-{cond}.mov', 
            tmin=-1, tmax=1, interpolation='linear',
            time_dilation=50, time_viewer=True) 
          
    # Note: if there are any issues with the plots/movies (e.g. showing 
    # horizontal bands), it's probably a rendering issue in Linux. 
    # Try running this script in Windows/Mac!

    plt.close('all')


# Extract ROI time courses from source estimates

window_size = 200  # sliding window size (ms) for calculating z-scores
window = int(window_size / 5)  # e.g. 10 = 50ms, 20 = 100ms, 40 = 200ms
            
(figures_ROI_dir / "all_ROIs").mkdir(parents=True, exist_ok=True)
(figures_ROI_zscores_dir / f"all_ROIs_{window_size}ms").mkdir(parents=True, exist_ok=True)

src = mne.read_source_spaces(src_fname)

if src_type == 'vol':
    # choose atlas

    # can use aparc 2009 parcellation
    fname_aseg = subjects_dir / subject / 'mri' / 'aparc.a2009s+aseg.mgz' # aparc = cortical; aseg = subcortical
    label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    rois = ['ctx_lh_G_cingul-Post-dorsal','ctx_lh_G_cingul-Post-ventral','ctx_rh_G_cingul-Post-dorsal','ctx_rh_G_cingul-Post-ventral',
            'ctx_lh_G_pariet_inf-Supramar','ctx_rh_G_pariet_inf-Supramar','ctx_lh_G_precuneus','ctx_rh_G_precuneus',
            'ctx_lh_G_temp_sup-Lateral','ctx_rh_G_temp_sup-Lateral','ctx_lh_Pole_temporal','ctx_rh_Pole_temporal',
            'ctx_lh_S_temporal_sup','ctx_rh_S_temporal_sup']  # can have multiple labels in this list
    #roi_idx = label_names.index(rois[0])

    # or use the HCP-MMP parcellation
    # Note: need to create the volumetric atlas & custom lookup table (LUT) for this first - see link below:
    # https://gist.github.com/larsoner/8e664205cd8285ca7c46211403ad12ce
    fname_aseg = subjects_dir / subject / 'mri' / 'HCPMMP1_combined+aseg.mgz' # HCPMMP1 = cortical; aseg = subcortical
    fname_lut = subjects_dir / subject / 'mri' / 'HCPMMP1_combinedColorLUT.txt'
    lut = mne.read_freesurfer_lut(fname_lut)
    label_names = mne.get_volume_labels_from_aseg(fname_aseg, atlas_ids=lut[0])
    # need to supply a list of dicts here (rather than strings), as we are using a custom LUT
    rois = [{'Anterior_Cingulate_and_Medial_Prefrontal_Cortex-lh': 1001},
            {'Anterior_Cingulate_and_Medial_Prefrontal_Cortex-rh': 2001},
            {'DorsoLateral_Prefrontal_Cortex-lh': 1004}, 
            {'DorsoLateral_Prefrontal_Cortex-rh': 2004},
            {'Posterior_Cingulate_Cortex-lh': 1015}, 
            {'Posterior_Cingulate_Cortex-rh': 2015},
            {'Temporo-Parieto-Occipital_Junction-lh': 1021}, 
            {'Temporo-Parieto-Occipital_Junction-rh': 2021}]       

    for label in rois:
        # check if label is a dict, if so then we extract the label_name here
        if isinstance(label, dict):
            label_name = list(label.keys())[0]
        
        #(figures_ROI_dir / label_name).mkdir(parents=True, exist_ok=True)
        #(figures_ROI_zscores_dir / label_name).mkdir(parents=True, exist_ok=True)

        # Plot GA ROI time series
        fig, axes = plt.subplots(1, layout="constrained")
        fig_z, axes_z = plt.subplots(1, layout="constrained")
        for cond in conds_ROI:
            label_ts = mne.extract_label_time_course(
                [GA_stcs[cond]], (fname_aseg, label), src, mode="auto"
            )
            label_ts = label_ts[0][0]
            axes.plot(1e3 * GA_stcs[cond].times, label_ts, label=cond)

            # calculate z-score at each time point (using a sliding time window)
            mu = np.mean(label_ts) # demean is based on the whole epoch
            label_ts = label_ts - mu
            
            zscores = [0] * (len(label_ts) - window) # initialise the z-scores array
            for t in range(0, len(label_ts) - window):
                sigma = np.std(label_ts[t:t+window], mean=0) # sigma is calculated on the time slice only, but need to manually set the mean to 0
                zscores[t] = label_ts[t] / sigma
            axes_z.plot(1e3 * GA_stcs[cond].times[:len(zscores)], zscores, label=cond)
            '''
            # take the time window centred on current time point (rather than on the RHS)
            window_half = int(window/2)
            zscores = [0] * (len(label_ts) - window_half) # initialise the z-scores array
            for t in range(window_half, len(label_ts) - window_half):
                sigma = np.std(label_ts[t-window_half:t+window_half], mean=0) # sigma is calculated on the time slice only, but need to manually set the mean to 0
                zscores[t] = label_ts[t] / sigma
            axes_z.plot(1e3 * GA_stcs[cond].times[window_half:len(zscores)], zscores[window_half:], label=cond)
            '''

        axes.axvline(linestyle='-', color='k') # add verticle line at time 0
        axes.set(xlabel="Time (ms)", ylabel="Activation")
        axes.legend()
        axes_z.axvline(linestyle='-', color='k') # add verticle line at time 0
        axes_z.set(xlabel="Time (ms)", ylabel="Activation (z-score)")
        axes_z.legend()

        #fig.savefig(figures_ROI_dir / label_name / "GA.png")
        fig.savefig(figures_ROI_dir / "all_ROIs" / f"{label_name}_GA.png") # to save an additional copy of all GA plots into one folder
        fig_z.savefig(figures_ROI_zscores_dir / f"all_ROIs_{window_size}ms" / f"{label_name}_GA.png") 
        plt.close('all')

        # Plot individual-subjects ROI time series
        for sub in use_subjects:
            fig, axes = plt.subplots(1, layout="constrained")
            fig_z, axes_z = plt.subplots(1, layout="constrained")
            for cond in conds_ROI:
                stc_file = op.join(source_results_dir, f'sub-{sub}_task-conversation_proc_{cond}{src_suffix}')    
                stc = mne.read_source_estimate(stc_file)
                label_ts = mne.extract_label_time_course(
                    [stc], (fname_aseg, label_name), src, mode="auto"
                )
                axes.plot(1e3 * stc.times, label_ts[0][0], label=cond)

                # calculate z-score at each time point
                #mu = np.mean(label_ts[0][0])
                #sigma = np.std(label_ts[0][0])
                #zscores = (label_ts - mu) / sigma
                #axes_z.plot(1e3 * stc.times, zscores[0][0], label=cond)

            axes.axvline(linestyle='-', color='k') # add verticle line at time 0
            axes.set(xlabel="Time (ms)", ylabel="Activation")
            axes.legend()
            #axes_z.axvline(linestyle='-', color='k') # add verticle line at time 0
            #axes_z.set(xlabel="Time (ms)", ylabel="Activation (z-score)")
            #axes_z.legend()

            fig.savefig(figures_ROI_dir / label_name / f"sub-{sub}.png")
            #fig_z.savefig(figures_ROI_zscores_dir / label_name / f"sub-{sub}.png")
            plt.close('all')

elif src_type == 'surface':
    # for surface source space, we need to create the Label object first
    # by reading from .annot or .label file
    # Can't use the mri file like above, as extract_label_time_course() will throw an error
    # https://mne.tools/stable/generated/mne.extract_label_time_course.html

    # Get labels for FreeSurfer 'aparc' cortical parcellation (69 labels)
    # https://freesurfer.net/fswiki/CorticalParcellation
    #labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir) # read all labels from annot file
    #labels = [labels_parc[60], labels_parc[61]] # 60: left STG; 61: right STG

    # or use 'aparc.a2009s' parcellation (150 labels)
    labels_parc = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)
    labels = [
        labels_parc[18]+labels_parc[20], labels_parc[19]+labels_parc[21], labels_parc[50], labels_parc[51], 
        labels_parc[58], labels_parc[59], labels_parc[66], labels_parc[67], labels_parc[84], labels_parc[85], labels_parc[144], labels_parc[145]
    ]
    # can check if a particular label is available in the atlas
    #print([label for label in labels_parc if "fusiform" in label.name])

    # or read a single label (e.g. V1, BA44, etc)
    #labels_parc = mne.read_label(op.join(subjects_dir, subject, 'label', 'lh.V1.label'))

    
    # or use the HCP-MMP parcellation
    # https://balsa.wustl.edu/WN56
    #mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir)
    # can use the fine-grained (~360 labels) "HCPMMP1" or coarse (~44 labels) "HCPMMP1_combined" atlas
    labels_parc = mne.read_labels_from_annot(
        'fsaverage', 'HCPMMP1_combined', subjects_dir=subjects_dir)
    labels = [labels_parc[2], labels_parc[3], labels_parc[8], labels_parc[9],
              labels_parc[30], labels_parc[31], labels_parc[42], labels_parc[43]]
    #figures_ROI_dir = figures_ROI_dir.parent / (figures_ROI_dir.name + "_HCPMMP")
    '''
    # to check the locations of ROIs
    brain = mne.viz.Brain('fsaverage', 'lh', 'inflated', subjects_dir=subjects_dir,
                cortex='low_contrast', background='white', size=(800, 600))
    #brain.add_annotation('HCPMMP1') # this adds "border lines" showing the parcellation
    #aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
    #brain.add_label(aud_label, borders=False) # this "colours in" the region
    for label in labels:
        brain.add_label(label)
    '''

    for label in labels:
        label_name = label.name 
        # or set a custom name for combined labels
        if label_name == 'G_cingul-Post-dorsal-lh + G_cingul-Post-ventral-lh':
            label_name = 'G_cingul-Post-lh'
        if label_name == 'G_cingul-Post-dorsal-rh + G_cingul-Post-ventral-rh':
            label_name = 'G_cingul-Post-rh'

        #(figures_ROI_dir / label_name).mkdir(parents=True, exist_ok=True)

        # Plot GA ROI time series
        fig, axes = plt.subplots(1, layout="constrained")
        fig_z, axes_z = plt.subplots(1, layout="constrained")
        for cond in conds_ROI:
            label_ts = mne.extract_label_time_course(
                [GA_stcs[cond]], label, src, mode="auto"
            )
            label_ts = label_ts[0][0]
            axes.plot(1e3 * GA_stcs[cond].times, label_ts, label=cond)

            # calculate z-score at each time point (using a sliding time window)
            mu = np.mean(label_ts) # demean is based on the whole epoch
            label_ts = label_ts - mu
            
            zscores = [0] * (len(label_ts) - window) # initialise the z-scores array
            for t in range(0, len(label_ts) - window):
                sigma = np.std(label_ts[t:t+window], mean=0) # sigma is calculated on the time slice only, but need to manually set the mean to 0
                zscores[t] = label_ts[t] / sigma
            axes_z.plot(1e3 * GA_stcs[cond].times[:len(zscores)], zscores, label=cond)

        axes.axvline(linestyle='-', color='k') # add verticle line at time 0
        axes.set(xlabel="Time (ms)", ylabel="Activation")
        axes.legend()
        axes_z.axvline(linestyle='-', color='k') # add verticle line at time 0
        axes_z.set(xlabel="Time (ms)", ylabel="Activation (z-score)")
        axes_z.legend()

        #fig.savefig(figures_ROI_dir / label_name / "GA.png")
        fig.savefig(figures_ROI_dir / "all_ROIs" / f"{label_name}_GA.png") # to save an additional copy of all GA plots into one folder
        fig_z.savefig(figures_ROI_zscores_dir / f"all_ROIs_{window_size}ms" / f"{label_name}_GA.png") 
        plt.close('all')
        
        # Plot individual-subjects ROI time series
        for sub in use_subjects:
            fig, axes = plt.subplots(1, layout="constrained")    
            for cond in conds_ROI:
                stc_file = op.join(source_results_dir, f'sub-{sub}_task-conversation_proc_{cond}{src_suffix}')    
                stc = mne.read_source_estimate(stc_file)
                label_ts = mne.extract_label_time_course(
                    [stc], label, src, mode="auto"
                )
                axes.plot(1e3 * stc.times, label_ts[0][0, :], label=cond)

            axes.axvline(linestyle='-', color='k') # add verticle line at time 0
            axes.set(xlabel="Time (ms)", ylabel="Activation")
            axes.legend()

            fig.savefig(figures_ROI_dir / label_name / f"sub-{sub}.png")
            plt.close(fig)
