"""BIDSify the Natural Conversations dataset.

https://mqoutlook.sharepoint.com/sites/Natural_Conversations/

Done
----
- File sanity/existence check
- Compute dev_head_t using _ini.mrk (no checking of others!)
- Check for acceptable drift rate
- Check event numbers in localiser
- Add EEG sensor positions
- Align MEG and EEG and concatenate channels
- Check jitter of triggers and auditory events on MEG for localiser
- Incorporate fixed/improved auditory localiser trigger timing
- Add reference regression to MNE-BIDS-Pipeline
- Incorporate Yifan's bad channels
- Incorporate Judy's manual -trans.fif coregistration
- Incorporate Judy's bad marker removal
- Judy add Yifan's annotations for speaking and listening using (-1, 1) sec segments
- CSP decoding
- Eyeball subject drop logs for bad channels

Todo
----
- Fix band-pass upper bound for gamma (45 or 49)
- Add end-of turn annotations as well
- Compare autoreject, LOF, and EEG-find-bad-channels-maxwell
- Run STRF-type analysis on M/EEG using auditory
- Anonymize for eventual sharing
- Improve BIDS descriptions and authors
- Yifan run scripts end-to-end on cloud infra to make sure everything reproduces
- Set tasks properly in BIDS formatting and update MNE-BIDS-Pipeline to handle it
"""  # noqa: E501

import copy
import glob
from pathlib import Path

import mne
import mne_bids
from mne.io.constants import FIFF
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import utils  # local module

n_bas_meg = dict(G03=99, G18=99, G19=98)
n_das_meg = dict(G15=99, G18=98, G19=99)
n_bas_eeg = dict(G03=99, G18=99, G19=98)
n_das_eeg = dict(G15=99, G18=98, G19=99)
drift_tols = dict(G04=0.08, G09=0.026, G15=0.024)
bad_envelope_subjects = ("G04", "G08", "G11", "G13", "G22")  # naive envelope fails
empty_map = dict(G02="G01", G05="G06", G13="G01", G18="G17", G20="G19")
always_bad = ["MEG 043"]  # bad for every subject

eeg_map = """
Fp1 AF3 AF7 Fz F1 F3 F5 F7 FC1 FC3
FC5 FT7 Cz C1 C3 C5 T7 CP1 CP3 CP5
TP7 TP9 Pz P1 P3 P5 P7 PO3 PO7 Oz
O1 ECG Fpz Fp2 AF4 AF8 F2 F4 F6 F8
FC2 FC4 FC6 FT8 C2 C4 C6 T8 CPz CP2
CP4 CP6 TP8 TP10 P2 P4 P6 P8 POz PO4
PO8 O2 EOG
""".strip().split()
assert len(eeg_map) == 63  # omit ref, which is at FCz
eeg_renames = {str(ci): ch for ci, ch in enumerate(eeg_map, 1)}
ch_types_map = dict(ECG="ecg", EOG="eog")
min_dur = 0.002  # for events

subjects = tuple(f"G{s:02d}" for s in range(1, 33))
manual_coreg = True  # use Judy's manual coregistration -trans.fif files to adjust coreg
blocks = dict(  # to BIDS task and run
    B1=("conversation", "01"),
    B2=("conversation", "02"),
    B3=("conversation", "03"),
    B4=("conversation", "04"),
    B5=("conversation", "05"),
    # TODO: Revert the wrong task name here once MBP supports multiple tasks
    localiser=("conversation", "06"),  # ("localizer", "01"),
    resting=("rest", None),
)
assert "empty" not in blocks
event_id = dict(ba=1, da=2, conversation=3, repetition=4)

bad_coils = {
    "G01": [0],
    "G02": [2],
    "G06": [0],
    "G08": [0],
    "G13": [3],
    "G18": [4],
    "G24": [3],
    "G28": [3],
    "G29": [1],
    "G32": [1],
}

# BIDS stuff
name = "natural-conversations"
datatype = suffix = "meg"

share_root = Path(__file__).parents[1] / "Natural_Conversations_study"
analysis_root = share_root / "analysis"
data_root = share_root / "data"
del share_root
bids_root = analysis_root / f"{name}-bids"
#analysis_root = Path("/mnt/d/Work/analysis_ME206/Natural_Conversations_study/analysis")
#data_root = Path("/mnt/d/Work/analysis_ME206/data")
#bids_root = Path("/mnt/e/M3/Natural_Conversations_study/analysis") / f"{name}-bids"

bids_root.mkdir(exist_ok=True)
mne_bids.make_dataset_description(
    path=bids_root,
    name=name,
    authors=["Paul Sowman", "Judy Zhu", "Eric Larson"],  # noqa: E501
    how_to_acknowledge='If you use this data, please cite the references provided in this dataset description.',  # noqa: E501
    data_license='CC-BY-SA',
    ethics_approvals=['Human Research Ethics at the Macquarie University'],
    references_and_links=[],
    overwrite=True,
)
(bids_root / 'README').write_text("""
Listening and speaking M/EEG dataset
====================================

This dataset contains M/EEG data.

The data were converted with MNE-BIDS:

- Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896
- Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data structure extended to magnetoencephalography. Scient ic Data, 5, 180110. https://doi.org/10.1038/sdata.2018.110
""".strip())
subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)  # put fsaverage in there
fs_fids, cf = mne.coreg.read_fiducials(
    subjects_dir / "fsaverage" / "bem" / "fsaverage-fiducials.fif"
)
assert cf == FIFF.FIFFV_COORD_MRI
del cf


def get_participant_turns(*, subject, block):
    """Get participant turn annotations."""
    min_event_duration = 1 # seconds
    # note: segments in repetition block are shorter (need to use 1s to catch most of them)
    turns_path = analysis_root / "audios_metadata_labelled" / f"{subject}_{block}.csv"
    assert turns_path.is_file(), turns_path
    df = pd.read_csv(turns_path)
    participant_turns = []
    for _, row in df.iterrows():
        if row['person'] == 'participant' \
            and (row['end']-row['start']) > min_event_duration \
            and bool(row['is_full_turn']):
                participant_turns.append([row['start'],row['end']])
    # Turn into onset/duration/description (annotation)
    assert len(block) == 2 and block[0] == "B", block
    block_num = int(block[1])
    assert block_num in range(1, 6)
    ttype = 'conversation' if block_num % 2 else 'repetition'
    print(f"    {len(participant_turns):2d} {ttype} turns")
    onset = []
    duration = []
    for turn in participant_turns:
        onset.append(turn[0])
        duration.append(turn[1] - turn[0])
    description = [ttype] * len(onset)
    assert len(onset) > 5, len(onset)
    return onset, duration, description

def get_interviewer_turns(*, subject, block):
    """Get interviewer turn annotations."""
    min_event_duration = 1 # seconds
    # note: segments in repetition block are shorter (need to use 1s to catch most of them)
    turns_path = analysis_root / "audios_metadata_labelled" / f"{subject}_{block}.csv"
    assert turns_path.is_file(), turns_path
    df = pd.read_csv(turns_path)
    interviewer_turns = []
    for _, row in df.iterrows():
        if row['person'] == 'interviewer' \
            and (row['end']-row['start']) > min_event_duration \
            and bool(row['is_full_turn']):
                interviewer_turns.append([row['start'],row['end']])
    # Turn into onset/duration/description (annotation)
    assert len(block) == 2 and block[0] == "B", block
    block_num = int(block[1])
    assert block_num in range(1, 6)
    ttype = 'conversation' if block_num % 2 else 'repetition'
    print(f"    {len(interviewer_turns):2d} {ttype} turns")
    onset = []
    duration = []
    for turn in interviewer_turns:
        onset.append(turn[0])
        duration.append(turn[1] - turn[0])
    description = [ttype] * len(onset)
    assert len(onset) > 5, len(onset)
    return onset, duration, description


# Load bad channels
bads = {
    ch_type: pd.read_csv(
        analysis_root / "bads" / f"{ch_type}_bad_channels_eyeballing.csv"
    )
    for ch_type in ("meg", "eeg")
}
for subject in subjects:
    print(f"Subject {subject}...")
    # Set up -trans.fif: convert fs fiducials to subject's coord frame
    trans = analysis_root / "coreg" / f"{subject}-trans.fif"
    assert trans.is_file()
    trans = mne.transforms.invert_transform(mne.read_trans(trans))
    assert trans["from"] == FIFF.FIFFV_COORD_MRI
    assert trans["to"] == FIFF.FIFFV_COORD_HEAD
    subj_fids = copy.deepcopy(fs_fids)
    for fid in subj_fids:
        fid["coord_frame"] = FIFF.FIFFV_COORD_HEAD
        fid["r"] = mne.transforms.apply_trans(trans, fid["r"])
    # Set bad channels to be the same for all runs
    subj_bads = None
    for block in blocks:
        print(f"  Block {block}...")
        # MEG
        subject_root = data_root / f"{subject}"
        fnames = dict(
            input_fname=subject_root / "meg" / f"*_{block}.con",
            mrk=subject_root / "meg" / "*_ini.mrk",
            elp=subject_root / "meg" / "*.elp",
            hsp=subject_root / "meg" / "*.hsp",
        )
        for key, pattern in fnames.items():
            fname = glob.glob(str(pattern))
            assert len(fname) == 1, (key, pattern, fname)
            fnames[key] = fname[0]
        # Until https://github.com/mne-tools/mne-python/pull/12394 lands
        if subject in bad_coils:
            if block == list(blocks)[0]:
                print(f"    Removing {len(bad_coils[subject])} bad coil(s) ...")
            mrk = mne.io.kit.read_mrk(fnames["mrk"])
            elp = mne.io.kit.coreg._read_dig_kit(fnames["elp"])
            fnames["mrk"] = np.delete(mrk, bad_coils[subject], 0)
            fnames["elp"] = np.delete(elp, np.array(bad_coils[subject]) + 3, 0)
        raw_meg = mne.io.read_raw_kit(
            **fnames,
            stim=[166, *range(176, 190)],
            slope="+",
            stim_code="channel",
            stimthresh=2,
        ).load_data()
        assert not np.allclose(raw_meg.info["dev_head_t"]["trans"], np.eye(4))
        assert raw_meg.first_time == 0

        # Adjust fiducials to make fiducial-based coregistration work
        if manual_coreg:
            for fid, fid_new in zip(raw_meg.info["dig"], subj_fids):
                for key in ("kind", "coord_frame", "ident"):
                    assert fid[key] == fid_new[key], (key, fid[key], fid_new[key])
                fid["r"][:] = fid_new["r"]

        # EEG
        eeg_fname = glob.glob(str(subject_root / "eeg" / f"*{block}_new.vhdr"))
        if len(eeg_fname) == 0:
            eeg_fname = glob.glob(str(subject_root / "eeg" / f"*{block}.vhdr"))
        assert len(eeg_fname) == 1
        eeg_fname = eeg_fname[0]
        raw_eeg = mne.io.read_raw_brainvision(eeg_fname).load_data()
        raw_eeg.rename_channels(eeg_renames)
        raw_eeg.set_channel_types(ch_types_map)
        assert raw_eeg.first_time == 0
        assert raw_eeg.info["sfreq"] == raw_meg.info["sfreq"]
        mne.add_reference_channels(raw_eeg, "FCz", copy=False)
        assert raw_eeg.ch_names[-1] == "FCz"
        raw_eeg.set_montage("standard_1020")
        raw_eeg.set_eeg_reference("average")
        # fix some accounting that MNE-Python should probably take care of for us
        with raw_eeg.info._unlock():
            # for ch in raw_eeg.info["chs"]:
            #     ch["loc"][3:6] = raw_eeg.info["chs"][-1]["loc"][3:6]
            raw_eeg.info["custom_ref_applied"] = 0
        raw_eeg.set_eeg_reference(projection=True)
        if block.startswith("B"):
            trig_offset = int(block[1]) - 1
        else:
            trig_offset = 0
        meg_event = mne.find_events(
            raw_meg,
            stim_channel=raw_meg.ch_names[181 + trig_offset],
            min_duration=min_dur,
        )[:1]
        eeg_event = mne.events_from_annotations(
            raw_eeg,
            event_id={f"Stimulus/S {53 + trig_offset}": 1},
        )[0][:1]
        meg_samp = meg_event[0, 0]
        eeg_samp = eeg_event[0, 0]
        # Instead of cropping MEG, let's just zero-order hold the first or last EEG
        # sample. This will make timing of events align with the original MEG
        # data.
        if eeg_samp < meg_samp:
            n_pad = meg_samp - eeg_samp
            raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
            assert len(raw_eeg_pad.times) == n_pad
            raw_eeg_pad._data[:] = raw_eeg[:, 0][0]
            raw_eeg_pad.set_annotations(None)
            raw_eeg = mne.concatenate_raws([raw_eeg_pad, raw_eeg])
            del raw_eeg_pad
        elif eeg_samp > meg_samp:
            raw_eeg.crop((eeg_samp - meg_samp) / raw_eeg.info["sfreq"], None)
        if len(raw_eeg.times) < len(raw_meg.times):
            n_pad = len(raw_meg.times) - len(raw_eeg.times)
            raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
            assert len(raw_eeg_pad.times) == n_pad
            raw_eeg_pad._data[:] = raw_eeg[:, -1][0]
            raw_eeg_pad.set_annotations(None)
            raw_eeg = mne.concatenate_raws([raw_eeg, raw_eeg_pad])
            del raw_eeg_pad
        elif len(raw_eeg.times) > len(raw_meg.times):
            raw_eeg.crop(0, (len(raw_meg.times) - 1) / raw_eeg.info["sfreq"])
        extra_idx = np.where([d["kind"] == 4 for d in raw_meg.info["dig"]])[0][0]
        for di, d in enumerate(raw_eeg.info["dig"][3:]):  # omit fiducials
            raw_meg.info["dig"].insert(extra_idx + di, d)
        raw_eeg.info["dig"][:] = raw_meg.info["dig"]
        for key in ("dev_head_t", "description"):
            raw_eeg.info[key] = raw_meg.info[key]
        with raw_eeg.info._unlock():
            for key in ("highpass", "lowpass"):
                raw_eeg.info[key] = raw_meg.info[key]
        raw_meg.add_channels([raw_eeg])

        if block == "localiser":
            # Figure out our first-order coefficient
            mba = mne.find_events(raw_meg, stim_channel=raw_meg.ch_names[181], min_duration=min_dur)
            mba[:, 2] = 1
            mda = mne.find_events(raw_meg, stim_channel=raw_meg.ch_names[182], min_duration=min_dur)
            mda[:, 2] = 2
            m_ev = np.concatenate([mba, mda])
            m_ev = m_ev[np.argsort(m_ev[:, 0])]
            eba = mne.events_from_annotations(raw_eeg, event_id={"Stimulus/S 53": 1})[0]
            eda = mne.events_from_annotations(raw_eeg, event_id={"Stimulus/S 54": 2})[0]
            e_ev = np.concatenate([eba, eda])
            e_ev = e_ev[np.argsort(e_ev[:, 0])]
            # if subject == "G18":
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(figsize=(6, 1.5), layout="constrained")
            #     ax.plot(raw_meg.times, raw_meg[182][0][0], zorder=4)
            #     ax.set(xlabel="Time (s)", ylabel="ch182")
            #     pe = eda[:, 0] / raw_eeg.info["sfreq"] + (mda[0, 0] / raw_meg.info["sfreq"] - eda[0, 0] / raw_eeg.info["sfreq"])
            #     for val in pe:
            #         ax.axvline(val, color='r', lw=0.5, zorder=3)
            assert len(mba) == n_bas_meg.get(subject, 100), len(mba)
            assert len(mda) == n_das_meg.get(subject, 100), len(mda)
            assert len(eba) == n_bas_eeg.get(subject, 100), len(eba)
            assert len(eda) == n_das_eeg.get(subject, 100), len(eda)
            #assert len(mba) > 95
            #assert len(mda) > 95
            #assert len(eba) > 95
            #assert len(eda) > 95
            e_ev = e_ev[:len(m_ev)]
            np.testing.assert_array_equal(m_ev[:, 2], e_ev[:, 2])
            m = m_ev[:, 0] / raw_meg.info["sfreq"] - raw_meg.first_time
            e = e_ev[:, 0] / raw_eeg.info["sfreq"] - raw_eeg.first_time
            p = np.polyfit(m, e, 1)
            first_ord = p[0]
            off = e - first_ord * m - p[1]
            np.testing.assert_allclose(off, 0., atol=2e-3)
            print(f"  Drift rate: {(1 - first_ord) * 1e6:+0.1f} PPM")
            # Correct jitter
            events, _ = utils.triggerCorrection(raw_meg, subject, plot=False)
            events[:, 2] -= 180  # 181, 182 -> 1, 2
            assert np.in1d(events[:, 2], (1, 2)).all()
            max_off = cdist(m_ev[:, :1], events[:, :1]).min(axis=0).max()
            assert max_off < 50, max_off
        elif block == "resting":
            events = None
        else:
            assert block in ("B1", "B2", "B3", "B4", "B5"), block
            onset, duration, description = get_participant_turns(
                subject=subject, block=block,
            )
            #onset, duration, description = get_interviewer_turns(
            #    subject=subject, block=block,
            #)
            raw_meg.set_annotations(mne.Annotations(onset, duration, description))
            events = None
        # Add bads
        assert raw_meg.info["bads"] == []
        if subj_bads is None:  # first block
            subj_bads = list()
            for ch_type in ("meg", "eeg"):
                these_bads = bads[ch_type].query(f"subject == '{subject}'")  # task == '{block}'
                these_bads = these_bads["bad_channels"].values
                for block_bads in these_bads:
                    assert isinstance(block_bads, str), type(block_bads)
                    block_bads = np.array(eval(block_bads))
                    assert len(block_bads) and block_bads.dtype == int
                    if ch_type == "eeg":
                        which_names = raw_eeg.ch_names
                    else:
                        assert ch_type == "meg"
                        which_names = raw_meg.ch_names
                    subj_bads += [which_names[pick] for pick in block_bads]
                    del block_bads
                del these_bads
            subj_bads += always_bad
            subj_bads = sorted(set(subj_bads), key=lambda x: raw_meg.ch_names.index(x))
            print(f"    Bad channels ({len(subj_bads)}): {subj_bads}")

        # All the same bads
        assert raw_meg.info["bads"] == []
        raw_meg.info["bads"] = subj_bads

        # Empty room data (if present)
        empty_room = empty_map.get(subject, subject)
        if empty_room is not None:
            empty_room = glob.glob(str(data_root / empty_room / "meg" / "*_empty.con"))
            assert len(empty_room) == 1, empty_room
            empty_room = mne.io.read_raw_kit(empty_room[0]).load_data()
            empty_room.info["dev_head_t"] = None
            assert empty_room.info["bads"] == []
            # need to skip EEG
            empty_room.info["bads"] = [
                ch_name for ch_name in subj_bads if ch_name in empty_room.ch_names
            ]
        # raise RuntimeError(str(subj_bads))

        # Write to BIDS
        task, run = blocks[block]
        int(subject[1:])  # make sure it's an int
        bids_subject = subject[1:]  # remove "G"
        bids_path = mne_bids.BIDSPath(
            subject=bids_subject,
            task=task,
            run=run,
            suffix=suffix,
            datatype=datatype,
            root=bids_root,
        )
        mne_bids.write_raw_bids(
            raw_meg,
            bids_path,
            events=events,
            event_id=event_id,
            empty_room=empty_room,
            overwrite=True,
            allow_preload=True,
            format="FIF",
            verbose="error",  # ignore stuff about events being empty for now
        )
