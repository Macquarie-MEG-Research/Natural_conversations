# This is script to calculate connectivity using raw np arrays,
# each assumed to be in the shape (channels/ROI's, time points)

import numpy as np
from datetime import datetime
import os
import dyconnmap
import bct
from scipy import stats
import pathlib

def operations_to_perform():    # Change as desired
    Cal_wpli = True
    Cal_graph_metrics = True
    return Cal_wpli, Cal_graph_metrics


def setup_dependencies():    # Change as desired
    input_dir = r'C:\Users\em17531\Desktop\New_project\source_loc_output'     # The directory containing files of interest
    output_dir = r''    # The output directory
    array_type = 'participant_conversation_alpha'     # This is the type of array that will be loaded for further calculations, eg. "participant_conversation_alpha"
    target_fb = [8,12]  # The target frequency band     TODO: set this up so variable can be a this can be list of bands and each generates it's own adjacency matrix
    fs = 250    # The sampling frequency
    saving = True
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir, array_type, fs, target_fb, saving


def prep_data(input_dir, array_type):
    data = []
    folder = pathlib.Path(input_dir).glob('*')
    for file in folder:
        if (os.path.basename(file)).endswith('.npy'):
            if array_type in os.path.basename(file):
                data.append(np.load(file))
    return np.array(data)


def wpli_conn(array, target_fb, fs):   # (participants, roi's, time_points)
    adj_matrix = []
    for participant in array:
        adj_matrix.append(dyconnmap.fc.wpli(participant, fs=fs, fb=target_fb))
    return np.array(adj_matrix)


def graph_metrics(adj_matrix):  # TODO: Add in stats, maybe nbs? Could use fdc as well?
    # Strength calculator
    Strength = []
    for participant in adj_matrix:
        Strength.append(bct.strengths_und(np.nan_to_num(participant)))
        Strength = np.array(Strength)

    # Zeroing negative phasing
    Strength[Strength < 0] = 0

    # Betweenness centrality calculator
    Betweenness = []
    for participant in adj_matrix:
        Betweenness.append(bct.betweenness_wei(np.nan_to_num(participant)))
        Betweenness = np.array(Betweenness)

    # Eigenvector centrality calculator
    Eigenvector = []
    for participant in adj_matrix:
        Eigenvector.append(bct.eigenvector_centrality_und(np.nan_to_num(participant)))
        Eigenvector = np.array(Eigenvector)

    # Clustering calculator
    Clustering = []
    for participant in adj_matrix:
        Clustering.append(bct.clustering_coef_wu(np.nan_to_num(participant)))
        Clustering = np.array(Clustering)

    return Strength, Betweenness, Eigenvector, Clustering


def main():     # TODO: put in the elif statements
    # Prep
    Cal_wpli, Cal_graph_metrics = operations_to_perform()
    input_dir, output_dir, array_type, fs, target_fb, saving = setup_dependencies()
    data = prep_data(input_dir, array_type)

    # Core functions
    if Cal_wpli:
        adj_matrix = wpli_conn(data, target_fb, fs)
    if Cal_graph_metrics:
        Strength, Betweenness, Eigenvector, Clustering = graph_metrics(adj_matrix)

    # Saving
    if saving:
        np.save(output_dir + 'All_Strength_' + array_type, Strength)
        np.save(output_dir + 'All_Betweenness_' + array_type, Betweenness)
        np.save(output_dir + 'All_Eigenvector_' + array_type, Eigenvector)
        np.save(output_dir + 'All_Clustering_' + array_type, Clustering)


if __name__ == "__main__":
    main()
