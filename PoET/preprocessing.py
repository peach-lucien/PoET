
import os

import pandas as pd
import numpy as np

from PoET.lib.patients import Patient, PatientCollection
from PoET.utils import compute_spectrogram


def construct_data(csv_files, fs, labels=None, scaling_factor=1, verbose=True, smooth=None):
        
    if isinstance(scaling_factor, int):
        scaling_factor = [scaling_factor] * len(csv_files)

    if isinstance(fs, int):
        fs = [fs] * len(csv_files)    

    patients = []
    for i, file in enumerate(csv_files):        
        
        # get filename 
        file_name = os.path.basename(file).split('.')[0]   
        
        if verbose:
            print('Loading: {}'.format(file_name))
        
        # load the csv file
        pose_estimation = pd.read_csv(file, header=[0,1], index_col=0)

        # ensure no capital letters in column names for consistency
        pose_estimation = columns_to_lowercase(pose_estimation)

        # construct a patient object
        p = Patient(pose_estimation,
                    fs[i],
                    patient_id=file_name,
                    likelihood_cutoff=0,
                    label=labels[i],
                    low_cut=0,
                    high_cut=None,
                    kalman=False,
                    clean=True,
                    scaling_factor=scaling_factor[i],
                    normalize=True,
                    spike_threshold=0,
                    interpolate_pose=True,
                    smooth=smooth,
                    )
        
        patients.append(p)    

    # constuct patient collection 
    pc = PatientCollection()
    pc.add_patient_list(patients)
    
   
    return pc


def columns_to_lowercase(data):    
    data.columns = pd.MultiIndex.from_frame(data.columns.to_frame().applymap(str.lower))
    return data



