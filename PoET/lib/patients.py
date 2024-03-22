"""
Classes for creating patient objects compatible with TWSTRS.
"""
import logging

import numpy as np
import pandas as pd
#import simdkalman
from scipy import ndimage
from scipy.signal import medfilt
from .structural_features import compute_distances
from .utils import pre_high_pass_filter, pre_low_pass_filter
from .pio import save_object, load_object
from copy import deepcopy

MIN_TIME_SAMPLES = 100

L = logging.getLogger(__name__)


class PatientCollection:
    """A collection of patient objects (see patient class)."""

    def __init__(self):
        """Initialise an empty list of patients."""
        self.patients = []
        self._markers = None

    def add_patient(self, patient, node_features=None, label=None):
        """Add a patient to the list.

        Args:
            patient (patient-like object): valid data representing patient (see convert_patient)
            node_feature (array): node feature matrix
            label (int): label of the patient
        """
        if not isinstance(patient, Patient):
            L.warning("Not a Patient Object")
        patient.id = len(self.patients)
        self.patients.append(patient)

    def add_patient_list(
        self, patient_list, patient_labels=None,
    ):
        """Add a list of patients.

        Args:
            patient_list (list(patient-like object)): valid data representig patients (see convert_patient)
            patient_labels (list(int)): label of the patients
        """
        for patient in patient_list:
            if not isinstance(patient, Patient):
                L.warning("Not a Patient Object")
            
            if patient.patient_id is not None:
                patient.id = patient.patient_id
            else:
                patient.id = len(self.patients)

            self.patients.append(patient)

    def __iter__(self):
        self.current_patient = -1
        return self

    def __next__(self):
        self.current_patient += 1
        if self.current_patient >= len(self.patients):
            raise StopIteration
        while self.patients[self.current_patient].disabled:
            self.current_patient += 1
            if self.current_patient >= len(self.patients):
                raise StopIteration
        return self.patients[self.current_patient]

    def get_n_node_features(self):
        """Get the number of features of the nodes."""
        n_node_features = self.patients[0].n_node_features
        for patient in self.patients:
            assert n_node_features == patient.n_node_features
        return n_node_features

    def get_num_disabled_patients(self):
        """Get the number of disabled patients."""
        return len(self.patients) - self.__len__()

    def get_patient_ids(self):
        """Get the list of active patient ids."""
        return [patient.patient_id for patient in self.patients if not patient.disabled]

    def aggregate_markers(self,aggregation_dict):
        for patient in self.patients:
            patient.aggregate_markers(aggregation_dict)

    def update_time(self):
        """ check if time column exists for all patients """
        for patient in self.patients:
            patient._check_time()
    
    def get_patient(self,patient_id):
        return [patient for patient in self.patients if patient.patient_id==patient_id][0]

    def remove_patient(self,patient_id):
        self.patients = [patient for patient in self.patients if patient.patient_id != patient_id]
    
    @property
    def markers(self):
        if self._markers is None:
            self._markers = self.patients[0].markers
        return self._markers

    def __len__(self):
        return sum([1 for patient in self.patients if not patient.disabled])

    def _check_cleaned(self):
        for patient in self.patients:
            assert patient.cleaned == 1
            
    def save(self, folder = './outputs/saved/', filename='patient_collection'):
        save_object(self, folder=folder, filename=filename)

    def load(self, folder = './outputs/saved/', filename='patient_collection'):        
        return load_object(folder=folder, filename=filename)


class Patient:
    """
    Class to encode a generic patient structure for twstrs.
    """

    def __init__(self,
                pose_estimation=None,
                sampling_frequency=None,
                label=None,
                label_name=None,
                patient_id=None,
                low_cut=0,
                high_cut=None,
                movement_labels=None,
                clean=True,
                normalize=True,
                expected_markers=None,
                likelihood_cutoff=0.9,
                scaling_factor=1,
                spike_threshold=0,
                interpolate_pose=False,
                smooth=None,
                smoothing_window_length=3,
                video_details = None
                ):
        """Defining the main patients quantities.

        Args:
            pose_estimation (DataFrame): dataframe of pose estimation, index as time id, and
                columns corresponding to tracked markers
            label (int): label of the patient
            label_name (any): name or other information on the patient label
            patient_id (str): identifiable information of patient
        """

        self.pose_estimation = pose_estimation
        self.sampling_frequency = sampling_frequency
    
        if pose_estimation is not None:
            self.markers = list(set(pose_estimation.columns.droplevel(1).tolist()))
            
        self.movement_labels_df = None
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.likelihood_cutoff=likelihood_cutoff
        self.scaling_factor=scaling_factor
        self.behavioural_predictions=dict()
        self.spike_threshold = spike_threshold
        self.interpolate_pose = interpolate_pose
        
        self.expected_markers = expected_markers 

        self.patient_id = patient_id
        self.label = label
        self.label_name = label_name
        self.disabled = False
        self.id = -1
        self.cleaned = 0
        self.dummy_variables = None
        self.smooth=smooth
        self.window_length=smoothing_window_length

        self.video_details = video_details

        if pose_estimation is not None:
            self._check_markers()
            self._check_length()
            self._check_time()
            self.marker_visibility()
            
        if normalize and pose_estimation is not None:
            self.normalize_data()
            
        if clean and pose_estimation is not None:
            self.clean_data()



        self.movement_labels = None

    def _check_length(self):
        """Verify if the patient video is long enough to be considered."""
        if len(self.pose_estimation.index) <= MIN_TIME_SAMPLES:
            self.disabled = True

    def _check_markers(self):
        """Verify if the patient video has all the correct markers."""
        if self.expected_markers is not None:
            if not set(self.markers).issubset(self.expected_markers):
                self.disabled = True
            self.pose_estimation = self.pose_estimation[self.expected_markers]
        
        if ('bodyparts','coords') in self.pose_estimation.columns:
            self.pose_estimation = self.pose_estimation.drop(('bodyparts','coords'),axis=1)
            
        self._update_markers()

    def _update_markers(self):
        """ Defining markers and removing standard markers """
        self.markers = list(set(self.pose_estimation.columns.droplevel(1).tolist()))
        if 'time' in self.markers:
            self.markers.remove('time')
        if 'bodyparts' in self.markers:
            self.markers.remove('bodyparts')

    def _check_time(self):
        """Verify if time column exists."""
        if 'time' not in self.pose_estimation.columns:
            self.pose_estimation['time'] = np.asarray(self.pose_estimation.index*(1/self.sampling_frequency))

    def _reset_index(self):
        """resetting index """
        self.pose_estimation = self.pose_estimation.reset_index(drop=True)

    def _update_time(self):
        """update time column """
        self.pose_estimation['time'] = np.asarray(self.pose_estimation.index*(1/self.sampling_frequency))

    def clean_data(self):
        """ Cleaning and filtering data """
            
        if self.interpolate_pose:
            self.pose_estimation = self.pose_estimation.interpolate(limit_direction='both') 

        if self.low_cut > 0:
            self.pose_estimation = filter_low_frequency(self.pose_estimation,self.markers,self.sampling_frequency,self.low_cut)
        
        if self.spike_threshold > 0:
            self.pose_estimation = remove_spikes(self.pose_estimation, self.markers, threshold = self.spike_threshold)

        if self.high_cut is not None:
            self.pose_estimation = filter_high_frequency(self.pose_estimation,self.markers,self.sampling_frequency,self.high_cut)

        if 'likelihood' in self.pose_estimation.columns.get_level_values(1):
            self.pose_estimation = remove_low_likelihood(self.pose_estimation, self.markers, self.likelihood_cutoff)
            
            if self.interpolate_pose:
                self.pose_estimation = self.pose_estimation.interpolate(method='linear', limit_direction='both') 


        if self.smooth is not None:
            if self.smooth=='median':
                self.pose_estimation = smooth_median(self.pose_estimation, self.markers, window_len=self.window_length)
        
        #if self.kalman:
        #    self.pose_estimation = filter_pose_estimation(self.pose_estimation, self.markers, self.sampling_frequency)
        
        self.cleaned = 1

    def normalize_data(self):
        """ Scale the coordinates by the average distance of all points """
        if isinstance(self.scaling_factor,str):
            if self.scaling_factor == 'manual':
                scale_factor = self.scaling_factor
                self.pose_estimation.loc[:,self.markers] *= scale_factor # mm/pixels

            elif self.scaling_factor == 'eye_distance':
                L.info("Using eye distance as scale factor.")                
                if 'REye' not in self.pose_estimation and 'LEye' not in self.pose_estimation:
                    self.aggregate_eyes()
                    
                eye_distance_pixels = np.nanmedian(np.sqrt(((self.pose_estimation['REye']-self.pose_estimation['LEye'])**2).sum(axis=1)))
                scale_factor = 60/eye_distance_pixels
                self.pose_estimation = self.pose_estimation.loc[:,self.markers].mul(scale_factor,axis=0)
                
            elif self.scaling_factor == 'median':            
                L.info("Scaling factor not given. Using median distances across all markers.")
                scale_factor = np.nanmedian(compute_distances(self.pose_estimation, self.markers))
                self.pose_estimation.loc[:,self.markers] *= scale_factor # mm/pixels

        elif np.isnan(self.scaling_factor):
            L.info("Scaling factor is NaN for {}".format(self.patient_id))
            scale_factor = 1            
            self.pose_estimation.loc[:,self.markers] *= scale_factor # mm/pixels
        elif isinstance(self.scaling_factor,float): 
            self.pose_estimation.loc[:,self.markers] *= self.scaling_factor # mm/pixels
        elif isinstance(self.scaling_factor,int): 
            self.pose_estimation.loc[:,self.markers] *= self.scaling_factor # mm/pixels            
        else:
            print('No proper scaling factor given -- not scaling')
        
    def copy(self):        
        return deepcopy(self)            

    def aggregate_eyes(self):
        aggregation_dict = {
            'LEye': ['LEye1','LEye2','LEye3','LEye4'],
            'REye': ['REye1','REye2','REye3','REye4']
            }
        self.aggregate_markers(aggregation_dict)

    def get_pose(self, marker=None):
        """Get time-series of given marker. """
        if marker is None:
            return self.pose_estimation
        else:
            return self.pose_estimation[marker]

    def aggregate_markers(self, aggregation_dict):
        """ Aggregating multiple markers into single markers via dictionary """
        self.pose_estimation = aggregate_markers(self.pose_estimation, aggregation_dict)
        if self.dummy_variables:            
            self.dummy_variables = aggregate_dummies(self.dummy_variables, aggregation_dict)
        self._update_markers()

    def marker_visibility(self):
        """ Creating dummy variables for visibility of markers """
        if 'likelihood' in self.pose_estimation.columns:
            self.dummy_variables = get_marker_visiblity_dummies(self.pose_estimation, self.markers, visibility_likelihood=0.01)



# def remove_spikes(data, markers, threshold = 20):
    
#     for marker in markers:
#         x = data[(marker,'x')]
#         y = data[(marker,'y')]        
        
#         for i in range(y.shape[0]-1):
            
#             diff_x = abs(x[i]-x[i+1])
#             diff_y = abs(y[i]-y[i+1])
    
#             if max(diff_x,diff_y) > threshold:
#                 y[i+1] = y[i]
#                 x[i+1] = x[i]
           
#         data[(marker,'x')] = x
#         data[(marker,'y')] = y
        
#     return data

def remove_spikes(data, markers, threshold = 20):
    
    for marker in markers:
        
        data[(marker,'x')] = _remove_spikes(data[(marker,'x')], threshold)
        data[(marker,'y')] = _remove_spikes(data[(marker,'y')], threshold)
        
        corrected_data = detect_and_correct_outliers(data[[(marker,'x'),(marker,'y')]].values)
        data[(marker,'x')] = corrected_data[:,0]
        data[(marker,'y')] = corrected_data[:,1]
        #data.loc[outlier_periods,(marker,'x')] = np.nan
        #data.loc[outlier_periods,(marker,'y')] = np.nan
        
        data[(marker,'x')] = data[(marker,'x')].interpolate(method='linear',limit_direction='both')
        data[(marker,'y')] = data[(marker,'y')].interpolate(method='linear',limit_direction='both')
        
        #data[(marker,'y')] = 
        
    return data

def _remove_spikes(signal, median_window=5, std_window=20, threshold_factor=3):
    # Ensure median_window is odd
    if median_window % 2 == 0:
        median_window += 1
    
    # Step 1: Apply median filter to smooth the signal
    median_filtered = pd.Series(signal).rolling(window=median_window, center=True).median().fillna(method='bfill').fillna(method='ffill')
    
    # Step 2: Calculate rolling standard deviation on the median-filtered signal
    rolling_std = median_filtered.rolling(window=std_window, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Step 3: Identify spikes - where the difference between the original signal and the median exceeds the threshold
    spikes = np.abs(signal - median_filtered) > (threshold_factor * rolling_std)
    
    # Replace spikes with NaN or another suitable method
    clean_signal = np.copy(signal)
    clean_signal[spikes] = np.nan
    
    # Interpolating to fill the gaps left by removing spikes
    clean_signal = pd.Series(clean_signal).interpolate(method='quadratic',limit_direction='both').to_numpy()
    
    return clean_signal


def detect_and_correct_outliers(xy_data, jump_threshold=100, return_threshold=100):
    """
    Detects outlier periods in x,y tracking data where the marker jumps away and then returns.

    :param xy_data: Array of shape (n, 2) representing x, y coordinates over time.
    :param jump_threshold: Distance indicating a significant jump, marking potential start of an outlier period.
    :param return_threshold: Distance to the last good point indicating the marker has returned.
    :return: Corrected x,y tracking data.
    """
    n = len(xy_data)
    is_outlier = np.zeros(n, dtype=bool)
    last_good_index = 0

    i = 1
    while i < n:
        distance_from_last_good = np.linalg.norm(xy_data[i] - xy_data[last_good_index])

        if distance_from_last_good > jump_threshold:
            # Potential start of an outlier period
            for j in range(i + 1, n):
                distance_from_last_good = np.linalg.norm(xy_data[j] - xy_data[last_good_index])
                if distance_from_last_good <= return_threshold:
                    # Found the end of the outlier period
                    is_outlier[i:j] = True
                    i = j  # Continue from the end of this outlier period
                    break
            last_good_index = i
        else:
            last_good_index = i
        i += 1

    # Correct the identified outliers
    corrected_data = xy_data.copy()
    for i in range(1, n-1):
        if is_outlier[i]:
            corrected_data[i] = np.interp(i, [i for i in range(n) if not is_outlier[i]], xy_data[~is_outlier, 0]), np.interp(i, [i for i in range(n) if not is_outlier[i]], xy_data[~is_outlier, 1])



    return corrected_data

def detect_outlier_periods(xy_data, distance_threshold=100, min_outlier_duration=1):
    """
    Detect outlier periods in 2D marker tracking data.
    
    :param xy_data: A numpy array of shape (n, 2) representing x, y coordinates over time.
    :param distance_threshold: Threshold for detecting spatial jumps as outliers.
    :param min_outlier_duration: Minimum duration (in number of frames) to consider a group of points as an outlier period.
    :return: A boolean array where True represents outliers.
    """
    # Calculate Euclidean distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(xy_data, axis=0)**2, axis=1))
    distances = np.insert(distances, 0, 0)  # Add a 0 at the start to keep the array the same length as xy_data
    
    # Identify points where the distance exceeds the threshold
    outliers = distances > distance_threshold
    
    # Temporally cluster outliers and filter out those that do not meet the minimum duration
    outlier_periods = pd.Series(outliers).rolling(window=min_outlier_duration, center=True).max().fillna(0).astype(bool)
    
    return outlier_periods.to_numpy()

def correct_outliers(xy_data, outlier_periods):
    """
    Correct outlier periods by linear interpolation.
    
    :param xy_data: Original x, y coordinates.
    :param outlier_periods: Boolean array indicating outlier periods.
    :return: Corrected x, y coordinates.
    """
    corrected_data = xy_data.copy()
    for i in range(len(outlier_periods)):
        if outlier_periods[i]:
            if i == 0 or i == len(outlier_periods) - 1:
                continue  # Skip correction if outlier is at the start or end
            prev_non_outlier = i - np.where(~outlier_periods[:i])[0][-1]
            next_non_outlier = np.where(~outlier_periods[i:])[0][0] + i
            corrected_data[i] = corrected_data[prev_non_outlier] + (corrected_data[next_non_outlier] - corrected_data[prev_non_outlier]) * ((i - prev_non_outlier) / (next_non_outlier - prev_non_outlier))
            
    return corrected_data


def _remove_spikes2(x, threshold):
    x_diff = np.where(abs(x.diff())>threshold)[0]
    
    if not x_diff.size==0:    
        sections = np.vstack([np.hstack([0,x_diff[:-1]]),x_diff]).T
        section_means = np.zeros(sections.shape[0])
        for i in range(sections.shape[0]):    
            section_means[i] = np.mean(x.iloc[sections[i,0]:sections[i,1]])
        
            section_diff = np.mean(x.iloc[sections[i,0]:sections[i,1]]) - np.nanmedian(x)
            if abs(section_diff) > threshold:
                x.loc[sections[i,0]:sections[i,1]] = np.nan
    
        return x.interpolate(method='linear', limit_direction='both')
    else:
        return x

def get_marker_visiblity_dummies(data, markers, visibility_likelihood=0.01):
    """ Create dummy variables for visibility of markers """
    dummy_df = pd.DataFrame()
    for marker in markers:
        dummy_df[marker+'_visible'] = (data[marker]['likelihood']>visibility_likelihood).astype(int)
    return dummy_df

def remove_low_likelihood(data, markers, likelihood):
    """ Setting low likelihood samples to zero """
    for marker in markers:
        data.loc[(data[marker]['likelihood']<likelihood),(marker,'x')] = np.nan
        data.loc[(data[marker]['likelihood']<likelihood),(marker,'y')] = np.nan
        
    data = data.drop('likelihood', axis=1, level=1)
    return data

def filter_low_frequency(
    data,
    markers,
    fps,
    low_cut=0,
):
    """ function to remove slow frequencies"""
    
    filtered_data = pre_high_pass_filter(data,markers,fps,low_cut=low_cut)    
    return filtered_data

def filter_high_frequency(
    data,
    markers,
    fps,
    high_cut=None,
):
    """ function to remove high frequencies"""
    filtered_data = pre_low_pass_filter(data,markers,fps,high_cut=high_cut)    
    return filtered_data

# def filter_pose_estimation(
#     data,
#     markers,
#     sampling_frequency,
#     keep_na=True,
# ):
#     """Function to smoothen marker estimation"""

#     kf = simdkalman.KalmanFilter(
#                                 state_transition = np.array([[1,1],[0,1]]),
#                                 process_noise = np.diag([0.1, 0.01]),
#                                 observation_model = np.array([[1,0]]),
#                                 observation_noise = 1.0
#                                 )
#     data_smoothed = data.copy()
#     for marker in markers:
#         data_smoothed[(marker,'x')] = kf.smooth(data_smoothed[(marker,'x')]).observations.mean
#         data_smoothed[(marker,'y')] = kf.smooth(data_smoothed[(marker,'y')]).observations.mean

#     if keep_na:
#         data_smoothed = data_smoothed.mask(data.isnull())

#     return data_smoothed

def smooth_median(data,
                    markers,
                    window_len=3,
                    keep_na=True,
):
    
    data_smoothed = data.copy()
    for marker in markers:
        data_smoothed[(marker,'x')] = ndimage.median_filter(data_smoothed[(marker,'x')], size=window_len)
        data_smoothed[(marker,'y')] = ndimage.median_filter(data_smoothed[(marker,'y')], size=window_len)
    
    if keep_na:
        data_smoothed = data_smoothed.mask(data.isnull())

    return data_smoothed


def smooth(x,window_len=10,window='flat'):

    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]
    

def aggregate_dummies(data,aggregation_dict):
    dummy_aggregation_dict = {f'{k}_visible': [f'{u}_visible' for u in v] for k, v in aggregation_dict.items()}
    for key in dummy_aggregation_dict.keys():
        if set(dummy_aggregation_dict[key]).issubset(set(data.columns.get_level_values(0))):
            sub_data = data[dummy_aggregation_dict[key]].copy()
            data[key] = sub_data.mean(axis=1)
            data[key] = sub_data.mean(axis=1)
            data = data.drop(dummy_aggregation_dict[key], axis = 1)
        else:
            L.warning('Dummy columns not present for key: {}'.format(key))   
    return data

def aggregate_markers(data, aggregation_dict):
    """ Aggregates markers """

    for key in aggregation_dict.keys():
        if set(aggregation_dict[key]).issubset(set(data.columns.get_level_values(0))):
            sub_data = data[aggregation_dict[key]].copy()
            data[pd.IndexSlice[key, 'x']] = sub_data.loc[:, pd.IndexSlice[:, 'x']].mean(axis=1)
            data[pd.IndexSlice[key, 'y']] = sub_data.loc[:, pd.IndexSlice[:, 'y']].mean(axis=1)
            data = data.drop(aggregation_dict[key], axis = 1, level=0)
        else:
            L.warning('Columns not present for key: {}'.format(key))

    return data



def combine_repeats(patient_collection):
    """ function to combine patients with repeats """
    
    patient_ids = patient_collection.get_patient_ids()
    patients = []
    repeated_patients = []
    for patient in patient_ids:
        split_name = patient.split(' ')
        
        if len(split_name) > 1:        
            patient_name =  split_name[0]   
        else: 
            patient_name = patient
            
        repeats = [p for p in patient_ids if patient_name in p]    
        repeated_patients.append(repeats)
        
    unique_patients = [list(x) for x in set(tuple(x) for x in repeated_patients)]    
    
    
    patients = []
    for unique_patient in unique_patients: 
        
        if len(unique_patient)>1:  
            
            pose_estimations = []
            video_details = []        
            for repeat in unique_patient:            
                patient_repeat = patient_collection.get_patient(repeat)
                pose_estimations.append(patient_repeat.pose_estimation)
                video_details.append(patient_repeat.video_details)
                
            patient_id = repeat.split(' ')[0]
            pose_estimation = pd.concat(pose_estimations)  
            likelihood = patient_repeat.likelihood_cutoff
                
            patient = Patient(pose_estimation,
                              video_details[0]['fps'],
                              patient_id=patient_id,
                              likelihood_cutoff=likelihood,
                              low_cut=0,
                              )   
            
            patient._reset_index()
            patient._update_time()
            
            patient.video_details = video_details
                
        else: 
             patient = patient_collection.get_patient(unique_patient[0])         
    
        patients.append(patient)
    
    pc = PatientCollection()
    pc.add_patient_list(patients)    
    
    return pc    




