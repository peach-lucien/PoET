""" 
Defining structural features to be computed 
"""


import logging

import numpy as np
import pandas as pd
from scipy import signal
import csv
import itertools

L = logging.getLogger(__name__)

EXPECTED_KEYS = ['distances','angles','areas','ratios','markers']

class StructuralFeatures:
    """ The collection of structural features to be computed."""

    def __init__(self, features=None, markers=[]):
        """ Creating an object with features to compute."""
        if isinstance(features,str):
            self.load_structural_features(features) # load from file
        elif isinstance(features,dict):
            self.features=features
        else:
            L.info("No features parsed: creating default where all pairwise distances are computed only.")
            self._set_default()
        
        self.markers = markers 
        self._check_keys()   
        self._check_features()          

    def add_feature(self, structural_type, markers):
        """Add a feature to the dict. """
        self.features[structural_type].append(markers) 

    def _check_keys(self):
        """ check if all keys are present in dictionary """
        if not set(EXPECTED_KEYS).issubset(self.features.keys()):
             L.warning("Not all keys are present in feature dictionary.")   

    def _check_features(self):
        """ check feature definitions are of correct length """
        for key in self.features.keys():
            if self.features[key]:

                feat_lens = np.asarray([len(f) for f in self.features[key]])
                if key=='distances':
                    if not all(feat_lens==2):
                        L.warning("Some pairwise distance features have more/less than 2 entries.") 
                elif key=='angles':
                    if not all(feat_lens==3):
                        L.warning("Some angle features have more/less than 3 entries.")  
                elif key=='areas':
                    if not all(feat_lens >= 3):
                        L.warning("Some area features have less than 3 entries.")  
                elif key=='ratios':
                    if self.markers:                    
                        self._check_ratios()
                    else:
                        L.warning('No markers were provided to check if the given ratio features are computable')
                elif key=='markers':
                    if not all(feat_lens==2):
                        L.warning("Some markers have more/less than 2 entries.") 

    def _check_ratios(self):
        for i,f in enumerate(self.features['ratios']):
            markers = f[0].split('_')[1:] + f[1].split('_')[1:]
            for marker in markers:
                if marker not in self.markers:
                    L.warning('Marker {} with name: {} not in data'.format(i,marker))

    def _set_default(self):
        feature_dict = {'distances': [], 'angles': [], 'areas': [], 'ratios': []}
        self.features = feature_dict

    def load_structural_features(self, project_name, folder="../structural_feature_files/"):
        """ Load structural features from file """
        for key in EXPECTED_KEYS:
            with open(folder + project_name +  "/structural_features_{}_".format(key)+".csv", 'r') as f:
                reader = csv.reader(f)
                list_of_rows = list(reader)
                self.features[key] = list_of_rows        
        self._check_keys()   
        self._check_features()         

    def save_structural_features(self, project_name, folder="../structural_feature_files/"):
        """ Saving structural features for another time """
        for key in self.features.keys():
            list_of_lists = self.features[key]
            with open(folder + project_name +  "/structural_features_{}_".format(key)+".csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(list_of_lists)


def compute_marker_dynamics(data, features, fs):
    markers_df = pd.DataFrame(index = data.index)
    
    for f in features:        
         freq, amplitude = frequency_amplitude(data, f, fs)   
         markers_df['marker_'+'_'.join(f[:2]) + '_dominant_freq'] = freq
         markers_df['marker_'+'_'.join(f[:2]) + '_amplitude'] = amplitude
         #markers_df['marker_'+'_'.join(f[:2])] = data[(f[0],f[1])]
    
    return markers_df

def frequency_amplitude(x, fs):
    
    x = np.pad(x,(int(fs/2),int(fs/2)),mode='symmetric')   
    
    min_freq = 0
    max_freq = 100

    amplitudes = []
    frequencies = []    

    for i in range(len(x)): 
        
        #y = x # like this it won't vary at all...  
        y = x[i:i+int(fs)]
        
        frequency, amplitude = signal.welch(y, fs, nperseg=len(y))

        freq_limits = (min_freq<=frequency) & (max_freq>=frequency)
        amplitude = np.sqrt(amplitude[freq_limits])
        frequency = frequency[freq_limits]   
        
        amplitudes.append(np.max(amplitude))
        frequencies.append(frequency[np.argmax(amplitude)])
              

    amplitude = np.asarray(amplitudes)
    dominant_frequency = np.asarray(frequencies)
    
    return np.mean(dominant_frequency), np.mean(amplitude)



def compute_distances(data, features):
    """ Compute distances between defined markers """

    
    if not any(isinstance(el, list) for el in features):
        features = list(itertools.combinations(features, 2))
        features = [list(feat) for feat in features]

    distances_df = pd.DataFrame(index = data.index)

    for f in features:
        distances_df['distance_'+'_'.join(f)] = distance(data, f)

    return distances_df 


def distance(data, feature):
    """ computing pairwise distance """
    
    # computing for only x and y distances if chosen
    if len(feature)>2:
        if feature[2] == 'x':
            delta_x = data.loc[:, pd.IndexSlice[feature[0], 'x']] - data.loc[:, pd.IndexSlice[feature[1], 'x']]
            dist = np.sqrt(delta_x**2)*np.sign(delta_x)
        elif feature[2] == 'y':
            delta_y = data.loc[:, pd.IndexSlice[feature[0], 'y']] - data.loc[:, pd.IndexSlice[feature[1], 'y']]
            dist = np.sqrt(delta_y**2)*np.sign(delta_y)
    else:
        delta_x = data.loc[:, pd.IndexSlice[feature[0], 'x']] - data.loc[:, pd.IndexSlice[feature[1], 'x']]
        delta_y = data.loc[:, pd.IndexSlice[feature[0], 'y']] - data.loc[:, pd.IndexSlice[feature[1], 'y']]
        
        if 'z' in data.columns.levels[1]:
            delta_z = data.loc[:, pd.IndexSlice[feature[0], 'z']] - data.loc[:, pd.IndexSlice[feature[1], 'z']]
            dist = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        else:            
            dist = np.sqrt(delta_x**2 + delta_y**2)
    
    return dist

def compute_areas(data, features):
    """ Compute areas between defined markers """

    areas_df = pd.DataFrame(index = data.index)

    for f in features:
        areas_df['area_'+'_'.join(f)] = area(data, f)
    
    return areas_df 


def area(data, feature, degrees=False):
    """ computing areas between n>2 points """

    x = data.loc[:, pd.IndexSlice[feature, 'x']].values
    y = data.loc[:, pd.IndexSlice[feature, 'y']].values

    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    area = np.zeros([x.shape[0]])
    for i in range(x.shape[0]):
        area[i] = PolyArea(x[i,:],y[i,:])

    return area


def compute_angles(data, features):
    """ Compute angles between defined markers """

    angles_df = pd.DataFrame(index = data.index)
    for f in features:
        angles_df['angle_'+'_'.join(f)] = angle(data, f)    
    return angles_df 


def angle(data, feature, degrees=True):
    """ computing angles between three points """
    
    if feature[0]=='horizontal':
        a = np.vstack([ data[(feature[2],'x')].values, data[(feature[1],'y')].values]).T
    else:        
        a = data[feature[0]].values
        
    b = data[feature[1]].values
    c = data[feature[2]].values
    ba = a - b
    bc = c - b

    #cosine_angle =  np.einsum('ij,ij->i', ba, bc) / (np.linalg.norm(ba,axis=1) * np.linalg.norm(bc,axis=1))
    cosine_angle =  np.sum(ba*bc, axis=1) / (np.linalg.norm(ba,axis=1) * np.linalg.norm(bc,axis=1))    
    angle = np.arccos(cosine_angle)
        
    if feature[0]=='horizontal':
        pos_neg_angle = np.sign(data[(feature[1],'y')].values - data[(feature[2],'y')].values)
        angle = angle*pos_neg_angle
    
    if degrees:
        angle = np.degrees(angle)        
    return angle


def compute_ratios(data, features):
    """ Compute ratios between features """
    ratios_df = pd.DataFrame(index = data.index)
    for f in features:
        ratios_df['ratio_'+'_'.join(f)] = ratio(data, f)
    return ratios_df 


def ratio(data, feature):
    """ computing difference of computed features """
    return data[feature[0]]-data[feature[1]]
    