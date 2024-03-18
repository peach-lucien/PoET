

import pandas as pd
import numpy as np

from scipy import signal
from scipy.signal import  hilbert, butter, lfilter
import matplotlib.pyplot as plt

from PoET.utils import identify_active_time_period, check_hand_


def assign_hand_time_periods(pc):
    
    # loop over patients and extract tapping features
    for p in pc.patients:
        
        # extract sampling frequency 
        fs = p.sampling_frequency
        
        # identify tracked hand
        hands = check_hand_(p) 
        
        # extract structual features
        structural_features = p.structural_features 
        
        # if analysing both hands in one video then assign time periods of dominant hand
        if len(hands)>1:
            time_periods = identify_active_time_period(structural_features, fs)    
        else:
            time_periods = None
        
        p.hand_time_periods = time_periods
    
    return pc


def spectrogram(x, fs):
    
    n = len(x)
    x = np.pad(x,(int(fs/2),int(fs/2)),mode='symmetric')   
    
    #min_freq = 0
    #max_freq = 100

    amplitudes = []
    frequencies = []    

    for i in range(n): 
        
        #y = x # like this it won't vary at all...  
        y = x[i:i+int(fs)]
        
        fft_result = np.fft.fft(y)
        frequency = np.fft.fftfreq(y.size, d=1/fs)
        
        # Calculate the magnitude of the FFT (amplitude spectrum)
        amplitude = 2* np.abs(fft_result) / len(y) # Multiplying by 2 because the spectrum is symmetrical for real-valued signals
        
        amplitudes.append(amplitude) #np.max(amplitude))
        frequencies.append(frequency)#frequency[np.argmax(amplitude)])              

    amplitude = np.vstack(amplitudes)
    #dominant_frequency = np.vstack(frequencies)
    
    # positive frequencies only
    amplitude = amplitude[:,frequency>=0]
    frequency = frequency[frequency>=0]
    
    return frequency, amplitude


def spectrum(y, fs):
    
    fft_result = np.fft.fft(y)
    frequency = np.fft.fftfreq(y.size, d=1/fs)
    
    # Calculate the magnitude of the FFT (amplitude spectrum)
    amplitude = 2* np.abs(fft_result) / len(y) # Multiplying by 2 because the spectrum is symmetrical for real-valued signals
    
    # positive frequencies only
    amplitude = amplitude[frequency>=0]
    frequency = frequency[frequency>=0]   
    
    return frequency, amplitude
    

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_tremor_features(pc, tremor_type='postural', plot=False, save_plots=False):
    
    # compute tremor features for postural or kinematic
    if tremor_type=='postural':
        features = extract_postural_tremor_features(pc,plot=plot,save_plots=save_plots)
    elif tremor_type=='kinematic':
        features = extract_kinematic_tremor_features(pc,plot=plot,save_plots=save_plots)
        
    # compute proximal tremor features
    proximal_features = extract_proximal_tremor_features(pc,plot=plot,save_plots=save_plots)
    
    features = pd.concat([features, proximal_features],axis=1)
    
    # order columns alphabetically
    features = features.reindex(sorted(features.columns), axis=1)
    
    return features

def extract_kinematic_tremor_features(pc, plot=False, save_plots=False):
    
    # define markers for analysing tapping
    feats ={'right':['marker_index_finger_tip_right_x','marker_index_finger_tip_right_y'],
            'left':['marker_index_finger_tip_left_x','marker_index_finger_tip_left_y'],
            }    
    
    features_df = pd.DataFrame(index = pc.get_patient_ids())

    # loop over patients and extract tapping features
    for p in pc.patients:
        
        # extract sampling frequency 
        fs = p.sampling_frequency
        
        # identify tracked hand
        hands = check_hand_(p) 
        
        # extract structual features
        structural_features = p.structural_features 
        
        # if analysing both hands in one video then assign time periods of dominant hand
        if len(hands)>1:
            if not p.hand_time_periods:
                time_periods = identify_active_time_period(structural_features, fs)    
            else:
                time_periods = p.hand_time_periods
        else:
            time_periods = None
            
        # loop over hands to analyse
        for hand in hands:
            
            # define feature
            f = feats[hand]
            
            # defining slice of time
            if time_periods:
                if len(time_periods[hand]) > 0:
                    start_frame = min([time_periods[hand][f_][0] for f_ in f])
                    end_frame = max([time_periods[hand][f_][1] for f_ in f])
                else:
                    print('Kinetic tremor of {} hand not found.'.format(hand))
                    continue
            else:
                start_frame = 0
                end_frame = structural_features.shape[0]

            # subsample the structure features
            data = structural_features[f]
            
            # interpolate data if points are missing
            data = data.interpolate()
            
            data = data[start_frame:end_frame].dropna()                
            
            if plot:
                plt.figure()
            
            x = data.loc[:,f[0]]
            y = data.loc[:,f[1]]
            
            # detrend the data
            x = signal.detrend(x)
            y = signal.detrend(y)
            
            # filter the data to remove arm movements
            x = butter_bandpass_filter(x, 3, 12, fs, order=5)
            y = butter_bandpass_filter(y, 3, 12, fs, order=5)

            # standard spectrum
            f_x, P_x = spectrum(x, fs) 
            f_y, P_y = spectrum(y, fs) 
            
            # spectrogram
            f_spectrogram_x, S_x = spectrogram(x, fs)
            f_spectrogram_y, S_y = spectrogram(y, fs)
            
            # hilbert
            analytic_signal_x = hilbert(x)
            instantaneous_amplitude_x = np.abs(analytic_signal_x)            
            analytic_signal_y = hilbert(y)
            instantaneous_amplitude_y = np.abs(analytic_signal_y)                   
            
            # plotting the tracking to ensure it works well
            if plot:
                plt.plot(x);
                plt.title(p.patient_id + '_' + hand)          
                if save_plots:
                    plt.savefig('./plots/kinetic_tremor_{}_{}.svg'.format(p.patient_id,hand))
    

            # hilbert features
            max_hilbert_amplitude_x = instantaneous_amplitude_x.max()
            max_hilbert_amplitude_y = instantaneous_amplitude_y.max()
            features_df.loc[p.patient_id, 'hilbert_max_amplitude_{}_hand'.format(hand)] = 2*np.sqrt(max_hilbert_amplitude_x**2 + max_hilbert_amplitude_y**2) 
            
            mean_hilbert_amplitude_x = instantaneous_amplitude_x.mean()
            mean_hilbert_amplitude_y = instantaneous_amplitude_y.mean()
            features_df.loc[p.patient_id, 'hilbert_mean_amplitude_{}_hand'.format(hand)] = 2*np.sqrt(mean_hilbert_amplitude_x**2 + mean_hilbert_amplitude_y**2)

            # spectrogram features
            mean_freq_x = f_spectrogram_x[S_x.mean(axis=0).argmax()] 
            mean_freq_y = f_spectrogram_y[S_y.mean(axis=0).argmax()] 
            features_df.loc[p.patient_id, 'spectrogram_mean_frequency_{}_hand'.format(hand)] = np.mean([mean_freq_x, mean_freq_y])
            
            max_freq_x = f_spectrogram_x[S_x.max(axis=0).argmax()]
            max_freq_y = f_spectrogram_y[S_y.max(axis=0).argmax()]
            features_df.loc[p.patient_id, 'spectrogram_max_frequency_{}_hand'.format(hand)] = np.mean([max_freq_x, max_freq_y])

            max_amplitude_x = S_x.max()
            max_amplitude_y = S_y.max()
            features_df.loc[p.patient_id, 'spectrogram_max_amplitude_{}_hand'.format(hand)] = 2*np.sqrt(max_amplitude_x**2 + max_amplitude_y**2)

            # power spectrum features
            if np.argmax([P_x.max(), P_y.max()])==0:
                dom_f_idx = P_x.argmax()
                dominant_frequency = f_x[dom_f_idx]
            else:
                dom_f_idx = P_y.argmax()
                dominant_frequency = f_y[dom_f_idx]                  
            features_df.loc[p.patient_id, 'power_spectral_dominant_frequency_{}_hand'.format(hand)] = dominant_frequency
            
            amplitude_x = P_x[dom_f_idx]
            amplitude_y = P_y[dom_f_idx]
            features_df.loc[p.patient_id, 'power_spectral_max_amplitude_{}_hand'.format(hand)] =  2*np.sqrt(amplitude_x**2 + amplitude_y**2)


    return features_df


def extract_postural_tremor_features(pc, plot=False, save_plots=False):
    
    
    # define markers for analysing postural tremor
    feats ={'right':['marker_middle_finger_tip_right_x','marker_middle_finger_tip_right_y'],
            'left':['marker_middle_finger_tip_left_x','marker_middle_finger_tip_left_y'],
            }    
    
    features_df = pd.DataFrame(index = pc.get_patient_ids())

    # loop over patients and extract tapping features
    for p in pc.patients:
        
        # extract sampling frequency 
        fs = p.sampling_frequency
        
        # identify tracked hand
        hands = check_hand_(p) 
        
        # extract structual features
        structural_features = p.structural_features 
        
        # if analysing both hands in one video then assign time periods of dominant hand
        if len(hands)>1:
            if not p.hand_time_periods:
                time_periods = identify_active_time_period(structural_features, fs)    
            else:
                time_periods = p.hand_time_periods
        else:
            time_periods = None
            
        # loop over hands to analyse
        for hand in hands:
            
            # define feature
            f = feats[hand]
            
            # defining slice of time
            if time_periods:
                if len(time_periods[hand]) > 0:
                    start_frame = min([time_periods[hand][f_][0] for f_ in f])
                    end_frame = max([time_periods[hand][f_][1] for f_ in f])
                else:
                    print('Postural tremor of {} hand not found.'.format(hand))
                    continue
            else:
                start_frame = 0
                end_frame = structural_features.shape[0]

            # subsample the structure features
            data = structural_features[f]
            
            # interpolate data if points are missing
            data = data.interpolate()
            
            data = data[start_frame:end_frame].dropna()                
            
            if plot:
                plt.figure()
            
            x = data.loc[:,f[0]]
            y = data.loc[:,f[1]]
            
            # detrend the data
            x = signal.detrend(x)
            y = signal.detrend(y)

            # filter the data to remove arm movements
            x = butter_bandpass_filter(x, 3, 12, fs, order=9)
            y = butter_bandpass_filter(y, 3, 12, fs, order=9)

            # standard spectrum
            f_x, P_x = spectrum(x, fs) 
            f_y, P_y = spectrum(y, fs) 
            
            # spectrogram
            f_spectrogram_x, S_x = spectrogram(x, fs)
            f_spectrogram_y, S_y = spectrogram(y, fs)
            
            # hilbert
            analytic_signal_x = hilbert(x)
            instantaneous_amplitude_x = np.abs(analytic_signal_x)            
            analytic_signal_y = hilbert(y)
            instantaneous_amplitude_y = np.abs(analytic_signal_y)  
            
            # plotting the tracking to ensure it works well
            if plot:
                plt.plot(x);
                plt.title(p.patient_id + '_' + hand)          
                if save_plots:
                    plt.savefig('./plots/kinetic_tremor_{}_{}.svg'.format(p.patient_id,hand))    

            # hilbert features
            max_hilbert_amplitude_x = instantaneous_amplitude_x.max()
            max_hilbert_amplitude_y = instantaneous_amplitude_y.max()
            features_df.loc[p.patient_id, 'hilbert_max_amplitude_{}_hand'.format(hand)] = 2*np.sqrt(max_hilbert_amplitude_x**2 + max_hilbert_amplitude_y**2) 
            
            mean_hilbert_amplitude_x = instantaneous_amplitude_x.mean()
            mean_hilbert_amplitude_y = instantaneous_amplitude_y.mean()
            features_df.loc[p.patient_id, 'hilbert_mean_amplitude_{}_hand'.format(hand)] = 2*np.sqrt(mean_hilbert_amplitude_x**2 + mean_hilbert_amplitude_y**2)

            # spectrogram features
            mean_freq_x = f_spectrogram_x[S_x.mean(axis=0).argmax()] 
            mean_freq_y = f_spectrogram_y[S_y.mean(axis=0).argmax()] 
            features_df.loc[p.patient_id, 'spectrogram_mean_frequency_{}_hand'.format(hand)] = np.mean([mean_freq_x, mean_freq_y])
            
            max_freq_x = f_spectrogram_x[S_x.max(axis=0).argmax()]
            max_freq_y = f_spectrogram_y[S_y.max(axis=0).argmax()]
            features_df.loc[p.patient_id, 'spectrogram_max_frequency_{}_hand'.format(hand)] = np.mean([max_freq_x, max_freq_y])

            max_amplitude_x = S_x.max()
            max_amplitude_y = S_y.max()
            features_df.loc[p.patient_id, 'spectrogram_max_amplitude_{}_hand'.format(hand)] = 2*np.sqrt(max_amplitude_x**2 + max_amplitude_y**2)

            # power spectrum features
            if np.argmax([P_x.max(), P_y.max()])==0:
                dom_f_idx = P_x.argmax()
                dominant_frequency = f_x[dom_f_idx]
            else:
                dom_f_idx = P_y.argmax()
                dominant_frequency = f_y[dom_f_idx]                  
            features_df.loc[p.patient_id, 'power_spectral_dominant_frequency_{}_hand'.format(hand)] = dominant_frequency
            
            amplitude_x = P_x[dom_f_idx]
            amplitude_y = P_y[dom_f_idx]
            features_df.loc[p.patient_id, 'power_spectral_max_amplitude_{}_hand'.format(hand)] =  2*np.sqrt(amplitude_x**2 + amplitude_y**2)


    return features_df


def extract_proximal_tremor_features(pc, plot=False, save_plots=False):
    
    
    # define markers for analysing proximal tremor at elbow
    feats ={'right':['marker_right_elbow_x','marker_right_elbow_y'],
            'left':['marker_left_elbow_x','marker_left_elbow_y'],
            }    
    
    features_df = pd.DataFrame(index = pc.get_patient_ids())

    # loop over patients and extract tapping features
    for p in pc.patients:
        
        # extract sampling frequency 
        fs = p.sampling_frequency
        
        # identify tracked hand
        hands = check_hand_(p) 
        
        # extract structual features
        structural_features = p.structural_features 
        
        # if analysing both hands in one video then assign time periods of dominant hand
        if len(hands)>1:
            if not p.hand_time_periods:
                time_periods = identify_active_time_period(structural_features, fs)    
            else:
                time_periods = p.hand_time_periods
        else:
            time_periods = None
            
        # loop over hands to analyse
        for hand in hands:
            
            # define feature
            f = feats[hand]
            
            # defining slice of time
            if time_periods:
                if len(time_periods[hand]) > 0:
                    start_frame = min([time_periods[hand][f_][0] for f_ in f])
                    end_frame = max([time_periods[hand][f_][1] for f_ in f])
                else:
                    print('Proximal tremor of {} hand not found.'.format(hand))
                    continue
            else:
                start_frame = 0
                end_frame = structural_features.shape[0]

            # subsample the structure features
            data = structural_features[f]
            
            # interpolate data if points are missing
            data = data.interpolate()
            
            data = data[start_frame:end_frame].dropna()                
            
            if plot:
                plt.figure()
            
            x = data.loc[:,f[0]]
            y = data.loc[:,f[1]]
            
            # detrend the data
            x = signal.detrend(x)
            y = signal.detrend(y)
            
            # filter the data to remove arm movements
            x = butter_bandpass_filter(x, 3, 12, fs, order=5)
            y = butter_bandpass_filter(y, 3, 12, fs, order=5)

            # standard spectrum
            f_x, P_x = spectrum(x, fs) 
            f_y, P_y = spectrum(y, fs) 
            
            # spectrogram
            f_spectrogram_x, S_x = spectrogram(x, fs)
            f_spectrogram_y, S_y = spectrogram(y, fs)
            
            # hilbert
            analytic_signal_x = hilbert(x)
            instantaneous_amplitude_x = np.abs(analytic_signal_x)            
            analytic_signal_y = hilbert(y)
            instantaneous_amplitude_y = np.abs(analytic_signal_y)        
            
            
            # plotting the tracking to ensure it works well
            if plot:
                plt.plot(x);
                plt.title(p.patient_id + '_' + hand)          
                if save_plots:
                    plt.savefig('./plots/proximal_tremor_{}_{}.svg'.format(p.patient_id,hand))
    

            # hilbert features
            max_hilbert_amplitude_x = instantaneous_amplitude_x.max()
            max_hilbert_amplitude_y = instantaneous_amplitude_y.max()
            features_df.loc[p.patient_id, 'hilbert_max_amplitude_{}_elbow'.format(hand)] = 2*np.sqrt(max_hilbert_amplitude_x**2 + max_hilbert_amplitude_y**2) 
            
            mean_hilbert_amplitude_x = instantaneous_amplitude_x.mean()
            mean_hilbert_amplitude_y = instantaneous_amplitude_y.mean()
            features_df.loc[p.patient_id, 'hilbert_mean_amplitude_{}_elbow'.format(hand)] = 2*np.sqrt(mean_hilbert_amplitude_x**2 + mean_hilbert_amplitude_y**2)

            # spectrogram features
            mean_freq_x = f_spectrogram_x[S_x.mean(axis=0).argmax()] 
            mean_freq_y = f_spectrogram_y[S_y.mean(axis=0).argmax()] 
            features_df.loc[p.patient_id, 'spectrogram_mean_frequency_{}_elbow'.format(hand)] = np.mean([mean_freq_x, mean_freq_y])
            
            max_freq_x = f_spectrogram_x[S_x.max(axis=0).argmax()]
            max_freq_y = f_spectrogram_y[S_y.max(axis=0).argmax()]
            features_df.loc[p.patient_id, 'spectrogram_max_frequency_{}_elbow'.format(hand)] = np.mean([max_freq_x, max_freq_y])

            max_amplitude_x = S_x.max()
            max_amplitude_y = S_y.max()
            features_df.loc[p.patient_id, 'spectrogram_max_amplitude_{}_elbow'.format(hand)] = 2*np.sqrt(max_amplitude_x**2 + max_amplitude_y**2)

            # power spectrum features
            if np.argmax([P_x.max(), P_y.max()])==0:
                dom_f_idx = P_x.argmax()
                dominant_frequency = f_x[dom_f_idx]
            else:
                dom_f_idx = P_y.argmax()
                dominant_frequency = f_y[dom_f_idx]                  
            features_df.loc[p.patient_id, 'power_spectral_dominant_frequency_{}_elbow'.format(hand)] = dominant_frequency
            
            amplitude_x = P_x[dom_f_idx]
            amplitude_y = P_y[dom_f_idx]
            features_df.loc[p.patient_id, 'power_spectral_max_amplitude_{}_elbow'.format(hand)] =  2*np.sqrt(amplitude_x**2 + amplitude_y**2)


    return features_df
