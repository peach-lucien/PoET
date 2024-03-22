import numpy as np


def check_hand_(patient):
    
    # extract labels for recording
    conditions = patient.label
    
    # check which hand has been tracked
    if conditions:
        if 'hand' not in conditions.keys():
            hand=['left','right']
        elif isinstance(conditions['hand'],list): 
            hand = conditions['hand']                
        else:
            hand = [conditions['hand']]
    else:        
        hand = ['left','right']
    
    return hand


def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        iax = ia[i]
        
        idx_start = np.where(iax == True)[0][z[np.where(iax == True)[0]].argmax()]
        frame_start = p[idx_start] # largest sequence of true first index
        
        if p.shape[0]==idx_start+1:
            frame_end = n
        else:
            frame_end = p[idx_start+1] # largest sequence of true first index
    
        return frame_start, frame_end
    

def identify_active_time_period(structural_features, fs):
    
    time_periods = {'left':{}, 'right':{}}
    
    if 'time' in structural_features.columns:
        structural_features = structural_features.drop('time',axis=1)
    
    for hand in time_periods.keys():
        features = [f for f in structural_features.columns if hand in f]
        
        for feat in features:
            
            if not structural_features.loc[:,feat].isna().any():
            
                amplitudes, frequencies = compute_spectrogram(structural_features.loc[:,feat], fs)
                spectrogram = np.vstack(amplitudes).T
                
                above_threshold = spectrogram.mean(axis=0) > np.median(spectrogram)
                frame_start, frame_end = rle(above_threshold)
            
                time_periods[hand][feat] = (frame_start, frame_end)        
            
    return time_periods


def compute_spectrogram(x, fs, min_freq=0, max_freq=100):
    
    x = np.pad(x,(int(fs/2),int(fs/2)),mode='symmetric')   
    
    #spectrums = []
    amplitudes = []
    frequencies = []    

    for i in range(len(x)-int(fs)+1): 
   
        xs = x[i:i+int(fs)]
        n_samples = len(xs)
        amplitude = 2/n_samples * abs(np.fft.fft(xs))
        amplitude = amplitude[1:int(len(amplitude)/2)]     
        
        frequency  = (np.fft.fftfreq(n_samples) * n_samples * 1 / (1/fs*len(xs)))
        frequency = frequency[1:int(len(frequency)/2)]   

        freq_limits = (min_freq<=frequency) & (max_freq>=frequency)
        amplitude = amplitude[freq_limits]
        frequency = frequency[freq_limits]        
        
        amplitudes.append(amplitude)
        frequencies.append(frequencies)
        
    return amplitudes, frequencies

def meanfilt(x, k):
    """Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    
    import numpy as np

    assert k % 2 == 1, "Mean filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.mean (y, axis=1)

def filter_peaks_and_troughs(peaks, troughs, signal):
    filtered_peaks = []
    filtered_troughs = []
    i, j = 0, 0

    while i < len(peaks) and j < len(troughs):
        current_peak = peaks[i]
        current_trough = troughs[j]

        # If there are sequential peaks, find the highest one
        while i < len(peaks) - 1 and peaks[i + 1] < current_trough:
            if signal[peaks[i + 1]] > signal[current_peak]:
                current_peak = peaks[i + 1]
            i += 1

        # If there are sequential troughs, find the lowest one
        while j < len(troughs) - 1 and troughs[j + 1] < current_peak:
            if signal[troughs[j + 1]] < signal[current_trough]:
                current_trough = troughs[j + 1]
            j += 1

        # Append the highest peak and lowest trough found in the sequence
        filtered_peaks.append(current_peak)
        filtered_troughs.append(current_trough)

        # Move to the next unprocessed peak and trough
        i = np.searchsorted(peaks, current_trough, side='right')
        j = np.searchsorted(troughs, current_peak, side='right')

    return np.unique(peaks), np.unique(filtered_troughs)

def ensure_peak_to_trough(peaks,troughs):
    
    if peaks[0] < troughs[0]:
        reverse = False
        x = peaks.tolist()
        y = troughs.tolist()
    else:
        reverse = True
        x = troughs.tolist()
        y = peaks.tolist()
        
    for r in range(100):
        for i in range(len(x)-1):
            try:          
                if x[i+1] < y[i]:
                    x.pop(i+1)
            except:
                continue
            
        for i in range(len(y)-1):
            try:          
                if y[i+1] < x[i+1]:
                    y.pop(i+1)
            except:
                continue  
            
    min_idx = np.min([len(x),len(y)])
    x = x[:min_idx]
    y = y[:min_idx]

    if reverse:
        return np.array(y), np.array(x)
    else:
        return np.array(x), np.array(y)
        