"""Utils functions."""
import logging
import multiprocessing
import multiprocessing.pool
from scipy import signal
import pandas as pd
import numpy as np

L = logging.getLogger(__name__)


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):  # pylint: disable=no-self-use
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess



def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band',analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def pre_low_pass_filter(data,columns,fs,high_cut=1):
    for col in columns:
        x = data.loc[:, pd.IndexSlice[col, 'x']]
        x = butter_lowpass_filter(x,high_cut,fs)
        data.loc[:, pd.IndexSlice[col, 'x']] = x

        y = data.loc[:, pd.IndexSlice[col, 'y']]
        y = butter_lowpass_filter(y,high_cut,fs)
        data.loc[:, pd.IndexSlice[col, 'y']] = y

    return data

def pre_high_pass_filter(data,columns,fs,low_cut=1):
    for col in columns:
        x = data.loc[:, pd.IndexSlice[col, 'x']]
        x = butter_highpass_filter(x,low_cut,fs)
        data.loc[:, pd.IndexSlice[col, 'x']] = x

        y = data.loc[:, pd.IndexSlice[col, 'y']]
        y = butter_highpass_filter(y,low_cut,fs)
        data.loc[:, pd.IndexSlice[col, 'y']] = y

    return data