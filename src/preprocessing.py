import numpy as np
from scipy.signal import butter, filtfilt

class preprocessing():
    def __init__(self, dataset) -> None:
        super.__init__(preprocessing)

    def remove_artifacts():
        ...
    
    def Butterworth_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist

        # Design the Butterworth band-pass filter
        b, a = butter(order, [low, high], btype='band')

        # Apply the filter to the data
        y = filtfilt(b, a, data)

        return y