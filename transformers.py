import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
import scipy.signal as signal
import scipy.interpolate as interpolate

class AmExtractor(BaseEstimator, TransformerMixin):
    """
        Extracts features according to signal amplitude
    """
    def __init__(self, frequency:int=60, max_harmonics=8):
        self.frequency = frequency
        self.max_harmonics = max_harmonics

    def get_harmonics(self, frequencies, psd, num_harmonics):
        harmonic_x = []
        harmonic_y = []
        i = 0
        j = 0
        next_f = self.frequency
        while i < num_harmonics:
            if (frequencies[j] >= next_f):
                if (frequencies[j] - next_f < next_f - frequencies[j-1]):
                    harmonic_x.append(frequencies[j])
                    harmonic_y.append(psd[j])
                else:
                    harmonic_x.append(frequencies[j-1])
                    harmonic_y.append(psd[j-1])
                    
                next_f += self.frequency
                i += 1
            j += 1
        return harmonic_x, harmonic_y

    def plot(self, x, y):
        plt.figure(figsize=(10, 5))
        plt.semilogy(x, y, color='blue', linewidth=1.5)
        plt.title('Power Spectral Density (PSD)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V²/Hz]')
        plt.grid(True, which="both")
        plt.show()

    def fit(self, X, y=None):
        pass
        
    def transform(self, X, y=None):
        # Amplitude features
        new_X = []
        for x in X:
            new_x = []
            for sub_x in x:
                frequencies, psd = signal.welch(sub_x["measurement"], sub_x["sample_rate"], nperseg=min(len(sub_x["measurement"]), 256))

                num_harmonics = min(int(sub_x["sample_rate"] / (2*self.frequency)), self.max_harmonics)
                harmonic_x, harmonic_y = self.get_harmonics(frequencies, psd, num_harmonics)

                new_x.append(harmonic_y)
            new_X.append(new_x)
        return new_X



class PWExtractor(BaseEstimator, TransformerMixin):
    """
        Filters the data according to amplitude, frequencie and pulse-width features
    """
    def __init__(self, switching_frequency=20.000, filter_order=4):
        self.switching_frequency = switching_frequency
        self.filter_order = filter_order

    def plot(self, x, y):
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, color='blue', linewidth=1.5)
        plt.title('Filtered Signal Measurement')
        plt.xlabel('Tension (V)')
        plt.ylabel('Time (s)')
        plt.grid(True, which="both")
        plt.show()

    def low_filter(self, X, sampling_rate):
        cutoff = 2 * self.switching_frequency / sampling_rate
        b, a = signal.butter(self.filter_order, cutoff, btype="low")
        return signal.filtfilt(b, a, X)

    def fit(self, X, y=None):
        pass
        
    def transform(self, X, y=None):
        # Amplitude features
        filtered_measurements = self.low_filter(X["measurement"], X["sample_rate"])
        self.plot(X["time"], filtered_measurements)


class UniformTimeInterval(BaseEstimator, TransformerMixin):
    """
        Interpolates the data so it intervals become the same
    """
    def __init__(
            self,
            measurement="tension (V)",
            min_measurement=-1.65,
            max_measurement=1.65
        ):
        self.measurement = measurement
        self.min_measurement = min_measurement
        self.max_measurement = max_measurement

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        new_X = []
        for x in X:
            tensions = x[self.measurement].to_numpy()
            timestamps = x["timestamp"].to_numpy()
            timestamps = (timestamps - timestamps[0]) / 1000000

            sample_rate = timestamps.shape[0] / timestamps[-1]

            uniform_time = np.linspace(0, timestamps[-1], len(timestamps))
            tensions_uniform = interpolate.interp1d(timestamps, tensions, kind='cubic')(uniform_time)
            new_X.append({
                "measurement":  tensions_uniform,
                "time":         uniform_time,
                "sample_rate":  sample_rate    
            })
        return new_X


class DataSpliter(BaseEstimator, TransformerMixin):
    """
        Restruct the data so it becomes a list of values in the intervals
    """
    def __init__(self, window_size_ms:int=100, shift_ms:int=0):
        self.window_size_ms = window_size_ms
        self.shift_ms = shift_ms

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        new_X = []
        for x in X:
            window_measurements = int((self.window_size_ms / 1000) * x["sample_rate"])
            num_of_measurements = x["time"].shape[0]

            new_x = []
            i = 0
            n_i = window_measurements
            while n_i < num_of_measurements:
                new_x.append(
                    {
                        "measurement":  x["measurement"][i:n_i],
                        "time":         x["time"][i:n_i],
                        "sample_rate":  x["sample_rate"]
                    }
                )
        
                i = n_i
                n_i += window_measurements
            new_X.append(new_x)

        return new_X
