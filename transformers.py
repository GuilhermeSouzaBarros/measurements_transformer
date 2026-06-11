import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class DataSpliter(BaseEstimator, TransformerMixin):
    """
        Restruct the data so it becomes a list of windows of measured tensions around the target power measurement y in the time t

        Makes pondered means to account for time differences
    """

    def __init__(self, window_size:int=5, shift_us:int=0):
        self.window_size = window_size
        self.shift_us = shift_us

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
        # X = {timestamp, tension}
        # y = {timestamp, power}
        num_measurements = X.shape[0]

        full_window = 2 * self.window_size + 1
        X_time = X["timestamp"].to_numpy()
        X_time = X_time - X_time.min()
        X_tens = X["tension (V)"].to_numpy()
        new_X = np.empty((num_measurements - full_window + 1, full_window))
        time_gap_us = (X_time.max() - X_time.min()) / num_measurements
        
        i = self.window_size
        last_i = 0
        last_valid = num_measurements - self.window_size
        X_timestamp_ended = False
        target_time = time_gap_us * self.window_size
        
        while i < last_valid:
            for j in range(0, full_window): # -window_size : center : window_size
                while (X_time[last_i + 1] < target_time):
                    last_i += 1

                    if (last_i + 1 >= num_measurements):
                        X_timestamp_ended = True
                        new_X.resize((i - self.window_size, full_window))
                        break

                if (X_timestamp_ended): break

                # pondered mean with the measurement before and after the target time
                weight_high = (target_time - X_time[last_i]) / (X_time[last_i+1] - X_time[last_i])
                weight_low = 1 - weight_high
                
                new_X[i - self.window_size, j] = (
                    weight_low  * X_tens[last_i] +
                    weight_high * X_tens[last_i+1]
                )

            if (X_timestamp_ended): break

            target_time += time_gap_us
            i += 1

        return new_X
