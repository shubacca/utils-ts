import numpy as np
import matplotlib.pyplot as plt

class TSGenerator:
    def __init__(self, time, baseline=10, amplitude=40, slope=0.05, noise_level=5, period=365) -> None:
        if time is None: 
            time = np.arange(4 * 365 + 1, dtype='float32')
        self.time = time
        self.baseline = baseline
        self.amplitude = amplitude
        self.slope = slope
        self.noise_level = noise_level
        self.period = period
        
    def plot_series(self, time, series, format="-", start=0, end=None):
        if time is None: 
            plt.plot(self.time[start:end], series[start:end], format)
        else: 
            plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)

    def trend(self):
        return self.slope * self.time

    def seasonal_pattern(self, season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.4,
                        np.cos(season_time * 2 * np.pi),
                        1 / np.exp(3 * season_time))

    def seasonality(self, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((self.time + phase) % self.period) / self.period
        return self.amplitude * self.seasonal_pattern(season_time)

    def noise(self, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(self.time)) * self.noise_level