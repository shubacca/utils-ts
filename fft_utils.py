import numpy as np
from numpy import fft
import pandas as pd
from scipy import signal as sig
from cmath import phase
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

class FFTFeatureExtractor:
    """
    This class creates Fast Fourier Transform variables for time series analysis. Two methods can be used to obtain inverse FFT of the peaks:
    1. decompose_df_into_pure_freq(): decompose FFT into frequencies and amplitudes and phases manually
    2. ifft_transform(): call numpy's IFFT transform on array of freq peaks
    Second method results in much smaller amplitudes of final time-domain signal, while first method helps to visualize each of the different 
    frequencies that make up the signal and the period of cyclicity/seasonality. 
    """
    def __init__(self, series, time_series) -> None:
        """
        Parameters
        ----------
        series : [np.array or list of ints]
            the y-value that needs to be performed FFT on 
        time_series : [np.array or list of datetime64[ns]]
            the x-values of time 
        """
        self.series = series
        self.ts = time_series

    def plotter_T_or_F(self, x, y, label, xlim_max=None, T_or_F='T'):        
        """Plots the time and frequency signatures 

        Parameters
        ----------
        x : [np.array or list of datetime64[ns] or floats]
            Either time values or frequency values can be passed
        y : [np.array or list of floats]
            Either values of series or power of transformed series can be passed
        label : [str]
            Plot title
        xlim_max : [float], optional
            Sets the max limit of the x axis on the graph, by default None
        T_or_F : str, optional
            Time or frequency graph, by default 'T'
        """
        self.label = label
        if T_or_F == 'T':
            plt.plot(x[:xlim_max], y[:xlim_max], label=label)
            plt.title(label)
            plt.ylabel( '{}'.format(label) )
            plt.xlabel( 'Time' )
        else:
            mask = (x > 0) & (x <= xlim_max)
            plt.plot(x[mask], y[mask], label=label)
            plt.title(label)
            plt.ylabel( 'Amplitude' )
            plt.xlabel( 'Frequency (1/Min]' )

            self.peaks = sig.find_peaks(y[x >=0], prominence=10**4)[0]
            self.peak_freq =  x[self.peaks]
            self.peak_power = y[self.peaks]
            plt.plot(self.peak_freq, self.peak_power, 'ro')
            plt.xlim(0,xlim_max)
        plt.xticks(rotation=90)
        plt.tight_layout()

    def fft_transform(self, freqlim_max=0.02, timelim_max=60*24):
        """Creates FFT variables and plots for visualization on the column of interest

        Parameters
        ----------
        freqlim_max : float, optional
            Max limit of the frequency to be shown on x-axis, by default 0.02 1/min
        timelim_max : float, optional
            Max limit of the time to be shown on x-axis, by default 60*24 mins
        """
        self.fftOutput = fft.fft(self.series)
        self.power = np.abs(self.fftOutput)
        self.freq = fft.fftfreq(len(self.series))

        plt.figure(figsize=(14,6))

        ax1 = plt.subplot(2,2,1)
        self.plotter_T_or_F(self.ts, self.series, label='Original Time Signature',T_or_F='T', xlim_max=len(self.ts))
        ax1 = plt.subplot(2,2,2)
        self.plotter_T_or_F(self.ts, self.series, label='Original Time Zoomed to {} days'.format(timelim_max/60/24),T_or_F='T', xlim_max=timelim_max)
        ax2 = plt.subplot(2,2,3)
        self.plotter_T_or_F(self.freq, self.power, label='Transformed Frequency Signature',T_or_F='F', xlim_max=max(self.freq))
        ax3 = plt.subplot(2,2,4)
        self.plotter_T_or_F(self.freq, self.power, label='Transformed Frequency Zoomed to (0, {}]'.format(freqlim_max),T_or_F='F', xlim_max=freqlim_max)

    def frequency_table_viewer(self):
        """Views the frequency table with index of the freq peaks, the freq peaks' values (1/min), the height/amplitude and the corresponding period in days
        generated after running fft_transform() function

        Returns
        -------
        [pd.Dataframe]
            Output dataframe is returned with index, top frequencies, amplitude, period and fft complex values
        """
        output = pd.DataFrame()
        output['index'] = self.peaks
        output['freq (1/min)'] = self.peak_freq
        output['amplitude'] = self.peak_power
        output['period (days)'] = 1 / self.peak_freq / 60 / 24
        output['fft'] = self.fftOutput[self.peaks]
        output = output.sort_values('amplitude', ascending=False)
        self.output = output
        return output

    def fourier_terms_df_creator(self):
        """Creates df fourier_terms with top FFT frequencies (complex notation), freq values (1/min), amplitude and phase; also creates an
        internal dictionary of this dataframe stored as fourier_terms_dict

        Returns
        -------
        [pd.DataFrame]
            Returns dataframe with top FFT frequencies (complex notation), freq values (1/min), amplitude and phase
        """
        fourier_terms = pd.DataFrame()
        fourier_terms['fft'] = self.output['fft']
        fourier_terms['freq (1/min)'] = self.output['freq (1/min)']
        fourier_terms['amplitude'] = fourier_terms.fft.apply(lambda z: abs(z)) 
        fourier_terms['phase'] = fourier_terms.fft.apply(lambda z: phase(z))
        fourier_terms.sort_values(by=['amplitude'], ascending=[0])

        # Create some helpful labels (FT_1..FT_N)
        fourier_terms['label'] = list(map(lambda n : 'FT_{}'.format(n), range(1, len(fourier_terms) + 1)))
        
        # Turn our dataframe into a dictionary for easy lookup
        fourier_terms = fourier_terms.set_index('label')
        self.fourier_terms = fourier_terms
        fourier_terms_dict = fourier_terms.to_dict('index')
        self.fourier_terms_dict = fourier_terms_dict
        return fourier_terms

    def decompose_df_into_pure_freq(self, signal, time_min):
        """Creates columnar df with time steps in one column, the actual signal in second column, and the decomposition of the signal into varying
        sign wave values under each column FT_1, FT_2; also included is the sum of all individual values in final column called FT_All

        Parameters
        ----------
        signal : [np.array or list of floats]
            the y-value that needs to be performed FFT on
        time_min : [np.array or list of datetime64[ns]]
            the x-values of time 

        Returns
        -------
        [pd.DataFrame]
            Returns columnar dataframe as mentioned above
        """
        data = pd.DataFrame()
        data['pass_count_standardized'] = signal
        data['time_min'] = time_min
        for key in self.fourier_terms_dict.keys():
            a = self.fourier_terms_dict[key]['amplitude']  
            w = 2 * np.pi * (self.fourier_terms_dict[key]['freq (1/min)'] / 60) # units in 1/s
            p = self.fourier_terms_dict[key]['phase']    
            data[key] = data['time_min'].apply(lambda t: a * math.cos(w*t*60 + p))

        data['FT_All'] = 0
        for column in list(self.fourier_terms.index):
            data['FT_All'] = data['FT_All'] + data[column]
        
        self.pureFreqDF = data
        return data
    
    def ifft_transform(self):
        """Performs the inverse FFT from the np.fft package on the filtered FFT output (np.series of freq indexes with 0's except for those
        indexes where a peak was registered from FFT). 

        Returns
        -------
        [pd.DataFrame]
            Time-domain signature that represents the FT_All combined feature, however the amplitude is much smaller than manual FT_All method
        """
        self.filtered_fft_output = np.array([f if i in list(self.output['index']) else 0 for i, f in enumerate(self.fftOutput)])
        self.filtered_residuals = fft.ifft(self.filtered_fft_output)
        return self.filtered_residuals
