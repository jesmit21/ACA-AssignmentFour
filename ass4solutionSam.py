import scipy as sp
import scipy.fft as fft
from scipy import signal as sig
import numpy as np
from scipy.io import wavfile
from scipy import stats
import matplotlib.pyplot as plt
import math
def  block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)
def hann_window(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * 
np.arange(iWindowLength)))
def compute_spectrogram(xb,fs):
    numBlocks = xb.shape[0]
    afWindow = hann_window(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 

        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2)
    fInHz = np.zeros(X.shape[0])
    for n in range(0,X.shape[0]):
        fInHz[n] =n*fs/xb.shape[1]
    return X,fInHz
def get_spectral_peaks(X):
    peaks = np.zeros((20,X.shape[1]))
    for i in range(X.shape[1]):
        #Each peak is the location in array f that contains the largest amp'd frequencies
        peaks[:,i] = np.argsort(X[:,i])[X.shape[0]-21:X.shape[0]-1]
    return peaks
def estimate_tuning_freq(x,blockSize,hopSize,fs):
    ########
    #What is happening here is that I am creating an array of equally tempered pitches by concatenating 
    #multiples of 12root(2) into a large array.
    tunedpoints = np.array([440.0])   
    while tunedpoints[tunedpoints.size-1] > 1:
        tunedpoints = np.concatenate((tunedpoints,[tunedpoints[tunedpoints.size-1]/(2**(1/12))]))
    while tunedpoints[tunedpoints.size-1] < fs/2:
        tunedpoints = np.concatenate((tunedpoints,[tunedpoints[tunedpoints.size-1]*(2**(1/12))]))
    #######
    xb,t = block_audio(x,blockSize,hopSize,fs)
    X,f = compute_spectrogram(xb,fs)
    q = get_spectral_peaks(X)
    histogram = np.zeros(21)
    closecent = np.arange(-50,50.1,5)
    #Iterate through our SPeak array, getting the diffincent for each value
    for i in range(q.shape[1]):
        for j in range(q.shape[0]):
            closeid = (np.abs(tunedpoints - f[int(q[j,i])])).argmin()
            diffincent = 1200*np.log2(f[int(q[j,i])]/tunedpoints[closeid])
            #Find the closest cent difference in our histogram, and count up the occurences
            #Histogram goes from -50,50 cents with a band width of 5
            histogram[(np.abs(closecent - diffincent)).argmin()] += 1  
    #Return our tuning frequency, which is the most present difference in cents modulating a 0 cent tuning of 440
    return 440*2**(closecent[np.argmax(histogram)]/1200)
t = np.linspace(0,1,44100)
x = np.sin(2*np.pi*t*666)
for i in range(30):
    x = x + np.sin(2*np.pi*t*666*(i+1))
print(estimate_tuning_freq(x,20000,10000,44100))


