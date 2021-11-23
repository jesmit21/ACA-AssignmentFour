### Assignment 4 Solution (MASTER)

import scipy.fft as fft
import numpy as np
from scipy.io import wavfile
from scipy.spatial import distance
import math
import os

# Extra Functions (not directly graded in assignment)

def block_audio(x,blockSize,hopSize,fs):
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
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

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

###### -- ######
###### A1 ######
###### -- ######

def get_spectral_peaks(X):
    peaks = np.zeros((20,X.shape[1]))
    for i in range(X.shape[1]):
        #Each peak is the location in array f that contains the largest amp'd frequencies
        peaks[:,i] = np.argsort(X[:,i])[X.shape[0]-21:X.shape[0]-1]
    return peaks

###### -- ######
###### A2 ######
###### -- ######

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

###### -- ######
###### B1 ######
###### -- ######

# INSERT CODE HERE JOEY

###### -- ######
###### B2 ######
###### -- ######

def detect_key(x, blockSize, hopSize, fs, bTune=False):
    if bTune==False:
        tfInHz=440
    else:
        tfInHz=estimate_tuning_freq(x, blockSize, hopSize, fs)
    pitchChroma=extract_pitch_chroma(x, fs, tfInHz)
    distance.euclidean([1, 0, 0], [0, 1, 0])
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    maj_pc=t_pc[0]/(sum(t_pc[0])) # normalize major key chroma so the sum of all the pitch classes =1
    min_pc=t_pc[1]/(sum(t_pc[1])) # normalize minor key chroma so the sum of all the pitch classes =1
    key_per_block=np.zeros(12,2) #creates a matrix to count the total numbers of times a key is detected in the blocks
        # row 0 is major, row 1 is minor 

    for m in pitchChroma: # looping though the pitch chroma for each block
        block_chroma=m[1]/(sum(m[1])) # normalize block chroma so the sum of all the pitch classes =1
        closest_maj=0 # number of shifts for the closest major key
        closestdist_maj=999999 # euclidean distance for the closest major key
        closest_min=0 # number of shifts for the closest minor key
        closestdist_min=999999 # euclidean distance for the closest minor key
        for i in range(len(maj_pc)):
            maj_shifted=np.roll(maj_pc, i) 
            min_shifted=np.roll(min_pc, i) 
            maj_dist=distance.euclidean(block_chroma, maj_shifted)
            min_dist=distance.euclidean(block_chroma, min_shifted)
            if maj_dist<closestdist_maj:
                closest_maj=i
                closestdist_maj=maj_dist
            if min_dist<closestdist_min:
                closest_min=i
                closestdist_min=min_dist
        if closestdist_min>closestdist_maj:
            key_per_block[closest_min,1]+=1
        else:
            key_per_block[closest_maj,0]+=1
        key_index = np.where(key_per_block == np.amax(key_per_block))
        keyEstimate=key_index[0]+key_index[1]*12
        return keyEstimate

###### -- ######
###### C1 ######
###### -- ######

def eval_tfe(pathToAudio, pathToGT):
    
    # Setting evaluation block size and hop size
    blockSize = 4096
    hopSize = 2048

    # Initialization
    numOfFiles = 0
    truthArr = np.empty(0, int)
    estTuningArr = np.empty(0, int)

    # Read in ground truth labels for key
    for file in os.listdir(pathToGT):
        truthVal = np.loadtxt(pathToGT + file)
        truthArr = np.append(truthArr, truthVal)

    # Read in audio from folder
    for file in os.listdir(pathToAudio):

        numOfFiles += 1
        
        # Make sure path is correct
        [fs, x] = wavfile.read(pathToAudio + file)

        # Estimated key with tuning frequency correction
        estTuning = estimate_tuning_freq(x, blockSize, hopSize, fs)
        estTuningArr = np.append(estTuningArr, estTuning)

    # Calculate the deviation between the ground truth and calculated tuning frequency
    diffArr = np.abs(truthArr - estTuningArr)

    # Calculate total mean of deviation
    diffArrMean = np.mean(diffArr)

    # Calculate average absolute deviation
    summation = 0
    for i in range(len(diffArr)):
        deviation = abs(diffArr - diffArrMean)
        summation = summation + deviation

    avgDeviation = summation/len(diffArr)

    return avgDeviation

###### -- ######
###### C2 ######
###### -- ######

def eval_key_detection(pathToAudio, pathToGT):

    # Setting evaluation block size and hop size
    blockSize = 4096
    hopSize = 2048

    # Initialization
    numOfFiles = 0
    truthArr = np.empty(0, int)
    estTuningTrue = np.empty(0, int)
    estTuningFalse = np.empty(0, int)

    # Read in ground truth labels for key
    for file in os.listdir(pathToGT):
        truthVal = np.loadtxt(pathToGT + file)
        truthArr = np.append(truthArr, int(truthVal))

    # Read in audio from folder
    for file in os.listdir(pathToAudio):

        numOfFiles += 1
        
        # Make sure path is correct
        [fs, x] = wavfile.read(pathToAudio + file)

        # Estimated key with tuning frequency correction
        tuningTrue = detect_key(x, blockSize, hopSize, fs, bTune = True)
        estTuningTrue = np.append(estTuningTrue, int(tuningTrue))

        # Estimated key without tuning frequency correction
        tuningFalse = detect_key(x, blockSize, hopSize, fs, bTune = False)
        estTuningFalse = np.append(estTuningFalse, int(tuningFalse))

    # Count how many key estimations are correct
    # To do this, we subtract the two arrays and find how many zeros there are.
    numMatchesTrue = np.count_nonzero((truthArr - estTuningTrue) == 0)
    numMatchesFalse = np.count_nonzero((truthArr - estTuningFalse) == 0)

    # Make a 2x1 vector with: Accuracy = (# of Correct) / (# of Songs)
    tuningTrueAcc = numMatchesTrue/numOfFiles
    tuningFalseAcc = numMatchesFalse/numOfFiles
    accuracy = np.array([tuningTrueAcc, tuningFalseAcc])

    return accuracy

###### -- ######
###### C3 ######
###### -- ######

def evaluate(pathToAudioKey,pathToGTKey,pathToAudioTf,pathToGTTf):
    
    avg_accuracy = eval_key_detection(pathToAudioKey,pathToGTKey)
    avg_deviationInCent = eval_tfe(pathToAudioTf,pathToGTTf)

    return(avg_accuracy, avg_deviationInCent)
