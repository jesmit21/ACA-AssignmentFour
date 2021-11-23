### Tuning Frequency Estimation and Key Detection ###
### Fall 2021 ###
### Professior Alexander Lerch ###

import scipy.fft as fft
import numpy as np
from scipy.io import wavfile
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import glob

def block_audio(x,blockSize,hopSize,fs):
    i = 0
    #Instantiate first block
    xb = x[i:i+blockSize]
    timeInSec = np.array([0])
    i = i + hopSize
    #Create the rest of the blocks and append to arrays
    while i<x.size:
        tempblock = x[i:i+blockSize]
        #0 Pad the last block
        if tempblock.size<blockSize:
            tempblock = np.pad(tempblock,int(np.floor((blockSize-tempblock.size)/2)))
            if tempblock.size+1 == blockSize:
                tempblock = np.append(tempblock,[0])
        timeInSec = np.append(timeInSec,[i/fs])
        xb = np.vstack([xb,tempblock])
        i = i + hopSize
    return [xb, timeInSec]


# 2b: detecting key:

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
        key_index = np.where(key_per_block == numpy.amax(key_per_block))
        keyEstimate=key_index[0]+key_index[1]*12
        return keyEstimate



    return keyEstimate

def evaluate(pathToAudioKey,pathToGTKey,pathToAudioTf,pathToGTTf):
    import numpy as np
    import os
    avg_accuracy = eval_key_detection(pathToAudioKey,pathToGTKey)
    avg_deviationInCent = eval_tfe(pathToAudioTf,pathToGTTf)
    
    #avg_accuracy = 0
    #avg_deviationInCent = 0
    #for file in os.listdir(pathToAudioKey):
    #    deviationVec = []
     #   xb = block_audio(file,4096,2048,44100)
    #    mean1 = np.mean(xb)
    #    deviation1 = np.sqrt(np.mean((xb - mean1)**2))
    #    deviationVec = np.concatenate(deviationVec,deviation1)
    #avg_deviationInCent = np.mean(deviationVec)

    return(avg_accuracy,avg_deviationInCent)
