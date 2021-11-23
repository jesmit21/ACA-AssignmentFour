### Tuning Frequency Estimation and Key Detection ###
### Fall 2021 ###
### Professior ALexander Learch ###

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
