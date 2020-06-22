#!/usr/bin/env python
# coding: utf-8

import sys
import time
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from scipy import fftpack as fp
from sklearn.cluster import KMeans
from museval.metrics import bss_eval_images, bss_eval_sources

### Function for audio pre-processing ###
def pre_processing(data, Fs, down_sam):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Down sampling and normalization of the wave
    if down_sam is not None:
        wavdata = sg.resample_poly(wavdata, down_sam, Fs)
        Fs = down_sam
    
    return wavdata, Fs

### Function for getting STFT ###
def get_STFT(wav, Fs, frame_length, frame_shift):
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    freqs, times, dft = sg.stft(wav, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    arg = np.angle(dft) #Preserve the phase
    Adft = np.abs(dft) #Preserve the absolute amplitude
    Y = Adft
    
    #Display the size of input
    print("Spectrogram size (freq, time) = " + str(Y.shape))
    
    return Y, arg, Fs, freqs, times

### Function for getting inverse STFT ###
def get_invSTFT(Y, arg, Fs, frame_length, frame_shift):
    
    #Restrive the phase from original wave
    Y = Y * np.exp(1j*arg)
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function for removing components closing to zero ###
def get_nonzero(tensor):
    
    tensor = np.where(np.abs(tensor) < 1e-10, 1e-10+tensor, tensor)
    return tensor

### Function for computing numerator of temporal continuity term ###
def continuity_numer(U):
    
    #Get the value at the start and end point in U
    start = U[:, 0][:, np.newaxis]
    end = U[:, -1][:, np.newaxis]
    
    #Get summation of squared U
    U2 = np.sum(U**2, axis=1, keepdims=True)
    
    #Compute the first term
    term1 = (np.append(U, end, axis=1) - np.append(start, U, axis=1))**2
    term1 = U * np.sum(term1, axis=1, keepdims=True) / get_nonzero(U2**2)
    
    #Compute the second term
    term2 = np.append(np.append(U, end, axis=1), end, axis=1)
    term2 = term2 + np.append(start, np.append(start, U, axis=1), axis=1)
    term2 = term2[:, 1:-1] / get_nonzero(U2)
    
    output = term1 + term2
    
    #Return numerator of temporal continuity term
    return output

### Function for computing denominator of temporal continuity term ###
def continuity_denom(U):
    
    output = U / get_nonzero(np.sum(U**2, axis=1, keepdims=True))
    return output

### Function for computing temporal continuity cost ###
def continuity_cost(U):
    
    #Get the value at the start and end point in U
    start = U[:, 0][:, np.newaxis]
    end = U[:, -1][:, np.newaxis]
    
    #Subtract adjacent values in U
    output = np.append(U, end, axis=1) - np.append(start, U, axis=1)
    
    #Get the sum of squares
    output = np.sum((output[:, 1:])**2, axis=1) / get_nonzero(np.sum(U**2, axis=1))
    output = np.sum(output)
    
    #Retern temporal continuity cost
    return output

### Function for getting basements and weights matrix by NMF ###
def get_NMF(Y, num_iter, num_base, loss_func, alpha, norm_H):
    
    #Initialize basements and weights based on the Y size(k, n)
    K, N = Y.shape[0], Y.shape[1]
    if num_base >= K or num_base >= N:
        print("The number of basements should be lower than input size.")
        sys.exit()
    
    #Remove Y entries closing to zero
    Y = get_nonzero(Y)
    
    #Initialize as random number
    H = np.random.rand(K, num_base) #basements (distionaries)
    U = np.random.rand(num_base, N) #weights (coupling coefficients)
    
    #Initialize loss
    loss = np.zeros(num_iter)
    
    #For a progress bar
    unit = int(np.floor(num_iter/10))
    bar = "#" + " " * int(np.floor(num_iter/unit))
    start = time.time()
    
    #In the case of squared Euclidean distance
    if loss_func == "EU":
        
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            
            #Update the basements
            X = H @ U
            H = H * (Y @ U.T) / get_nonzero(X @ U.T)
            #Normalize the basements
            if norm_H == True:
                H = H / H.sum(axis=0, keepdims=True)
            
            #Update the weights
            X = H @ U
            denom_U = H.T @ X + 4*alpha*N*continuity_denom(U)
            numer_U = H.T @ Y + 2*alpha*N*continuity_numer(U)
            U = U * numer_U / get_nonzero(denom_U)
            
            #Normalize to ensure equal energy
            if norm_H == False:
                A = np.sqrt(np.sum(U**2, axis=1)/np.sum(H**2, axis=0))
                H = H * A[np.newaxis, :]
                U = U / A[:, np.newaxis]
            
            #Compute the loss function
            X = H @ U
            loss[i] = np.sum((Y - X)**2)
            loss[i] = loss[i] + alpha*continuity_cost(U)
    
    #In the case of Kullback–Leibler divergence
    elif loss_func == "KL":
        
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            
            #Update the basements
            X = get_nonzero(H @ U)
            denom_H = U.T.sum(axis=0, keepdims=True)
            H = H * ((Y / X) @ U.T) / get_nonzero(denom_H)
            #Normalize the basements
            if norm_H == True:
                H = H / H.sum(axis=0, keepdims=True)
            
            #Update the weights
            X = get_nonzero(H @ U)
            denom_U = H.T.sum(axis=1, keepdims=True) + 4*alpha*N*continuity_denom(U)
            numer_U = H.T @ (Y / X) + 2*alpha*N*continuity_numer(U)
            U = U * numer_U / get_nonzero(denom_U)
            
            #Normalize to ensure equal energy
            if norm_H == False:
                A = np.sqrt(np.sum(U**2, axis=1)/np.sum(H**2, axis=0))
                H = H * A[np.newaxis, :]
                U = U / A[:, np.newaxis]
            
            #Compute the loss function
            X = get_nonzero(H @ U)
            loss[i] = np.sum(Y*np.log(Y) - Y*np.log(X) - Y + X)
            loss[i] = loss[i] + alpha*continuity_cost(U)
    
    #In the case of Itakura–Saito divergence
    elif loss_func == "IS":
            
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            
            #Update the basements
            X = get_nonzero(H @ U)
            denom_H = np.sqrt(X**-1 @ U.T)
            H = H * np.sqrt((Y / X**2) @ U.T) / get_nonzero(denom_H)
            #Normalize the basements (it is recommended when IS divergence)
            H = H / H.sum(axis=0, keepdims=True)
            
            #Update the weights
            X = get_nonzero(H @ U)
            denom_U = np.sqrt(H.T @ X**-1) + 4*alpha*N*continuity_denom(U)
            numer_U = np.sqrt(H.T @ (Y / X**2)) + 2*alpha*N*continuity_numer(U)
            U = U * numer_U / get_nonzero(denom_U)
            
            #Compute the loss function
            X = get_nonzero(X)
            loss[i] = np.sum(Y / X - np.log(Y) + np.log(X) - 1)
            loss[i] = loss[i] + alpha*continuity_cost(U)
    
    else:
        print("The deviation shold be either 'EU', 'KL', or 'IS'.")
        sys.exit()
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_iter/unit))
    print("\rNMF:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, i+1, num_iter, time.time()-start), end="")
    print()
    
    return H, U, loss

### Function for plotting Spectrogram and loss curve ###
def display_graph(Y, X, times, freqs, loss_func, num_iter):
    
    #Plot the original spectrogram
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title('An original spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    Y = 10*np.log10(np.abs(Y))
    plt.pcolormesh(times, freqs, Y, cmap='jet')
    plt.colorbar(orientation='horizontal').set_label('Power')
    plt.savefig("./log/original_spec.png", dpi=200)
    
    #Plot the approximated spectrogram
    plt.subplot(1, 2, 2)
    plt.title('The spectrogram approximated by NMF')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    X = 10*np.log10(np.abs(X))
    cm = plt.pcolormesh(times, freqs, X, cmap='jet', vmin=np.min(Y), vmax=np.max(Y))
    plt.colorbar(cm, orientation='horizontal').set_label('Power')
    plt.savefig("./log/reconstructed_spec.png", dpi=200)
    
    #Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, num_iter+1), loss[:], marker='.')
    plt.title(loss_func + '_loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.savefig("./log/loss_curve.png", dpi=200)
    
    return

### Function for generating Mel-scale filters ###
def melFilterBank(Fs, fftsize, Mel_channel, Mel_norm, Amax):
    
    #Mel-frequency is proportional to "log(f/Mel_scale + 1)" [Default]700 or 1000
    Mel_scale = 700
    
    #Define Mel-scale parameter m0 based on "1000Mel = 1000Hz"
    m0 = 1000.0 / np.log(1000.0 / Mel_scale + 1.0)
    
    #Resolution of frequency
    df = Fs / fftsize
    
    #Define Nyquist frequency (the end point) as Hz, mel, and index scale
    Nyq = Fs / 2
    mel_Nyq = m0 * np.log(Nyq / Mel_scale + 1.0)
    n_Nyq = int(np.floor(fftsize / 2))+1
    
    #Calculate the Mel-scale interval between triangle-shaped structures
    #Divided by channel+1 because the termination is not the center of triangle but its right edge
    dmel = mel_Nyq / (Mel_channel + 1)
    
    #List up the center position of each triangle
    mel_center = np.arange(1, Mel_channel + 1) * dmel
    
    #Convert the center position into Hz-scale
    f_center = Mel_scale * (np.exp(mel_center / m0) - 1.0)
    
    #Define the center, start, and end position of triangle as index-scale
    n_center = np.round(f_center / df)
    n_start = np.hstack(([0], n_center[0 : Mel_channel - 1]))
    n_stop = np.hstack((n_center[1 : Mel_channel], [n_Nyq]))
    
    #Initial condition is defined as 0 padding matrix
    output = np.zeros((n_Nyq, Mel_channel))
    
    #Mel-scale filters are periodic triangle-shaped structures
    #Repeat every channel
    for c in np.arange(0, Mel_channel):
        
        #Slope of a triangle(growing slope)
        upslope = 1.0 / (n_center[c] - n_start[c])
        
        #Add a linear function passing through (nstart, 0) to output matrix 
        for x in np.arange(n_start[c], n_center[c]):
            #Add to output matrix
            x = int(x)
            output[x, c] = (x - n_start[c]) * upslope
        
        #Slope of a triangle(declining slope)
        dwslope = 1.0 / (n_stop[c] - n_center[c])
        
        #Add a linear function passing through (ncenter, 1) to output matrix 
        for x in np.arange(n_center[c], n_stop[c]):
            #Add to output matrix
            x = int(x)
            output[x, c] = 1.0 - ((x - n_center[c]) * dwslope)
        
        #Normalize area underneath each Mel-filter into 1
        #[Ref] T.Ganchev, N.Fakotakis, and G.Kokkinakis, Proc. of SPECOM 1, 191-194 (2005)
        #[URL] https://pdfs.semanticscholar.org/f4b9/8dbd75c87a86a8bf0d7e09e3ebbb63d14954.pdf
        if Mel_norm == True:
            output[:, c] = output[:, c] * 2 / (n_stop[c] - n_start[c])
    
    #Return Mel-scale filters as list (row=frequency, column=Mel channel)
    return output

### Function for calculating mel-based feature ###
def get_Melfeature(A, Fs, frame_length, frame_shift, Mel_channel, Mel_norm, MFCC_num, Amax, clu_mode):
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Call my function for generating Mel-scale filters(row: fftsize/2, column: Channel)
    filterbank = melFilterBank(Fs, FL, Mel_channel, Mel_norm, Amax)
    
    #Multiply the filters into the STFT amplitude
    melA = A.T @ filterbank
    
    #Normalization and get logarithm
    melA = melA * Amax / np.amax(melA)
    melA = np.log10(melA + 1)
    
    #In the case of k-means clustering method
    if clu_mode == "kmeans":
        #Get the DCT coefficients (DCT: Discrete Cosine Transformation)
        output = fp.realtransforms.dct(melA, type=2, norm="ortho", axis=1)
        
        #Trim the MFCC features from C(0) to C(MFCC_num-1)
        output = np.array(output[:, 0:MFCC_num])
    
    #In the case of second NMF clustering method
    elif clu_mode == "2ndNMF":
        output = melA
    
    #Return MFCC or mel-spectrogram as (frames, order) numpy array
    return output

### Function for getting metrics such as SDR ###
def get_metrics(truth, estimates):
    
    #Compute the SDR by bss_eval from museval library ver.4
    truth = truth[np.newaxis, :, np.newaxis]
    estimates = estimates[np.newaxis, :, np.newaxis]
    sdr, isr, sir, sar, perm = bss_eval_images(truth, estimates)
    #The function 'bss_eval_sources' is NOT recommended by documentation
    #[Ref] J. Le Roux et.al., "SDR-half-baked or well done?" (2018)
    #[URL] https://arxiv.org/pdf/1811.02508.pdf
    #sdr, sir, sar, perm = bss_eval_sources(truth, estimates)
    
    return sdr[0,0], isr[0,0], sir[0,0], sar[0,0], perm[0,0]

### Main ###
if __name__ == "__main__":
    
    #Setup
    down_sam = None        #Downsampling rate (Hz) [Default]None
    frame_length = 0.064   #STFT window width (second) [Default]0.064
    frame_shift = 0.032    #STFT window shift (second) [Default]0.032
    num_iter = 200         #The number of iteration in NMF [Default]200
    num_base = 25          #The number of basements in NMF [Default]20~30
    alpha = 0              #Weight of temporal continuity [Default]0 or 1e-4
    loss_func = "KL"       #Select EU, KL, or IS divergence [Default]KL
    Mel_channel = 20       #The number of frequency channel for Mel-scale filters [Default]20
    Mel_norm = True        #Normalize the area underneath each Mel-filter into 1 [Default]True
    MFCC_num = 9           #The number of MFCCs including C(0) [Default]9
    Amax = 1e4             #Normalization for log-Mel conversion [Default]1e4 (10000)
    clu_mode = "kmeans"    #Clustering method proposed in the original paper [Default]kmeans or 2ndNMF
    clu_loss = "EU"        #Using 2ndNMF, select EU or KL divergence [Default]EU
    clu_iter = 100         #Using 2ndNMF, specify the number of iterations [Default]100
    num_rep = 5            #The number of repetitions [Default]5
    
    #Define random seed
    np.random.seed(seed=32)
    
    #File path
    source1 = "./music/mixed.wav" #decompose it without training
    source2 = "./music/instrument1.wav" #for evaluation only
    source3 = "./music/instrument2.wav" #for evaluation only
    
    #Initialize variable for each metric
    SDR = np.zeros(num_rep)
    ISR = np.zeros(num_rep)
    SAR = np.zeros(num_rep)
    
    #Repeat for each iteration
    for rep in range(num_rep):
        
        #Prepare for process-log
        if clu_mode == "kmeans":
            log_path = "./log/{},normalNMF_{},{}".format(music, loss_func, clu_mode) + ".txt"
        else:
            log_path = "./log/{},normalNMF_{},{}_{}".format(music, loss_func, clu_mode, clu_loss) + ".txt"
        with open(log_path, "w") as f:
            f.write("")
        
        ### NMF step (to get basements matrix H) ###
        #Read mixed audio and true sources
        data, Fs = sf.read(source1)
        truth1, Fs = sf.read(source2)
        truth2, Fs = sf.read(source3)
        
        #Call my function for audio pre-processing
        data, Fs = pre_processing(data, Fs, down_sam)
        truth1, Fs = pre_processing(truth1, Fs, down_sam)
        truth2, Fs = pre_processing(truth2, Fs, down_sam)
        
        #Call my function for getting STFT (amplitude or power)
        Y, arg, Fs, freqs, times = get_STFT(data, Fs, frame_length, frame_shift)
        
        #Call my function for updating NMF basements and weights
        H, U, loss = get_NMF(Y, num_iter, num_base, loss_func, alpha, False)
        
        #Call my function for getting inverse STFT
        X = H @ U
        rec_wav, Fs = get_invSTFT(X, arg, Fs, frame_length, frame_shift)
        rec_wav = rec_wav[: int(data.shape[0])] #inverse stft includes residual part due to zero padding
        
        #Call my function for displaying graph
        #display_graph(Y, X, times, freqs, loss_func, num_iter)
        
        ### Clustering step (to get label for each sound source) ###
        #In the case of k-means clustering
        if clu_mode == "kmeans":
            
            #Call my function for getting MFCCs
            MFCC = get_Melfeature(H**2, Fs, frame_length, frame_shift, Mel_channel, Mel_norm, MFCC_num, Amax, clu_mode)
            
            #Normalize along with basements-axis
            MFCC = MFCC - np.mean(MFCC, axis=0, keepdims=True)
            MFCC = MFCC / np.std(MFCC, axis=0, keepdims=True)
            #This column normalization is not written in the original paper
            MFCC = MFCC - np.mean(MFCC, axis=1, keepdims=True)
            MFCC = MFCC / np.std(MFCC, axis=1, keepdims=True)
            
            #Get clustering by kmeans++
            clf = KMeans(n_clusters=2, init='k-means++', n_jobs=4)
            km_model = clf.fit(MFCC)
            label1 = np.array(km_model.labels_)
        
        #In the case of second NMF clustering
        elif clu_mode == "2ndNMF":
            
            #Call my function for getting mel-spectrogram
            melA = get_Melfeature(H**2, Fs, frame_length, frame_shift, Mel_channel, Mel_norm, MFCC_num, Amax, clu_mode)
            
            #Call my function for getting second NMF
            W, V, loss = get_NMF(melA.T, clu_iter, 2, clu_loss, 0, True)
            
            #Plot the loss curve for 2nd NMF
            #plt.figure(figsize=(10, 5))
            #plt.plot(np.arange(1, clu_iter+1), loss[:], marker='.')
            #plt.show()
            
            #Get clustering by second NMF
            label1 = np.argmax(V, axis=0)
        
        else:
            print("The 'clu_mode' should be either 'kmeans' or '2ndNMF'.")
            sys.exit()
        
        #print("Clustering vector a(i):{}".format(label1))
        label2 = np.ones(num_base) - label1
        label1 = label1[np.newaxis, :]
        label2 = label2[np.newaxis, :]
        
        #Decide which label corresponds to source1
        X = (H * label1) @ U
        rec_wav, Fs = get_invSTFT(X, arg, Fs, frame_length, frame_shift)
        rec_wav = rec_wav[: int(truth1.shape[0])] #inverse stft includes residual part due to zero padding
        sdr1,_,_,_,_ = get_metrics(truth1, rec_wav)
        sdr2,_,_,_,_ = get_metrics(truth2, rec_wav)
        if sdr1 > sdr2:
            H1 = H * label1
            H2 = H * label2
        else:
            H1 = H * label2
            H2 = H * label1
        
        #Get separation by using Wiener filter
        X1 = Y * (H1 @ U) / (H @ U)
        X2 = Y * (H2 @ U) / (H @ U)
        
        #Call my function for getting inverse STFT
        sep_wav1, Fs = get_invSTFT(X1, arg, Fs, frame_length, frame_shift)
        sep_wav1 = sep_wav1[: int(truth1.shape[0])] #inverse stft includes residual part due to zero padding
        sep_wav2, Fs = get_invSTFT(X2, arg, Fs, frame_length, frame_shift)
        sep_wav2 = sep_wav2[: int(truth2.shape[0])] #inverse stft includes residual part due to zero padding
        
        ### Evaluation step (to get SDR (signal-to-distortion ratio) of estimates) ###
        #Save the estimated sources
        sf.write("./log/" + str(rep) + "_Truth1.wav", truth1, Fs)
        sf.write("./log/" + str(rep) + "_Estimate1.wav", sep_wav1, Fs)
        sf.write("./log/" + str(rep) + "_Truth2.wav", truth2, Fs)
        sf.write("./log/" + str(rep) + "_Estimate2.wav", sep_wav2, Fs)
        
        #Call my function for getting metrics such as SDR
        sdr, isr, sir, sar, perm = get_metrics(truth1, sep_wav1)
        print("SDR: {:.3f} [dB]".format(sdr))
            
        #Save metric for each iteration
        SDR[rep], ISR[rep], SAR[rep] = sdr, isr, sar
        
    #Calculate average and confidence interval for each metric
    aveSDR, seSDR = np.average(SDR), 1.96*np.std(SDR) / np.sqrt(num_rep-1)
    aveISR, seISR = np.average(ISR), 1.96*np.std(ISR) / np.sqrt(num_rep-1)
    aveSAR, seSAR = np.average(SAR), 1.96*np.std(SAR) / np.sqrt(num_rep-1)
    with open(log_path, "a") as f:
        f.write(u"SDR={:.4f}\u00b1{:.4f}[dB]\nISR={:.4f}\u00b1{:.4f}[dB]\nSAR={:.4f}\u00b1{:.4f}[dB]".format(
            aveSDR, seSDR, aveISR, seISR, aveSAR, seSAR))