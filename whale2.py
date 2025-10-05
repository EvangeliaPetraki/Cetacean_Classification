# This file is with the augmentations. 

print('~~~~~~~~      This is where we remove both augmentations the learning rate is 0.001 and I keep a dropout layer in the MLP (I have marked the new layers)~~~~~~~~', flush = True)  

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torch.multiprocessing
from torchvision import datasets, transforms
from torchaudio.transforms import Resample
import scipy.io.wavfile as wavfile
import os
import soundfile as sf
from scipy.signal import resample
# from torchcodec.decoders import AudioDecoder
# decoder = AudioDecoder("C:/full/path/to/file.wav", ffmpeg_lib_path="C:/full/path/to/ffmpeg/bin/ffmpeg.exe")
from timeit import default_timer as timer
from kymatio.torch import Scattering1D
import torch.nn.functional as F
import random

from torch.utils.data import Dataset


class AugmentedTensorDataset(Dataset):
    def __init__(self, X, y, augment=False, sr=47600):
        self.X = X
        self.y = y
        self.augment = augment
        self.sr = sr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        label = self.y[idx]

        # if self.augment:
            # Apply waveform augmentation only if this is training
            # x = augment_waveform(x.unsqueeze(0), self.sr).squeeze(0)

        return x, label

import torchaudio

# --- Waveform augmentations ---
def augment_waveform(x, p=0.5):
    if random.random() < p:  # small Gaussian noise
        x = x + 0.001 * torch.randn_like(x)
    if random.random() < p:  # small gain jitter
        x = x * random.uniform(0.95, 1.05)
    if random.random() < p:  # small circular time shift (±8% window)
        shift = int(random.uniform(-0.02, 0.02) * x.shape[1])
        x = torch.roll(x, shifts=shift, dims=1)
    return x

# --- Spectrogram augmentations ---
def augment_spectrogram(spec, p=0.5):
    if random.random() < p:
        spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)(spec)
    if random.random() < p:
        spec = torchaudio.transforms.TimeMasking(time_mask_param=20)(spec)
    return spec

# from zipfile import ZipFile 
if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# print(device)

import matplotlib.pyplot as plt

class MelSpecDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, sr=47600, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
        self.sr = sr
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=64, normalized=True
        )

        # SpecAugment ops
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, T)
        y = self.y[idx]

        # light waveform augs ONLY for training
        # if self.augment:
        #     x = augment_waveform(x)

        m = self.mel(x)              # (n_mels, time)
        
        # if self.augment:             # SpecAugment only on train
        #    if random.random() < 0.5: #I add a light spec augment  
        #        m = self.freq_mask(m)
        #    if random.random() < 0.5:
        #        m = self.time_mask(m)

        # m = m.unsqueeze(0)           # (1, n_mels, time) -> conv2d input
        return m, y

def plot_training_curves(loss_train, loss_eval, acc_train, acc_eval, title="Training Progress", fname="training_curves.png"):
    epochs = range(1, len(loss_train) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_eval, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_train, label="Train Accuracy")
    plt.plot(epochs, acc_eval, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid()

    # plt.suptitle(title)
    # plt.show()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)   # save as PNG file
    plt.close()


file = 'E:/Cetaceos de Canarias Base/_common-frecuent/'
# file = "/home/f/fratzeska/E/Cetacean_Classification/_common-frecuent"


results_tot = {}
# def findOccurrences(s, ch):
#     return [i for i, letter in enumerate(s) if letter == ch]


test=[]
for item in os.listdir(file):
    # print(item)
    test.append(item)
classes_set = list(set(test))
# print(test)
# for item in os.listdir(file):
#     # print(item)
#     # if '_' not in item and item!= 'data/' and 'wav' in item:
#     print(item)
#     pos=findOccurrences(item,'/')
#     # print(pos)
#     test.append(item[pos[0]+1:pos[1]])
# classes_set=list(set(test))
# print(test)

y_labs = [] #this will contain the class labels 
data_full_list = [] #this will contain the audio data
srate_full_list = [] #this will contain the sample rate of each audio

target_sr = 47600   #Sets the target sampling rate for all audio

for root, dirs, files in os.walk(file):
    for fname in files:
        if fname.lower().endswith(".wav"):
            full_path = os.path.join(root, fname)
            
            # Class is the first folder name inside dataset
            rel_path = os.path.relpath(full_path, file)
            class_name = rel_path.split(os.sep)[0]

            try:

                # info = sf.info(full_path) #Reads the audio file header without loading the full data. It contains metadata: sample rate, number of channels, subtype (bits per sample), number of frames, etc.

                # if info.bits_per_sample not in ['PCM_16', 'PCM_24', 'PCM_32']: #Checks bits per sample, and Only allows 16, 24, or 32-bit PCM audio.

                #     print(f"Skipping {full_path}: unsupported bits_per_sample = {info.bits_per_sample}")
                    
                #     # If the audio is something else (e.g., float32 WAV or compressed), it skips it.
                    
                #     continue 
                
                x, sr = sf.read(full_path, dtype='float32') 
                # Reads the actual audio data. 
                # 
                # x: numpy array shape [samples] for mono, [samples, channels] for stereo/multi-channel.
                # 
                # sr → original sample rate of the audio.
                # 
                # dtype='float32' → ensures the same type as torchaudio loads by default.

                if x.ndim == 1:
                    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                else:
                    x = torch.tensor(x.T, dtype=torch.float32)


                #Converts the NumPy array to a PyTorch tensor.
                #
                # Ensures shape [channels, samples], same as torchaudio.
                #
                # unsqueeze(0) → adds a channel dimension if mono.
                #
                # x.T → transposes multi-channel audio so channels are first.

                if sr != target_sr:
                    n_samples = int(x.shape[1] * target_sr / sr)
                    x_resampled = resample(x.numpy(), n_samples, axis=1)
                    x = torch.tensor(x_resampled, dtype=torch.float32)

                # Checks if the audio needs resampling.
                #
                # Calculates the new number of samples to match target_sr.
                #
                # resample interpolates along samples axis (axis=1).
                #
                # Converts back to a PyTorch tensor [channels, samples].

                data_full_list.append(x)    # adds processed audio to the list
                y_labs.append(class_name) # adds class label
                srate_full_list.append(target_sr)  # ads resampled sample rate
            
            
            except Exception as e:
                print(f"Skipping {full_path}: {e}", flush = True)

            # If reading or processing the file fails, it prints a warning and skips the file.


signal_len = np.array([x.shape[1]/srate_full_list[k] for (k,x) in enumerate(data_full_list)]) 

# Loops over all audio signals stored in data_full_list.
#
# x.shape[1] gives the number of samples in the signal (remember, shape [channels, samples]).
#
# srate_full_list[k] is the sampling rate of the k-th signal (samples per second).
#
# Dividing x.shape[1] by srate_full_list[k] gives the duration in seconds of that signal.
#
# Wraps everything into a NumPy array → signal_len is now a 1D array of all audio durations in seconds.

avg = np.mean(signal_len) # computes the mean of all the durations

sd = np.std(signal_len) # computes the standard deviation of all the durations - Measures how spread out the audio lengths are from the average.

idx = np.where(signal_len <= avg+100*sd)[0] # selects signals that are not extremely long outliers.
# [0] extracts the array of indices.
#
# So idx contains indices of signals whose length is less than average + 100×SD.
#
# Side note: 100×SD is huge, basically keeps almost everything.

selected_items = signal_len[idx] # Uses the indices idx to select the durations of the “allowed” signals.
# selected_items is now an array of durations that passed the filter.


from torch.nn.functional import pad
def cutter(X,cut_point): #cuts and centers
# Purpose: Prepare for cutting or padding audio signals to a fixed length cut_point.

    cut_list = [] # it will store the processed tensors.
    cut_point = int(cut_point) # ensures the length is an integer.
    
    for x in X:
        n_len = x.shape[1] # number of samples in the signal 
        add_pts = cut_point - n_len #number of samples needed to reach cut_point.

        if n_len <= cut_point: #if the length is less than the cut point, 
            pp_left = int(add_pts / 2) # we pad the signal equally left and right
            pp_right = add_pts - pp_left 
            # cut_list.append(pad(x, (pp_left, pp_right)))
            cut_list.append(pad(x, (pp_left, pp_right))) #and then we transform it in a tensor and we add it to a list
        else:
            center_time = int(n_len / 2) #if the length is more than the cut point 
            pp_left = int(cut_point-center_time)    #we cut the signal equally from the left and right
            pp_right = cut_point - pp_left
            cut_list.append(x[:, center_time - pp_left:center_time + pp_right]) #and we add it to the list
    return torch.cat(cut_list)

y_sel = np.array(y_labs)[idx] #we select the longer signals?
data_sel = [data_full_list[j] for j in idx] #this is the array that contains the data with the longer signals

lens = [x.shape[1] for x in data_sel] 
cut_point= 8000 
X_cut = cutter(data_sel, cut_point) #normalises lengths to 8000 samples 
dict = {}

for name in set(y_labs):
    dict[name] = np.where(y_sel == name) #groups samples by class

keep_idx = [] #keeps classes with more than 50 samples
thrs = 50

for k in dict.keys():
    els = len(dict[k][0])
    if els > thrs:
        keep_idx.append(dict[k][0]) 

keep = set() 
#[keep.add(set(l)) for l in keep_idx]

for l in keep_idx:
    keep = keep.union(set(l))
kp = list(keep)
XC = X_cut[kp, :]   #<- this is the balanced dataset
#XP = X_pad[kp,:]   
y = y_sel[kp]       #<- this is the corresponding labels


def standardize(X): #normalises each signal to zero mean and unit variance
    st = torch.std(X, dim=1, keepdim=True)
    mn = torch.mean(X, dim=1, keepdim=True)
    return (X - mn) / st


X = standardize(XC) #standardization
X = XC
X = X.squeeze(1)

df_X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])]) #converts to dataframe 

# Add a column for y
df_X['y'] = y
df_X_no_duplicates = df_X.drop_duplicates(subset=df_X.iloc[:, 0:cut_point]) #removes duplicate recordings
X = torch.from_numpy(df_X_no_duplicates.iloc[:, 0:8000].values)
y = df_X_no_duplicates.iloc[:, -1].values

batches = [64, 128, 256]

#~~~~~~~~~~~~ FEATURE EXTRACTION ~~~~~~~~~~~~~~~

# from torchaudio.transforms import MelSpectrogram
batch_size = 500
N_batch =  int(X.shape[0]/batch_size)+1
# spectr = MelSpectrogram(normalized= True, n_mels = 64).to(device) # converts waveform to Mel spectrogram

# mx = [] #this will be the mel spectrogram dataset
# for n in range(N_batch):
#     x = X[n*batch_size:(n+1)*batch_size].to(torch.float32).to(device) #splits into batches
#     MX_tmp = spectr(x)
#     mx.append(MX_tmp)
# MX = torch.concat(mx)
# # print("Before unsqueeze:", MX.shape)
# MX = MX.unsqueeze(1)
# # print("After unsqueeze:", MX.shape)

#extracts mel features (batch, n_mels, time), adds a channel dimension -> supposedly ready for CNN input

batches = [64, 128, 256]
JQ = [(7, 10), (6, 16), (8,14)]


J, Q = JQ[2]
T = X.shape[1]

scattering=Scattering1D(J,T,Q)  #builds a scattering operator with scale J and quality Q
# scattering.cuda()
scattering = scattering.to(device)

#applies scattering transform batch-wise
#produces features [batch, channels, time]
wst = [] 
for n in range(N_batch):
    x = X[n*batch_size:(n+1)*batch_size].to(device)
    # x = x.unsqueeze(1).to(torch.float32).to(device)
    SX_tmp = scattering(x)
    wst.append(SX_tmp)
SX = torch.concat(wst) #this is the wavelet scattering coefficients dataset

# SX = SX.squeeze(1)

# print(SX.shape)

#groups scattering coefficients into 0th, 1st and 2nd order terms
meta = scattering.meta() #returns a dictionary describing the scattering coefficients.

# order0 = torch.from_numpy(np.where(meta['order'] == 0)[0])
# order1 = torch.from_numpy(np.where(meta['order'] == 1)[0])
# order2 = torch.from_numpy(np.where(meta['order'] == 2)[0])
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)


def median_norm(X):
    md = torch.median(X)
    sn = torch.std(X)
    return (X - md) / sn
# It computes median and std of a tensor X and normalizes it to median 0 and std 1.
# Output is a tensor with the same shape as X.

SX_med = SX
# SX_med = SX_med.unsqueeze(1)

# print(SX_med.shape)
for i in range(SX.shape[0]): 

#     SX.shape[0] is the number of signals (samples).
# For each sample i, this selects the subsets of scattering coefficients corresponding to order0, order1, order2 and applies median_norm() to each subset.
# median_norm(X) computes (X - median(X)) / std(X) where median/std are scalars computed over all elements of X (so normalization is per-sample-per-order, because you pass slices for each sample).
# The assignments replace those coefficients in SX_med with their normalized versions.

    SX_med[i][order0] = median_norm(SX[i][order0])
    SX_med[i][order1] = median_norm(SX[i][order1])
    SX_med[i][order2] = median_norm(SX[i][order2])
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

batch_size = 256

from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder() #Encodes class labels y to integer labels with LabelEncoder.
y_trc = torch.as_tensor(lbe.fit_transform(y))
index_shuffle=np.arange(len(y_trc))
idx_train, idx_test, y_trXX, y_testXX = train_test_split(index_shuffle, y_trc, test_size=.25, stratify=y) #Creates an index array and splits indices into train/test with stratification (so proportions per class are kept).

#y_trXX and y_testXX are label arrays corresponding to indices returned by train_test_split.

batch_size = batches[0]

#Builds train/val TensorDataset and DataLoader for Mel spectrograms (MX). MX.cpu() ensures data is on CPU for dataset storage/pickling.

# train_dataset_mel = TensorDataset(MX.cpu()[idx_train], y_trXX)
# train_dataset_mel = AugmentedTensorDataset(X[idx_train], y_trXX, augment=True, sr=target_sr)

# val_dataset_mel = TensorDataset(MX.cpu()[idx_test], y_testXX)
# val_dataset_mel   = AugmentedTensorDataset(X[idx_test], y_testXX, augment=False, sr=target_sr)

# train_dataloader_mel = DataLoader(train_dataset_mel, batch_size=batch_size, shuffle=True)
# train_dataloader_mel = DataLoader(train_dataset_mel, batch_size=batch_size, shuffle=True)

# val_dataloader_mel = DataLoader(val_dataset_mel, batch_size=batch_size, shuffle=False)
# val_dataloader_mel   = DataLoader(val_dataset_mel, batch_size=batch_size, shuffle=False)

train_dataset_mel = MelSpecDataset(X[idx_train], y_trXX, sr=target_sr, augment=True)
val_dataset_mel   = MelSpecDataset(X[idx_test],  y_testXX, sr=target_sr, augment=False)

train_dataloader_mel = DataLoader(train_dataset_mel, batch_size=batch_size, shuffle=True)
val_dataloader_mel   = DataLoader(val_dataset_mel,   batch_size=batch_size, shuffle=False)

#Builds TensorDatasets for scattering order1 and order2. SX_med[idx_train][:,order1] selects the training samples and the channels corresponding to order1. .cpu() moves the tensors to CPU before storing in dataset.
train_dataset_1 = TensorDataset(SX_med[idx_train][:,order1].cpu(), y_trXX)
val_dataset_1 = TensorDataset(SX_med[idx_test][:,order1].cpu(), y_testXX)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False)
train_dataset_2 = TensorDataset(SX_med[idx_train][:,order2].cpu(), y_trXX)
val_dataset_2 = TensorDataset(SX_med[idx_test][:,order2].cpu(), y_testXX)
train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
val_dataloader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.3) # <----- This is the new Dropout in the resnet (to remove you have to remove the one a few lines below too)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x) #<------ this is the new dropout in the resnet (to remove you have to remove the one a few lines above too)
        x = self.fc(x)

        return x

# Instantiate the model
model_mel = ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)
model_wst_1=ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)
model_wst_2=ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)
learning_rate_mel = .01
# learning_rate_mel = .001
# learning_rate_mel = 3e-4 # <---- we lower the lr here
optimizer_mel = torch.optim.AdamW(model_mel.parameters(), lr=learning_rate_mel,amsgrad= True, weight_decay= .001 )
scheduler_mel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mel, 'min')
learning_rate_1 = .01
# learning_rate_1 = 3e-4 # <---- we lower the lr here
# learning_rate_1 = 0.001
optimizer_1 = torch.optim.AdamW(model_wst_1.parameters(), lr=learning_rate_1,amsgrad= True, weight_decay= .001 )
scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min')
learning_rate_2 = .01
# learning_rate_2 = 0.001
# learning_rate_2 = 3e-4 # <---- we lower the lr here

optimizer_2 = torch.optim.AdamW(model_wst_2.parameters(), lr=learning_rate_2,amsgrad= True, weight_decay= .001 )
scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min')

def training_resnet(model,train_dataloader,val_dataloader,learning_rate,optimizer,scheduler, fname):
    criterion = nn.CrossEntropyLoss()
    
    n_total_steps = len(train_dataloader)
    num_epochs = 100
    # num_epochs = 1
    loss_train = []
    acc_train = []
    acc_eval = []
    loss_eval = []
    for epoch in range(num_epochs):
        print("starting new epoch ", flush = True)
        print(epoch, flush = True)
        loss_ep_train = 0
        n_samples = 0
        n_correct = 0
        for i, (x, labels) in enumerate(train_dataloader):
    
            x = x.to(device)
    
            labels = labels.to(device, dtype=torch.long)
    
            #forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)
    
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            loss_ep_train += loss.item()
            _, predictions = torch.max(outputs, 1)
    
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
    
    
    
            if (i + 1) % 100 == 0:
                print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ', flush = True)
    
        acc_tr = 100 * n_correct / n_samples
        acc_train.append(acc_tr)
        loss_train.append(loss_ep_train/len(train_dataloader))
    
        loss_ep_eval = 0
    
        with torch.no_grad():
    
            n_correct = 0
            n_samples = 0
    
            for x, labels in val_dataloader:
                x = x.to(device)
    
                labels = labels.to(device)
                outputs = model(x)
                lossvv = criterion(outputs, labels)
    
                _, predictions = torch.max(outputs, 1)
    
                n_samples += labels.shape[0]
                n_correct += (predictions == labels).sum().item()
                loss_ep_eval += lossvv.item()
    
            acc = 100 * n_correct / n_samples
    
        acc_eval.append(acc)
        loss_eval.append(loss_ep_eval/len(val_dataloader))
    
        print(f' validation accuracy = {acc}', flush = True)
    
    res = np.array([loss_train, loss_eval, acc_train, acc_eval])
    
    
    namefile = f'{fname}_{J,Q}_{batch_size}'
    np.save(namefile, res)

    from sklearn.metrics import roc_auc_score
    yp = []
    ytr = []
    y_prob = []
    import time 
    times=[]
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
    
        for x, labels in val_dataloader:
            x = x.to(device)
    
            labels = labels.to(device)
            tick=time.time()
            outputs = model(x)
            tick=time.time()-tick
            times.append(tick/len(labels))
            pr_out = torch.softmax(outputs, dim = 1)
    
            proba, predictions = torch.max(pr_out, 1)
    
            yp.append(predictions.cpu().numpy())
            ytr.append(labels.cpu().numpy())
            y_prob.append(pr_out.cpu().numpy())
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
    
        
        acc = 100 * n_correct / n_samples


training_resnet(model_mel,train_dataloader_mel,val_dataloader_mel,learning_rate_mel,optimizer_mel,scheduler_mel, 'modelmel')
training_resnet(model_wst_1,train_dataloader_1,val_dataloader_1,learning_rate_1,optimizer_1,scheduler_1,'modelws1')
training_resnet(model_wst_2,train_dataloader_2,val_dataloader_2,learning_rate_2,optimizer_2,scheduler_2, 'modelws2')

train_dataloader_1_fin = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=False)
val_dataloader_1_fin = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False)
train_dataloader_2_fin = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=False)
val_dataloader_2_fin = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)
list_prob_1=[]
list_prob_2=[]
list_prob_1_val=[]
list_prob_2_val=[]
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    for x, labels in train_dataloader_1_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_1(x)
        list_prob_1.append(outputs)
    for x, labels in val_dataloader_1_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_1(x)
        list_prob_1_val.append(outputs)
    for x, labels in train_dataloader_2_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_2(x)
        list_prob_2.append(outputs)
    for x, labels in val_dataloader_2_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_wst_2(x)
        list_prob_2_val.append(outputs)


prob_train_1=torch.concat(list_prob_1)
prob_train_2=torch.concat(list_prob_2)
train=torch.hstack((prob_train_1,prob_train_2))
prob_val_1=torch.concat(list_prob_1_val)
prob_val_2=torch.concat(list_prob_2_val)
val=torch.hstack((prob_val_1,prob_val_2))


train_final = TensorDataset(train, y_trXX)
val_final = TensorDataset(val, y_testXX)
train_final_load = DataLoader(train_final, batch_size=batch_size, shuffle=True)
val_final_load = DataLoader(val_final, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        # super(MLP, self).__init__()
        # self.linear=nn.Linear(64,256)
        # self.activation=nn.ReLU()
        # self.linear2=nn.Linear(256,128)
        # self.activation=nn.ReLU()
        # self.linear3=nn.Linear(128,32)

        super(MLP, self).__init__() #This whole block is to add the new dropout in the MLP. If you want to remove remove all of it and uncomment the block above
        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 32)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        

    # def forward(self, x):
    #     out=self.linear(x)
    #     out=self.activation(out)
    #     out=self.linear2(out)
    #     out=self.activation(out)
    #     out=self.linear3(out)
    #     return out

    def forward(self, x): #this block is also to add the new dropout in the MLP . If you want to remove you have to remove all of it and uncomment the closk above
        out = self.activation(self.linear1(x))
        out = self.dropout(out)
        out = self.activation(self.linear2(out))
        out = self.dropout(out)
        out = self.linear3(out)
        return out


model_MLP = MLP().to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = .001
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=learning_rate )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
n_total_steps = len(train_final_load)
num_epochs = 500
# num_epochs = 1

loss_train = []
acc_train = []
acc_eval = []
loss_eval = []
for epoch in range(num_epochs):

    loss_ep_train = 0
    n_samples = 0
    n_correct = 0
    for i, (x, labels) in enumerate(train_final_load):

        x = x.to(device)

        labels = labels.to(device)

        #forward pass
        outputs = model_MLP(x)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()



        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ', flush = True)

    acc_tr = 100 * n_correct / n_samples
    acc_train.append(acc_tr)
    loss_train.append(loss_ep_train/len(train_final_load))

    loss_ep_eval = 0

    with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for x, labels in val_final_load:
            x = x.to(device)

            labels = labels.to(device)
            outputs = model_MLP(x)
            lossvv = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            loss_ep_eval += lossvv.item()

        acc = 100 * n_correct / n_samples
    acc_eval.append(acc)
    loss_eval.append(loss_ep_eval/len(val_final_load))

    if epoch%100==0:
        print(f' validation accuracy = {acc}', flush = True)
res = np.array([loss_train, loss_eval, acc_train, acc_eval])

namefile = f'S1+S2_{J,Q}_{batch_size}'
np.save(namefile, res)

    
train_dataloader_mel_fin = DataLoader(train_dataset_mel, batch_size=batch_size, shuffle=False)
val_dataloader_mel_fin = DataLoader(val_dataset_mel, batch_size=batch_size, shuffle=False)
list_prob_mel=[]
list_prob_mel_val=[]
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    for x, labels in train_dataloader_mel_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_mel(x)
        list_prob_mel.append(outputs)
    for x, labels in val_dataloader_mel_fin:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_mel(x)
        list_prob_mel_val.append(outputs)


train_dataloader_star = DataLoader(train_final, batch_size=batch_size, shuffle=False)
val_dataloader_star = DataLoader(val_final, batch_size=batch_size, shuffle=False)
list_prob_star=[]
list_prob_star_val=[]
with torch.no_grad():

    n_correct = 0
    n_samples = 0

    for x, labels in train_dataloader_star:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_MLP(x)
        list_prob_star.append(outputs)
    for x, labels in val_dataloader_star:
        x = x.to(device)

        labels = labels.to(device)
        outputs = model_MLP(x)
        list_prob_star_val.append(outputs)

prob_train_star=torch.concat(list_prob_star)
prob_train_mel=torch.concat(list_prob_mel)
# train=torch.hstack((prob_train_1,prob_train_2))
prob_val_star=torch.concat(list_prob_star_val)
prob_val_mel=torch.concat(list_prob_mel_val)
# val=torch.hstack((prob_val_1,prob_val_2))

outputs=torch.max(torch.hstack((prob_val_star.unsqueeze(1),prob_val_mel.unsqueeze(1))),1)[0]
# print(outputs.shape)
_, predictions = torch.max(outputs, 1)
# n_correct = (predictions == y_testXX.cuda() ).sum().item()

n_correct = (predictions == y_testXX.to(device)).sum().item()

accuracy=n_correct/predictions.shape[0]
final_res = {}
print(f'{accuracy}', flush = True)
final_res['max_merge'] = accuracy
def get_best_lambda(pi_train1, pi_train2, pi_val1, pi_val2, y_train, y_val, n_lambda = 11):

    lambda_range = np.linspace(0,1, n_lambda)

    res = {}

    for l in lambda_range:

        pi_end_train = l * pi_train1 + (1- l) * pi_train2

        pi_end_val = l * pi_val1 + (1- l) * pi_val2

 

        _, pred_train = torch.max(pi_end_train, dim = 1)

        correct_predictions = (pred_train == y_train).sum()

        acc_train = correct_predictions/y_train.shape[0]

 

        _, pred_val = torch.max(pi_end_val, dim = 1)

        correct_predictions = (pred_val == y_val).sum()

        acc_val = correct_predictions/y_val.shape[0]

 

        res[l] = [acc_train.item(), acc_val.item()]
    return res
# print(get_best_lambda(prob_train_star, prob_train_mel, prob_val_star, prob_val_mel, y_trXX.cuda(), y_testXX.cuda(), n_lambda = 31))
print(get_best_lambda(prob_train_star, prob_train_mel, prob_val_star, prob_val_mel, y_trXX.to(device), y_testXX.to(device), n_lambda = 31), flush = True)
# final_res['lambdas'] = get_best_lambda(prob_train_star, prob_train_mel, prob_val_star, prob_val_mel, y_trXX.cuda(), y_testXX.cuda(), n_lambda = 31)
final_res['lambdas'] = get_best_lambda(prob_train_star, prob_train_mel, prob_val_star, prob_val_mel, y_trXX.to(device), y_testXX.to(device), n_lambda = 31)
train_hard=torch.hstack((prob_train_star,prob_train_mel))
val_hard=torch.hstack((prob_val_star,prob_val_mel))
train_boh = TensorDataset(train_hard, y_trXX)
val_boh = TensorDataset(val_hard, y_testXX)
train_hard_load = DataLoader(train_boh, batch_size=batch_size, shuffle=True)
val_hard_load = DataLoader(val_boh, batch_size=batch_size, shuffle=False)

model_MLP = MLP().to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = .001
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=learning_rate )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
n_total_steps = len(train_hard_load)
num_epochs = 500
# num_epochs = 1

loss_train = []
acc_train = []
acc_eval = []
loss_eval = []
for epoch in range(num_epochs):

    loss_ep_train = 0
    n_samples = 0
    n_correct = 0
    for i, (x, labels) in enumerate(train_hard_load):

        x = x.to(device)

        labels = labels.to(device)

        #forward pass
        outputs = model_MLP(x)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()



        if (i + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ', flush = True)

    acc_tr = 100 * n_correct / n_samples
    acc_train.append(acc_tr)
    loss_train.append(loss_ep_train/len(train_hard_load))

    loss_ep_eval = 0

    with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for x, labels in val_hard_load:
            x = x.to(device)

            labels = labels.to(device)
            outputs = model_MLP(x)
            lossvv = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            loss_ep_eval += lossvv.item()

        acc = 100 * n_correct / n_samples
    acc_eval.append(acc)
    loss_eval.append(loss_ep_eval/len(val_hard_load))

    if epoch%100==0:
        print(f' validation accuracy = {acc}', flush = True)
    
res = np.array([loss_train, loss_eval, acc_train, acc_eval])

namefile = f'MLP_S+Mel{J,Q}_{batch_size}'
np.save(namefile, res)

plot_training_curves(loss_train, loss_eval, acc_train, acc_eval, title="ResNet Mel Training", fname="resnet_mel_training.png")

