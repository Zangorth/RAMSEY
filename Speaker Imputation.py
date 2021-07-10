from sklearn import preprocessing
from pydub import AudioSegment
from torch import nn
import pandas as pd
import numpy as np
import librosa
import pickle
import torch
import os

os.chdir(r'C:\Users\Samuel\Audio\Audio Full')

class Discriminator(nn.Module):
    def __init__(self, a, b, drop):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(193, a),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Linear(b, 8),
            nn.Softmax(dim=1)
            )
        
        
    def forward(self, x):
        output = self.model(x)
        return output

results = pickle.load(open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\optimization results.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\optimization scaler.pkl', 'rb'))

discriminator = Discriminator(results[1], results[2], results[4])
discriminator.load_state_dict(torch.load(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\Voice Recognition.pt'))
discriminator.eval()


for file in os.listdir():
    panda = pd.DataFrame(columns=['start', 'end'] + [i for i in range(193)])
    sound = AudioSegment.from_file(file)
    cut = 0
    
    while cut <= len(sound):
        out = sound[cut:cut+3000]
        out.export(r'C:\Users\Samuel\Audio\Audio Segment\tobedeleted.wav', format='wav')
        
        y, rate = librosa.load(r'C:\Users\Samuel\Audio\Audio Segment\tobedeleted.wav', res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
        
        features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
        features = [float(f) for f in features]
        features = [cut/1000, (cut+3000)/1000] + features
        
        temp = pd.DataFrame(features).T
        temp.columns = ['start', 'end'] + [i for i in range(193)]
        
        panda = panda.append(temp, ignore_index=True, sort=False)
        
        
        cut += 1000
        
        
    
        
