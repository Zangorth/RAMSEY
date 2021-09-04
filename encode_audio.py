import pandas as pd
import numpy as np
import warnings
import librosa
import pydub
import scipy
import ray

@ray.remote
def encode_audio(sound):
    warnings.filterwarnings('ignore')
    
    speaker = sound[0]
    index = sound[1]
    second = sound[2]
    sound = sound[3]
    
    try:
        y, rate = librosa.load(sound.export(format='wav'), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
        
        features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
        features = [float(f) for f in features]
        features = [speaker, index, second] + features
        
        features = pd.DataFrame([features])
        
    except ValueError:
        features = pd.DataFrame(index=[0])
        features.iloc[0, 0:3] = [speaker, index, second]
    
    except Exception:
        features = pd.DataFrame(index=[0])
        features.iloc[0, 0:3] = [speaker, index, second]
    
    return features