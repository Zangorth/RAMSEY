import numpy as np
import warnings
import librosa

warnings.filterwarnings('ignore')


class ParameterError(Exception):
    pass

def extract_audio(sound):
    
    try:
        file = sound[0]
        cut = sound[1]
        sound = sound[2]
        
        y, rate = librosa.load(sound.export(format='wav'), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
        
        features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
        features = [float(f) for f in features]
        features = [file, cut] + features
        
    except ValueError:
        features = []
        
    except ParameterError:
        features = []
    
    except Exception:
        features = []
    
    return features