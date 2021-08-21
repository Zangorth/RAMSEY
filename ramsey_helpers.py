from pydub.playback import play
import numpy as np
import warnings
import librosa

class ParameterError(Exception):
    pass       
        
#####################
# Audio Exctraction #
#####################
def extract_audio(sound):
    warnings.filterwarnings('ignore')
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

##################
# Audio Training #
##################
def train_audio(sound, lead, second, link, prediction='', iterator='', size=''):
    status = '' if iterator == '' and size == '' else f'{iterator}/{size}'
    string_pred = f' {prediction}?' if prediction != '' else ''

    train = 0
    while train == 0:
        play(sound)

        train = input(f'{status} Label{string_pred}: ')
        train = train.upper()
        train = 0 if train == '0' else train

        if str(train).lower() == '':
            train = prediction

        elif str(train).lower() == 'lead':
            train = 0
            play(lead)

        elif str(train).lower() == 'show':
            train = 0
            timestamp = f'{int(round(second/60, 0))}:{second % 60}'
            print(f'{timestamp} - {link}')

        else:
            pass

    return train

##############
# Lags/Leads #
##############
def shift(x, group, lags, leads, exclude = []):
    out = x.copy()
    x = out[[col for col in out.columns if col not in exclude]]
    
    for i in range(lags):
        lag = x.groupby(group).shift(i)
        lag.columns = [f'{col}_lag{i}' for col in lag.columns]
        
        out = out.merge(lag, left_index=True, right_index=True)
        
    for i in range(leads):
        lead = x.groupby(group).shift(-i)
        lead.columns = [f'{col}_lead{i}' for col in lead.columns]
        
        out = out.merge(lead, left_index=True, right_index=True)
        
    return out