from sklearn.model_selection import train_test_split as split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from skorch import NeuralNetClassifier as nnc
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from collections import OrderedDict
from xgboost import XGBClassifier
from pydub.playback import play
from torch import nn
import pandas as pd
import numpy as np
import warnings
import librosa
import torch

warnings.filterwarnings('error', category=ConvergenceWarning)

class ParameterError(Exception):
    pass

###########################
# Neural Network Function #
###########################
class Discriminator(nn.Module):
    def __init__(self, shape, transforms, drop, output, layers=3):
        super().__init__()
        
        transforms = [shape] + transforms
        sequential = OrderedDict()
        
        i = 0
        while i < layers:
            sequential[f'linear_{i}'] = nn.Linear(transforms[i], transforms[i+1])
            sequential[f'relu_{i}'] = nn.ReLU()
            sequential[f'drop_{i}'] = nn.Dropout(drop)
            i+=1
            
        sequential['linear_final'] = nn.Linear(transforms[i], output)
        sequential['softmax'] = nn.Softmax(dim=1)
        
        self.model = nn.Sequential(sequential)
        
    def forward(self, x):
        output = self.model(x)
        return output
        
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
    prediction = f' {prediction}' if prediction != '' else ''
    
    train = 0
    while train == 0:
        play(sound)
        
        train = input(f'{status} Label{prediction}: ')
        train = train.upper()
        train = 0 if train == '0' else train

        if str(train).lower() == '':
            train = prediction.strip()
        
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

##############################
# Cross Validation Functions #
##############################
def cv(model, x, y, 
       n_samples, n_estimators, lr_gbc, max_depth,
       transforms, drop, layers, epochs, lr_nn, output=9, 
       cv=20, frac=0.1, over=True, avg=False, wrong=False):
    avg_out = pd.DataFrame(columns=['speaker_id', 'f1'])
    wrong_out = pd.DataFrame(columns=['index', 'real', 'pred'])
    f1 = []
    
    for i in range(cv):
        x_train, x_test, y_train, y_test = split(x, y, test_size=frac, stratify=y)
        
        if over:
            oversample = SMOTE()
            x_train, y_train = oversample.fit_resample(x_train, y_train)
        
        test_idx = x_test.index
        
        if model == 'logit':
            discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
        
        elif model == 'rfc':
            discriminator = RandomForestClassifier(n_estimators=n_samples, n_jobs=-1)
            
        elif model == 'gbc':
            discriminator = XGBClassifier(n_estimators=n_estimators, learning_rate=lr_gbc, 
                                          max_depth=max_depth, n_jobs=-1, use_label_encoder=False,
                                          objective='multi:softmax', eval_metric='mlogloss',
                                          tree_method='gpu_hist')
        
        elif model == 'nn':
            x_train, y_train = x_train.values.astype(np.float32), y_train.astype(np.int64)
            x_test, y_test = x_test.values.astype(np.float32), y_test.astype(np.int64)
            classifier = Discriminator(x_train.shape[1], transforms, drop, output, layers)
            discriminator = nnc(classifier, max_epochs=epochs, lr=lr_nn,
                                optimizer=torch.optim.Adam, batch_size=2**9, criterion=nn.CrossEntropyLoss,
                                device='cuda:0', verbose=0, train_split=None)
            
        try:
            discriminator.fit(x_train, y_train)
            predictions = discriminator.predict(x_test)
            f1.append(f1_score(y_test, predictions, average='macro'))
        except ConvergenceWarning:
            f1.append(0)
            
        if avg:
            append = pd.DataFrame(f1_score(y_test, predictions, average=None)).reset_index()
            append.columns = avg_out.columns
            avg_out = avg_out.append(append, ignore_index=True, sort=False)
            
        if wrong:
            append = pd.DataFrame({'index': test_idx, 'real': y_test, 'pred': predictions})
            wrong_out.append(append, ignore_index=True, sort=False)
            wrong_out = wrong_out.loc[wrong_out.real != wrong_out.pred]
            
    if avg and wrong:
        message = 'Choose either avg=True or wrong=True'
        return message
    
    elif avg:
        return avg_out.groupby('speaker_id').mean()
    
    elif wrong:
        return wrong_out.drop_duplicates().reset_index(drop=True)
            
    else:
        return np.mean(f1)

