from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, accuracy_score
from collections import OrderedDict
from xgboost import XGBClassifier
from torch import nn
import pandas as pd
import numpy as np
import warnings
import librosa
import torch

device = torch.device('cuda:0')

warnings.filterwarnings('error', category=ConvergenceWarning)


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
            
            
def cv_logit(x, y, semi):
    f1 = []
    
    for i in range(5):
        x_test = x.loc[semi == 'semi'].groupby(y, group_keys=False).apply(lambda x: x.sample(min(len(x), 25))).sort_index()
        x_train = x.loc[~x.index.isin(x_test.index)].sort_index()
        
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]
        
        try:
            discriminator = LogisticRegression(max_iter=500)
            discriminator.fit(x_train, y_train)
            predictions = discriminator.predict(x_test)
            f1.append(f1_score(y_test, predictions, average='micro'))
        except ConvergenceWarning:
            f1.append(0)
    
    return np.mean(f1)

def cv_rfc(x, y, semi, n_estimators):
    f1 = []
    
    for i in range(5):
        x_test = x.loc[semi == 'semi'].groupby(y, group_keys=False).apply(lambda x: x.sample(min(len(x), 25))).sort_index()
        x_train = x.loc[~x.index.isin(x_test.index)].sort_index()
        
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]
        
        discriminator = RandomForestClassifier(n_estimators=n_estimators, n_jobs=8)
        discriminator.fit(x_train, y_train)
        predictions = discriminator.predict(x_test)
        f1.append(f1_score(y_test, predictions, average='micro'))
    
    return np.mean(f1)
        
def cv_gbc(x, y, semi, n_estimators, lr_gbc, max_depth):
    f1 = []
    
    for i in range(5):
        x_test = x.loc[semi == 'semi'].groupby(y, group_keys=False).apply(lambda x: x.sample(min(len(x), 25))).sort_index()
        x_train = x.loc[~x.index.isin(x_test.index)].sort_index()
        
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]
    
        discriminator = XGBClassifier(n_estimators=n_estimators, learning_rate=lr_gbc, 
                                      max_depth=max_depth, n_jobs=10, use_label_encoder=False,
                                      objective='multi:softmax', eval_metric='mlogloss',
                                      tree_method='gpu_hist')
        discriminator.fit(x_train, y_train)
        predictions = discriminator.predict(x_test)
        f1.append(f1_score(y_test, predictions, average='micro'))
    
    return np.mean(f1)

def cv_nn(x, y, semi, transforms, drop, lr_nn, epochs, layers=3, output=10, prin=False):
    f1 = []
    
    for i in range(20):
        x_test = x.loc[semi == 'semi'].groupby(y, group_keys=False).apply(lambda x: x.sample(min(len(x), 25))).sort_index()
        x_train = x.loc[~x.index.isin(x_test.index)].sort_index()
        
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]
        
        col_count = x_train.shape[1]
        x_train, x_test = torch.from_numpy(x_train.values).to(device), torch.from_numpy(x_test.values).to(device)
        y_train, y_test = torch.from_numpy(y_train.values).to(device), torch.from_numpy(y_test.values).to(device)
        
        train_set = [(x_train[i].to(device), y_train[i].to(device)) for i in range(len(y_train))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**10, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = Discriminator(col_count, transforms, drop, output, layers).to(device)
        optim = torch.optim.Adam(discriminator.parameters(), lr=lr_nn)
    
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                discriminator.zero_grad()
                yhat = discriminator(inputs.float())
                loss = loss_function(yhat, targets.long())
                loss.backward()
                optim.step()
                
        test_hat = discriminator(x_test.float())
        f1.append(f1_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1), average='micro'))
        
        if prin:
            print(f'Accuracy: {accuracy_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1))}')
            print('')
            print('F1')
            print(pd.DataFrame(f1_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1), average=None)))
            
    
    return np.mean(f1)


class ParameterError(Exception):
    pass

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

