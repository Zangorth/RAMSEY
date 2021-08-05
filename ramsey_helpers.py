from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from torch import nn
import numpy as np
import warnings
import torch

device = torch.device('cuda:0')

warnings.filterwarnings('error', category=ConvergenceWarning)


def lags(x, group, lags, exclude = []):
    out = x.copy()
    out = out[[col for col in out.columns if col not in exclude]]
    
    for i in range(lags):
        lag = x.groupby(group).shift(i)
        lag.columns = [f'{col}_lag{i}' for col in lag.columns]
        
        out = out.merge(lag, left_index=True, right_index=True)
        
    return out

class Discriminator(nn.Module):
    def __init__(self, a, b, drop, shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(shape, a),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Linear(b, 10),
            nn.Softmax(dim=1)
            )
        
        
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
        
        discriminator = RandomForestClassifier(n_estimators=n_estimators, n_jobs=4)
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
                                      max_depth=max_depth, n_jobs=4, use_label_encoder=False,
                                      objective='multi:softmax', eval_metric='mlogloss')
        discriminator.fit(x_train, y_train)
        predictions = discriminator.predict(x_test)
        f1.append(f1_score(y_test, predictions, average='micro'))
    
    return np.mean(f1)

def cv_nn(x, y, semi, a, b, drop, lr_nn, epochs):
    f1 = []
    
    for i in range(5):
        x_test = x.loc[semi == 'semi'].groupby(y, group_keys=False).apply(lambda x: x.sample(min(len(x), 25))).sort_index()
        x_train = x.loc[~x.index.isin(x_test.index)].sort_index()
        
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]
        
        col_count = x_train.shape[1]
        x_train, x_test = torch.from_numpy(x_train.values).to(device), torch.from_numpy(x_test.values).to(device)
        y_train, y_test = torch.from_numpy(y_train.values).to(device), torch.from_numpy(y_test.values).to(device)
        
        train_set = [(x_train[i].to(device), y_train[i].to(device)) for i in range(len(y_train))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**7, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = Discriminator(a, b, drop, col_count).to(device)
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
    
    return np.mean(f1)