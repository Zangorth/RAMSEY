from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sqlalchemy import create_engine
from sklearn import preprocessing
from skopt import plots
import pyodbc as sql
from torch import nn
import pandas as pd
import numpy as np
import urllib
import skopt
import torch

####################
# Define Functions #
####################

device = torch.device('cuda:0')

class Discriminator(nn.Module):
    def __init__(self, a, b, drop):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(580, a),
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


#############
# Read Data #
#############

# Training Audio
con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT speaker, code.*
FROM RAMSEY.dbo.AudioTraining AS train
LEFT JOIN RAMSEY.dbo.AudioCoding AS code
    ON train.id = code.id
    AND train.[second] = code.[second]
'''

train_audio = pd.read_sql(query, con)

query = '''SELECT * FROM RAMSEY.dbo.AudioCoding'''
#audio = pd.read_sql(query, con)

con.close()

##############
# Clean Data #
##############
train_audio['y'] = train_audio['speaker'].astype('category').cat.codes

mapped = train_audio[['y', 'speaker']].drop_duplicates().reset_index(drop=True)
mapped = mapped.sort_values('y')

x = train_audio.drop(['id', 'second', 'speaker', 'y'], axis=1)

scaler = preprocessing.StandardScaler().fit(x)
x = pd.DataFrame(scaler.transform(x), columns=train_audio.columns[3:-1])
x = train_audio[['id']].merge(x, left_index=True, right_index=True)

lag1 = x.groupby('id').shift(1)
lag1.columns = [f'{col}_lag1' for col in x.columns if col != 'id']

lag2 = x.groupby('id').shift(2)
lag2.columns = [f'{col}_lag2' for col in x.columns if col != 'id']

x = x.merge(lag1, left_index=True, right_index=True)
x = x.merge(lag2, left_index=True, right_index=True).dropna()

del [lag1, lag2]

kf = StratifiedKFold()
y = train_audio.iloc[x.index, -1]
y = torch.from_numpy(y.values)

x = torch.from_numpy(x.values)

#transform = pd.DataFrame(scaler.transform(audio.drop(['id', 'second'], axis=1)), columns=audio.columns[2:])
#audio = audio[['id', 'second']].merge(transform, right_index=True, left_index=True, how='outer')

####################
# Optimize Network #
####################
space = [
    skopt.space.Integer(1, 30, name='epochs'),
    skopt.space.Integer(2**2, 2**10, name='a'),
    skopt.space.Integer(2**2, 2**10, name='b'),
    skopt.space.Real(0.001, 0.5, name='lr', prior='log-uniform'),
    skopt.space.Real(0.0001, 1, name='drop', prior='log-uniform')
    ]

epochs, a, b, drop, lr = 20, 64, 32, 0.15, 0.01

@skopt.utils.use_named_args(space)
def net(epochs, a, b, drop, lr):
    
    f1 = []
    for train_index, test_index in kf.split(x, y):
        x_train = x[train_index].to(device)
        x_test = x[test_index].to(device)
        
        y_train = y[train_index].to(device)
        y_test = y[test_index].to(device)
        
        train_set = [(x_train[i].to(device), y_train[i].to(device)) for i in range(len(y_train))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**7, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = Discriminator(a, b, drop).to(device)
        optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                discriminator.zero_grad()
                yhat = discriminator(inputs.float())
                loss = loss_function(yhat, targets.long())
                loss.backward()
                optim.step()
                
        test_hat = discriminator(x_test.float())
        f1.append(f1_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1), average='micro'))
    
    print(np.mean(f1))
    return (- 1.0 * np.mean(f1))
        

result = skopt.forest_minimize(net, space, acq_func='PI')

print(f'Max F1: {result.fun}')
print(f'Parameters: {result.x}')
plots.plot_evaluations(result)

####################
# Validate Network #
####################
kf = StratifiedKFold(10)
f1 = []
for train_index, test_index in kf.split(x, y):
    x_train = x[train_index].to(device)
    x_test = x[test_index].to(device)
    
    y_train = y[train_index].to(device)
    y_test = y[test_index].to(device)
    
    train_set = [(x_train[i], y_train[i]) for i in range(len(y_train))]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    discriminator = Discriminator(result.x[1], result.x[2], result.x[4]).to(device)
    optim = torch.optim.Adam(discriminator.parameters(), lr=result.x[3])

    for epoch in range(result.x[0]):
        for i, (inputs, targets) in enumerate(train_loader):
            discriminator.zero_grad()
            yhat = discriminator(inputs.float())
            loss = loss_function(yhat, targets.long())
            loss.backward()
            optim.step()
            
    test_hat = discriminator(x_test.float())
    f1.append(f1_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1), average='micro'))
    
    print(f'Accuracy: {accuracy_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1))}')
    mapped['f1'] = f1_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1), average=None)
    mapped['recall'] = recall_score(y_test.cpu(), np.argmax(test_hat.cpu().detach().numpy(), axis=1), average=None)
    print(mapped)
    print('')

print(np.mean(f1))


#################
# Train Network #
#################
torch.cuda.empty_cache()
x, y = x.to(device), y.to(device)
train_set = [(x[i], y[i]) for i in range(len(y))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

loss_function = nn.CrossEntropyLoss()
discriminator = Discriminator(result.x[1], result.x[2], result.x[4]).to(device)
optim = torch.optim.Adam(discriminator.parameters(), lr=result.x[3])

for epoch in range(result.x[0]):
    for i, (inputs, targets) in enumerate(train_loader):
        discriminator.zero_grad()
        yhat = discriminator(inputs.float())
        loss = loss_function(yhat, targets.long())
        loss.backward()
        optim.step()


#########################
# Predicting Full Audio #
#########################
# GPU barely too small to handle all the data, so just iterating over it
predictions = pd.DataFrame(columns=['speaker', 'confidence'])
i = 0
while i <= len(audio):
    print(f'{i}:{i+100000} / {len(audio)}')
    
    audio_x = torch.from_numpy(transform.iloc[i:i+100000].values).float()
    
    append = pd.DataFrame({'speaker_id': np.argmax(discriminator(audio_x.to(device)).cpu().detach().numpy(), axis=1),
                           'confidence': np.max(discriminator(audio_x.to(device)).cpu().detach().numpy(), axis=1)})
    append = append.merge(mapped, how='left', left_on='speaker_id', right_on='y')[['speaker', 'confidence']]
    
    predictions = predictions.append(append, ignore_index=True, sort=False)
    
    i+=100000
    
    
predictions = audio[['id', 'second']].merge(predictions, right_index=True, left_index=True, how='outer')
    
conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
con = urllib.parse.quote_plus(conn_str)

engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')

predictions.to_sql(name='predictions', con=engine, schema='dbo', if_exists='replace', index=False)
