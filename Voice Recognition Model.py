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
kf = StratifiedKFold()
device = torch.device('cuda:0')

class Discriminator(nn.Module):
    def __init__(self, a, b, c, drop, shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(shape, a),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(b, c),
            nn.ReLU(),
            nn.Linear(c, 9),
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
SET NOCOUNT ON

SELECT id, [second]
INTO #speakers
FROM RAMSEY.dbo.AudioTraining
WHERE speaker != 'M'
UNION
SELECT id, [second]-1
FROM RAMSEY.dbo.AudioTraining
WHERE speaker != 'M'
UNION
SELECT id, [second]-2
FROM RAMSEY.dbo.AudioTraining
WHERE speaker != 'M'

SELECT speaker , YEAR(publish_date) AS 'publish_year', code.*
FROM #speakers
LEFT JOIN RAMSEY.dbo.AudioTraining AS train
    ON #speakers.id = train.id
    AND #speakers.[second] = train.[second]
LEFT JOIN RAMSEY.dbo.AudioCoding AS code
    ON #speakers.id = code.id
    AND #speakers.[second] = code.[second]
LEFT JOIN RAMSEY.dbo.metadata AS meta
    ON #speakers.id = meta.id
WHERE code.id IS NOT NULL AND speaker != 'M'
ORDER BY id, [second]
'''

train_audio = pd.read_sql(query, con)

query = '''
SELECT YEAR(publish_date) AS 'publish_year', 
    audio.*
FROM RAMSEY.dbo.AudioCoding AS audio
RIGHT JOIN RAMSEY.dbo.metadata AS meta
    ON audio.id = meta.id
WHERE meta.seconds <= 900
'''
audio = pd.read_sql(query, con)

con.close()

##############
# Clean Data #
##############
train_audio['y'] = train_audio['speaker'].astype('category').cat.codes
y = train_audio['y']

mapped = train_audio[['y', 'speaker']].drop_duplicates().reset_index(drop=True)
mapped = mapped.loc[mapped.y != -1].sort_values('y').reset_index(drop=True)

years = pd.get_dummies(train_audio['publish_year'])
x = train_audio.drop(['id', 'second', 'speaker', 'y', 'publish_year'], axis=1)

scaler = preprocessing.StandardScaler().fit(x)
x = pd.DataFrame(scaler.transform(x), columns=train_audio.columns[4:-1])
x = train_audio[['id']].merge(x, left_index=True, right_index=True)

lag1 = x.groupby('id').shift(1)
lag1.columns = [f'{col}_lag1' for col in x.columns if col != 'id']

lag2 = x.groupby('id').shift(2)
lag2.columns = [f'{col}_lag2' for col in x.columns if col != 'id']

x = x.merge(lag1, left_index=True, right_index=True)
x = x.merge(lag2, left_index=True, right_index=True)
x = x.merge(years, left_index=True, right_index=True)
x = x.drop('id', axis=1)

transform = pd.DataFrame(scaler.transform(audio.drop(['id', 'second', 'publish_year'], axis=1)), columns=audio.columns[3:])
transform = audio[['id']].merge(transform, right_index=True, left_index=True)

lag1 = transform.groupby('id').shift(1)
lag1.columns = [f'{col}_lag1' for col in transform.columns if col != 'id']

lag2 = transform.groupby('id').shift(2)
lag2.columns = [f'{col}_lag1' for col in transform.columns if col != 'id']
years = pd.get_dummies(audio['publish_year'])

audio = audio[['id', 'second']].merge(transform.drop('id', axis=1), right_index=True, left_index=True)
audio = audio.merge(lag1, right_index=True, left_index=True)
audio = audio.merge(lag2, right_index=True, left_index=True)
audio = audio.merge(years, right_index=True, left_index=True).dropna().reset_index(drop=True)

del [lag1, lag2, years]

####################
# Optimize Network #
####################
space = [
    skopt.space.Integer(1, 50, name='epochs'),
    skopt.space.Integer(2**2, 2**10, name='a'),
    skopt.space.Integer(2**2, 2**10, name='b'),
    skopt.space.Integer(2**2, 2**10, name='c'),
    skopt.space.Real(0.0001, 0.1, name='lr', prior='log-uniform'),
    skopt.space.Real(0.0001, 1, name='drop', prior='log-uniform'),
    skopt.space.Integer(0, 2, name='lags')
    ]


epochs, a, b, drop, lr, lags = 20, 64, 32, 0.15, 0.01, 1

@skopt.utils.use_named_args(space)
def net(epochs, a, b, c, lr, drop, lags):
    if lags == 0:
        lx = x[[col for col in x.columns if 'lag' not in str(col)]]
        lx = lx.loc[y != -1].dropna()
        ly = y.loc[lx.index]
    
    elif lags == 1:
        lx = x[[col for col in x.columns if 'lag2' not in str(col)]]
        lx = lx.loc[y != -1].dropna()
        ly = y.loc[lx.index]
    
    else:
        lx = x.loc[y != -1].dropna()
        ly = y.loc[lx.index]
    
    lx, ly = lx.reset_index(drop=True), ly.reset_index(drop=True)
    
    f1 = []
    for i in range(0, 5):
        x_train = lx.groupby(ly).apply(lambda x: x.sample(frac=0.2))
        x_train.index = [x_train.index[i][1] for i in range(len(x_train))]
        x_train = x_train.sort_index()
        
        x_test = lx.loc[~lx.index.isin(x_train.index)]
        y_train = ly.loc[x_train.index]
        y_test = ly.loc[~ly.index.isin(x_train.index)]
        
        x_train, x_test = torch.from_numpy(x_train.values).to(device), torch.from_numpy(x_test.values).to(device)
        y_train, y_test = torch.from_numpy(y_train.values).to(device), torch.from_numpy(y_test.values).to(device)
        
        train_set = [(x_train[i].to(device), y_train[i].to(device)) for i in range(len(y_train))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**7, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = Discriminator(a, b, c, drop, lx.shape[1]).to(device)
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
    
    print(round(np.mean(f1), 2))
    return (- 1.0 * np.mean(f1))
        
optimize = True
if optimize:
    result = skopt.forest_minimize(net, space, acq_func='PI', n_calls=100, x0=[36,949,846,998,0.00011,0.00032,2])
    print(f'Max F1: {result.fun}')
    print(f'Parameters: {result.x}')
    plots.plot_evaluations(result)
    
    result = {'epochs': result.x[0], 'a': result.x[1], 'b': result.x[2],
              'c': result.x[3], 'lr': result.x[4], 'drop': result.x[5],
              'lags': result.x[6]}
    
else:
    result = {'epochs': 43, 'a': 748, 'b': 264, 'c': 298, 
              'lr': 0.000146, 'drop': 0.00541, 'lags': 2}


####################
# Validate Network #
####################
if result['lags'] == 0:
    x = x[[col for col in x.columns if 'lag' not in str(col)]]
    x = x.loc[y != -1].dropna()
    y = y.loc[x.index]

elif result['lags'] == 1:
    x = x[[col for col in x.columns if 'lag2' not in str(col)]].dropna()
    x = x.loc[y != -1].dropna()
    y = y.loc[x.index]

else:
    x = x.loc[y != -1].dropna()
    y = y.loc[x.index]
    
x, y = x.reset_index(drop=True), y.reset_index(drop=True)

f1 = []
for i in range(0, 5):
    x_train = x.groupby(y).apply(lambda x: x.sample(frac=0.2))
    x_train.index = [x_train.index[i][1] for i in range(len(x_train))]
    x_train = x_train.sort_index()
    
    x_test = x.loc[~x.index.isin(x_train.index)]
    y_train = y.loc[x_train.index]
    y_test = y.loc[~y.index.isin(x_train.index)]
    
    x_train, x_test = torch.from_numpy(x_train.values).to(device), torch.from_numpy(x_test.values).to(device)
    y_train, y_test = torch.from_numpy(y_train.values).to(device), torch.from_numpy(y_test.values).to(device)
    
    train_set = [(x_train[i].to(device), y_train[i].to(device)) for i in range(len(y_train))]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2**7, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    discriminator = Discriminator(result['a'], result['b'], result['c'], result['drop'], x.shape[1]).to(device)
    optim = torch.optim.Adam(discriminator.parameters(), lr=result['lr'])

    for epoch in range(result['epochs']):
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


#################
# Train Network #
#################
torch.cuda.empty_cache()
x, y = torch.from_numpy(x.values).to(device), torch.from_numpy(y.values).to(device)
train_set = [(x[i], y[i]) for i in range(len(y))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

loss_function = nn.CrossEntropyLoss()
discriminator = Discriminator(result['a'], result['b'], result['c'], result['drop'], x.shape[1]).to(device)
optim = torch.optim.Adam(discriminator.parameters(), lr=result['lr'])

for epoch in range(result['epochs']):
    for i, (inputs, targets) in enumerate(train_loader):
        discriminator.zero_grad()
        yhat = discriminator(inputs.float())
        loss = loss_function(yhat, targets.long())
        loss.backward()
        optim.step()


#########################
# Predicting Full Audio #
#########################
if result['lags'] == 0:
    audio = audio[[col for col in audio.columns if 'lag' not in str(col)]].dropna().reset_index(drop=True)

elif result['lags'] == 1:
    audio = audio[[col for col in audio.columns if 'lag2' not in str(col)]].dropna().reset_index(drop=True)

else:
    audio = audio.dropna().reset_index(drop=True)


# GPU barely too small to handle all the data, so just iterating over it
predictions = pd.DataFrame(columns=['speaker', 'confidence'])
i = 0
while i <= len(audio):
    print(f'{i}:{i+100000} / {len(audio)}')
    
    audio_x = torch.from_numpy(audio.drop(['id', 'second'], axis=1)[i:i+100000].values).float()
    
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
