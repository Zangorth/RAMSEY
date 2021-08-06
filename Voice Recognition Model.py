from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sklearn import preprocessing
from datetime import datetime
import seaborn as sea
import pyodbc as sql
from torch import nn
import pandas as pd
import numpy as np
import warnings
import urllib
import pickle
import skopt
import torch
import sys

sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
import ramsey_helpers as RH

year_continuous = True
full = True
optimize = False

warnings.filterwarnings('error', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
device = torch.device('cuda:0')

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
UNION
SELECT id, [second]-1
FROM RAMSEY.dbo.AudioTraining
UNION
SELECT id, [second]-2
FROM RAMSEY.dbo.AudioTraining

SELECT speaker , YEAR(publish_date) AS 'publish_year', 
    [source], code.*
FROM #speakers
LEFT JOIN RAMSEY.dbo.AudioTraining AS train
    ON #speakers.id = train.id
    AND #speakers.[second] = train.[second]
LEFT JOIN RAMSEY.dbo.AudioCoding AS code
    ON #speakers.id = code.id
    AND #speakers.[second] = code.[second]
LEFT JOIN RAMSEY.dbo.metadata AS meta
    ON #speakers.id = meta.id
WHERE code.id IS NOT NULL
ORDER BY id, [second]
'''

train_audio = pd.read_sql(query, con)

if full:
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

semi = train_audio['source']
train_audio = train_audio.drop('source', axis=1)

mapped = train_audio[['y', 'speaker']].drop_duplicates().reset_index(drop=True)
mapped = mapped.loc[mapped.y != -1].sort_values('y').reset_index(drop=True)

if year_continuous:
    years = pd.DataFrame(datetime.today().year - train_audio.publish_year)
    years['publish_year_2'] = years.publish_year ** 2
    years['publish_year_3'] = years.publish_year ** 3
    years.columns = ['year1', 'year2', 'year3']
    
    x = train_audio.drop(['id', 'second', 'speaker', 'y', 'publish_year'], axis=1)
    x = x.merge(years, left_index=True, right_index=True)
    
    save_cols = x.columns
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=save_cols)
    x = train_audio[['id']].merge(x, left_index=True, right_index=True)

else:
    years = pd.get_dummies(train_audio['publish_year'])
    x = train_audio.drop(['id', 'second', 'speaker', 'y', 'publish_year'], axis=1)
    
    save_cols = x.columns
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=save_cols)
    x = train_audio[['id']].merge(x, left_index=True, right_index=True)

    x = x.merge(years, left_index=True, right_index=True)

if full:
    if year_continuous:
        years = pd.DataFrame(datetime.today().year - audio.publish_year)
        years['publish_year_2'] = years.publish_year ** 2
        years['publish_year_3'] = years.publish_year ** 3
        years.columns = ['year1', 'year2', 'year3']
        
        transform = audio.drop(['id', 'second', 'publish_year'], axis=1).merge(years, left_index=True, right_index=True)
        
        save_cols = transform.columns
        
        transform = pd.DataFrame(scaler.transform(transform), columns=save_cols)
        
        audio = audio[['id', 'second']].merge(transform, right_index=True, left_index=True)
            
    
    else:
        transform = audio.drop(['id', 'second', 'publish_year'], axis=1)
        save_cols = transform.columns
        
        transform = pd.Dataframe(scaler.transform(transform), columns=save_cols)
        
        years = pd.get_dummies(audio['publish_year'])
        
        audio = audio[['id', 'second']].merge(transform, right_index=True, left_index=True)
        audio = audio.merge(years, right_index=True, left_index=True).dropna().reset_index(drop=True)

####################
# Optimize Network #
####################
calls, max_f1 = 100, []
max_shift = 5
space = [
    skopt.space.Categorical(['nn'], name='model'),
    skopt.space.Integer(0, max_shift, name='lags'),
    skopt.space.Integer(0, max_shift, name='leads'),
    #skopt.space.Integer(50, 500, name='n_samples'),
    #skopt.space.Integer(50, 500, name='n_estimators'),
    #skopt.space.Real(0.0001, 0.1, name='lr_gbc', prior='log-uniform'),
    skopt.space.Integer(1, 3, name='layers'),
    skopt.space.Integer(1, 100, name='epochs'),
    skopt.space.Real(0.0001, 0.2, name='drop', prior='log-uniform'),
    skopt.space.Real(0.00001, 0.04, name='lr_nn', prior='log-uniform')
    ]

space = space + [skopt.space.Integer(2**2, 2**10, name=f'transform_{i}') for i in range(10)]
#space = space + [skopt.space.Integer(0, 1, name=f'column_{i}') for i in range((len(x.columns)-1)*(max_lags+1))]

tracker, i = [], 0

@skopt.utils.use_named_args(space)
def net(model, lags, leads, n_samples=None, n_estimators=None, lr_gbc=None, max_depth=None, 
        drop=None, lr_nn=None, epochs=None, layers=None, **kwargs):
    lx = RH.shift(x, 'id', lags, leads, exclude=years.columns)
    lx = lx.loc[y != -1].drop('id', axis=1).dropna()
    ly, ls = y.loc[lx.index], semi[lx.index]
    
    lx, ly, ls = lx.reset_index(drop=True), ly.reset_index(drop=True), ls.reset_index(drop=True)
    
    if 'column_0' in kwargs:
        select_cols = [kwargs[key] for key in kwargs if 'column' in key]
        select_cols = [lx.columns[i] for i in range(len(lx.columns)) if select_cols[i] == 1]
        lx = lx[select_cols]
        
    else:
        select_cols = None
    
    transform = None if 'transform_0' not in kwargs else [kwargs[key] for key in kwargs if 'transform' in key]
    
    if model == 'logit':
        f1 = RH.cv_logit(lx, ly, ls)
        
    elif model == 'rfc':
        f1 = RH.cv_rfc(lx, ly, ls, n_samples)
        
    elif model == 'gbc':
        f1 = RH.cv_gbc(lx, ly, ls, n_estimators, lr_gbc, max_depth)
        
    elif model == 'nn':
        f1 = RH.cv_nn(lx, ly, ls, transform, drop, lr_nn, epochs, layers=layers)
        #cv_nn(x, y, semi, transforms, drop, lr_nn, epochs, output=10)
        
    else:
        print('Improperly Specified Model')
        f1 = 0
    
    global i
    i += 1
    
    global tracker
    tracker.append([model, lags, n_samples, n_estimators, lr_gbc, max_depth,
                    drop, lr_nn, epochs, layers, transform, select_cols])
    
    print(f'({i}/{calls}) {model}: {round(f1, 2)}')
    return (- 1.0 * f1)
        
if optimize:
    result = skopt.forest_minimize(net, space, acq_func='PI', n_initial_points=10, n_calls=calls, n_jobs=-1)
    
    features = {'lags': 1, 'leads': 2, 'layers': 3, 'epochs': 4, 'drop': 5, 'lr': 6}
    
    for feature in features:
        print(features[feature])
        vals = [result.x_iters[i][features[feature]] for i in range(len(result.x_iters))]
        sea.distplot(vals, kde=False)
        plt.title(feature)
        plt.show()
        plt.close()
    
    max_f1.append(result)
    print(f'Max F1: {result.fun}')
    print(f'Parameters: {result.x}')
    
    results = {'model': result.x[0], 'lags': result.x[1], 'leads': result.x[2], 
               'layers': result.x[3], 'epochs': result.x[4], 'drop': result.x[5],
               'lr_nn': result.x[6], 'transform': [result.x[i] for i in range(7, len(result.x))]}
    
    pickle.dump(results, open('results.pkl', 'wb'))
else:
    results = pickle.load(open('results.pkl', 'rb'))


####################
# Validate Network #
####################
x = RH.shift(x, 'id', results['lags'], results['leads'], exclude=years.columns)
x = x.loc[y != -1].drop('id', axis=1).dropna()
y, semi = y.loc[x.index], semi[x.index]

x, y, semi = x.reset_index(drop=True), y.reset_index(drop=True), semi.reset_index(drop=True)


f1 = RH.cv_nn(x, y, semi, results['transform'], results['drop'], results['lr_nn'], 
              results['epochs'], results['layers'], prin=True)
    


#################
# Train Network #
#################
torch.cuda.empty_cache()
x, y = torch.from_numpy(x.values).to(device), torch.from_numpy(y.values).to(device)
train_set = [(x[i], y[i]) for i in range(len(y))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

loss_function = nn.CrossEntropyLoss()
discriminator = RH.Discriminator(x.shape[1], results['transform'], results['drop'], 10).to(device)
optim = torch.optim.Adam(discriminator.parameters(), lr=results['lr_nn'])

for epoch in range(results['epochs']):
    for i, (inputs, targets) in enumerate(train_loader):
        discriminator.zero_grad()
        yhat = discriminator(inputs.float())
        loss = loss_function(yhat, targets.long())
        loss.backward()
        optim.step()

discriminator.eval()

#########################
# Predicting Full Audio #
#########################

# GPU barely too small to handle all the data, so just iterating over it
predictions = pd.DataFrame(columns=['speaker', 'confidence'])
i = 0
while i <= len(audio):
    print(f'{i}:{i+100000} / {len(audio)}')
    
    audio_x = audio.iloc[i:i+100000]
    audio_x = RH.shift(audio_x, 'id', results['lags'], results['leads'], exclude=['second'] + list(years.columns))
    audio_x = audio_x.dropna()
    
    audio_x = torch.from_numpy(audio_x.drop(['id', 'second'], axis=1).values).float()
    
    append = pd.DataFrame({'speaker_id': np.argmax(discriminator(audio_x.to(device)).cpu().detach().numpy(), axis=1),
                           'confidence': np.max(discriminator(audio_x.to(device)).cpu().detach().numpy(), axis=1)})
    append = append.merge(mapped, how='left', left_on='speaker_id', right_on='y')[['speaker', 'confidence']]
    
    predictions = predictions.append(append, ignore_index=True, sort=False)
    
    i+=100000
    
    
predictions = audio[['id', 'second']].merge(predictions, right_index=True, left_index=True, how='right')

conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
con = urllib.parse.quote_plus(conn_str)

engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')

predictions.to_sql(name='predictions', con=engine, schema='dbo', if_exists='replace', index=False)
