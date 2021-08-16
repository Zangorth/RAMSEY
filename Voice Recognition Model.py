from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from pure_sklearn.map import convert_estimator
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn import preprocessing
from pydub.playback import play
from pydub import AudioSegment
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
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
import ramsey_helpers as RH

full = False
optimize = True
check = False

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

SELECT id, [second], [source]
INTO #speakers
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]-1, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]-2, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]-3, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]-4, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]-5, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]+1, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]+2, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]+3, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]+4, [source]
FROM RAMSEY.training.Speaker
UNION
SELECT id, [second]+5, [source]
FROM RAMSEY.training.Speaker

SELECT speaker , YEAR(publish_date) AS 'publish_year', 
    DATENAME(DW, publish_date) AS 'dow', 
    CONCAT(YEAR(publish_date),'-',  DATENAME(DW, publish_date)) AS 'interaction',
    #speakers.[source], code.*
FROM #speakers
LEFT JOIN RAMSEY.training.Speaker AS train
    ON #speakers.id = train.id
    AND #speakers.[second] = train.[second]
LEFT JOIN RAMSEY.dbo.AudioCoding AS code
    ON #speakers.id = code.id
    AND #speakers.[second] = code.[second]
LEFT JOIN RAMSEY.dbo.metadata AS meta
    ON #speakers.id = meta.id
WHERE code.id IS NOT NULL
    AND #speakers.[source] != 'test'
ORDER BY id, [second]
'''

train_audio = pd.read_sql(query, con)

if full:
    query = '''
    SELECT YEAR(publish_date) AS 'publish_year', 
        DATENAME(DW, publish_date) AS 'dow', 
        CONCAT(YEAR(publish_date),'-',  DATENAME(DW, publish_date)) AS 'interaction',
        audio.*
    FROM RAMSEY.dbo.AudioCoding AS audio
    RIGHT JOIN RAMSEY.dbo.metadata AS meta
        ON audio.id = meta.id
    WHERE meta.seconds <= 900
    '''
    audio = pd.read_sql(query, con)
    audio = audio.dropna().reset_index(drop=True)

con.close()

##############
# Clean Data #
##############
train_audio['y'] = train_audio['speaker'].astype('category').cat.codes
y = train_audio['y']

train_audio = train_audio.drop('source', axis=1)

mapped = train_audio[['y', 'speaker']].drop_duplicates().reset_index(drop=True)
mapped = mapped.loc[mapped.y != -1].sort_values('y').reset_index(drop=True)

years = pd.get_dummies(train_audio['publish_year'])
years.columns = [str(col) for col in years.columns]

days = pd.get_dummies(train_audio['dow'])
interaction = pd.get_dummies(train_audio['interaction'])

exclude = list(years.columns) + list(days.columns) + list(interaction.columns)

x = train_audio.drop(['id', 'second', 'speaker', 'y', 'publish_year', 'dow', 'interaction'], axis=1)
save_cols = x.columns

scaler = preprocessing.StandardScaler().fit(x)
x = pd.DataFrame(scaler.transform(x), columns=save_cols)

pca = PCA(4)
pca = np.argmax(pca.fit_transform(x), axis=1)

km = KMeans(6)
km = np.argmax(km.fit_transform(x), axis=1)

x = train_audio[['id']].merge(x, left_index=True, right_index=True)
x = x.merge(years, left_index=True, right_index=True)
x = x.merge(days, left_index=True, right_index=True)
x = x.merge(interaction, left_index=True, right_index=True)
x['pca'] = pca
x['km'] = km

if full:
    transform = audio.drop(['id', 'second', 'publish_year', 'dow', 'interaction'], axis=1)
    save_cols = transform.columns
    
    transform = pd.DataFrame(scaler.transform(transform), columns=save_cols)
    
    pca = PCA(4)
    pca = np.argmax(pca.fit_transform(transform), axis=1)
    
    km = KMeans(6)
    km = np.argmax(km.fit_transform(transform), axis=1)
    
    years = pd.get_dummies(audio['publish_year'])
    years.columns = [str(col) for col in years.columns]
    
    days = pd.get_dummies(audio['dow'])
    interaction = pd.get_dummies(audio['interaction'])
    
    audio = audio[['id', 'second']].merge(transform, right_index=True, left_index=True)
    audio = audio.merge(years, right_index=True, left_index=True)
    audio = audio.merge(days, right_index=True, left_index=True)
    audio = audio.merge(interaction, right_index=True, left_index=True)
    
    audio['pca'] = pca
    audio['km'] = km
    
    audio = audio.dropna().reset_index(drop=True)

####################
# Optimize Network #
####################
calls, max_f1 = 25, []
max_shift = 5
space = [
    skopt.space.Categorical(['nn'], name='model'),
    skopt.space.Integer(0, max_shift, name='lags'),
    skopt.space.Integer(0, max_shift, name='leads'),
    skopt.space.Integer(0, 1, name='pca'),
    skopt.space.Integer(0, 1, name='km'),
    skopt.space.Integer(50, 500, name='n_samples'),
    skopt.space.Integer(50, 500, name='n_estimators'),
    skopt.space.Real(0.0001, 0.1, name='lr_gbc', prior='log-uniform'),
    skopt.space.Integer(1, 5, name='layers'),
    skopt.space.Integer(1, 100, name='epochs'),
    skopt.space.Real(0.0001, 0.2, name='drop', prior='log-uniform'),
    skopt.space.Real(0.00001, 0.04, name='lr_nn', prior='log-uniform')
    ]

space = space + [skopt.space.Integer(2**2, 2**10, name=f'transform_{i}') for i in range(5)]
#space = space + [skopt.space.Integer(0, 1, name=f'column_{i}') for i in range((len(x.columns)-1)*(max_lags+1))]

tracker, i = [], 0

lags, leads, km, pca = 3, 3, 1, 0
drop, lr_nn, epochs, layers = 0.001, 0.001, 20, 4
transforms = [132, 64, 32, 16]

@skopt.utils.use_named_args(space)
def net(model, lags, leads, km, pca, n_samples=None, n_estimators=None, lr_gbc=None, max_depth=None, 
        drop=None, lr_nn=None, epochs=None, layers=None, **kwargs):
    lx = RH.shift(x, 'id', lags, leads, exclude=exclude)
    lx = lx.loc[y != -1].drop('id', axis=1).dropna()
    ly = y.loc[lx.index]
    
    lx, ly = lx.reset_index(drop=True), ly.reset_index(drop=True)
    
    if pca == 0:
        lx = lx.drop('pca', axis=1)
        
    if km == 0:
        lx = lx.drop('km', axis=1)
    
    transform = None if 'transform_0' not in kwargs else [kwargs[key] for key in kwargs if 'transform' in key]
    
    f1 = RH.cv(model, lx, ly, n_samples, n_estimators, lr_gbc, max_depth,
               transforms, drop, layers, epochs, lr_nn)
    
    global i
    i += 1
    
    global tracker
    tracker.append([model, lags, n_samples, n_estimators, lr_gbc, max_depth,
                    drop, lr_nn, epochs, layers, transform])
    
    print(f'({i}/{calls}) {model}: {round(f1, 2)}')
    return (- 1.0 * f1)
        
if optimize:
    result = skopt.forest_minimize(net, space, acq_func='PI', n_initial_points=10, n_calls=calls)
    
    features = {'lags': 1, 'leads': 2, 'pca': 3, 'km': 4, 'layers': 5, 'epochs': 6, 'drop': 7, 'lr': 8}
    
    for feature in features:
        print(features[feature])
        vals = [result.x_iters[i][features[feature]] for i in range(len(result.x_iters))]
        sea.distplot(vals, kde=False)
        plt.title(feature)
        plt.show()
        plt.close()
    
    max_f1.append(result)
    print(f'Max F1: {result.fun}')
    
    results = {'model': result.x[0], 'lags': result.x[1], 'leads': result.x[2], 
               'pca': result.x[3], 'km': result.x[4], 'n_samples': result.x[5],
               'n_estimators': result.x[6], 'lr_gbc': result.x[7], 
               'layers': result.x[8], 'epochs': result.x[9], 'drop': result.x[10],
               'lr_nn': result.x[11], 'transform': [result.x[i] for i in range(12, len(result.x))]}
    
    print(results)
    
    pickle.dump(results, open('results.pkl', 'wb'))
else:
    results = pickle.load(open('results.pkl', 'rb'))


####################
# Validate Network #
####################
x = RH.shift(x, 'id', results['lags'], results['leads'], exclude=exclude)
x = x.loc[y != -1].drop('id', axis=1).dropna()
y = y.loc[x.index]

if results['pca'] == 0:
    x = x.drop('pca', axis=1)
    
if results['km'] == 0:
    x = x.drop('km', axis=1)

test_audio = train_audio.iloc[x.index]

x, y = x.reset_index(drop=True), y.reset_index(drop=True)

if results['model'] == 'logit':
    avg = RH.cv_logit(x, y, avg=True)
    print(avg)
    
elif results['model'] == 'nn':
    avg = RH.cv_nn(x, y, results['transform'], results['drop'], results['lr_nn'], 
                   results['epochs'], results['layers'], avg=True)
    avg = avg.merge(mapped, left_on='speaker_id', right_on='y')[['speaker', 'f1']]
    print(avg)
    
    if check:
        f1 = RH.cv_nn(x, y, results['transform'], results['drop'], results['lr_nn'], 
                       results['epochs'], results['layers'], wrong=True)
        
        f1['pred'] = f1.merge(mapped, left_on='pred', right_on='y', how='left')['speaker']
        f1['real'] = f1.merge(mapped, left_on='real', right_on='y', how='left')['speaker']
        

##################################
# Double Check Wrong Predictions #
##################################
if check:
    test_audio = test_audio.iloc[f1['index']]
    test_audio = test_audio[['id', 'second']].reset_index(drop=True)
    test_audio = test_audio.merge(f1[['pred', 'real']], left_index=True, right_index=True)
    test_audio = test_audio.sort_values(['id', 'second']).reset_index(drop=True)
    
    for i in range(len(test_audio)):
        sample = test_audio['id'][i]
        second = test_audio['second'][i]
        label = test_audio['real'][i]
        pred = test_audio['pred'][i]
        
        sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
        
        lead = sound[second*1000-3000:second*1000+3000]
        sound = sound[second*1000:second*1000+1000]
        
        speaker = 0
            
        while speaker == 0:
            play(sound)
    
            speaker = input(f'({sample}: {second}) Speaker {label} or {pred}? ')
            speaker = speaker.upper()
            speaker = 0 if speaker == '0' else speaker
    
            if str(speaker).lower() == 'lead':
                play(lead)
                speaker = 0
                
            elif speaker == 0:
                pass
                
            elif speaker == '':
                speaker = label
            
            else:
                con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                          Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                          Trusted_Connection=yes;''')
                         
                csr = con.cursor()
                query = f'''
                UPDATE RAMSEY.training.Speaker
                SET speaker = '{speaker}'
                WHERE id = {sample} AND second = {second}
                '''
                
                csr.execute(query)
                csr.commit()
                con.close()

###############
# Predictions #
###############
predictions = pd.DataFrame(columns=['id', 'second', 'prediction'])

if results['model'] == 'logit':
    discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
    discriminator.fit(x, y)
    
    discriminator = convert_estimator(discriminator)
    
    preds = []
    i = 0
    while i <= len(audio):
        print(f'{i}:{i+100000} / {len(audio)}')
        
        audio_x = audio.iloc[i:i+100000]
        audio_x = RH.shift(audio_x, 'id', results['lags'], results['leads'], exclude=['second'] + exclude + ['2014-Saturday'])
        audio_x = audio_x.dropna()
        
        audio_x['prediction'] = discriminator.predict(audio_x.drop([col for col in audio_x.columns if col not in x.columns], axis=1).values.tolist())
        predictions = predictions.append(audio_x[['id', 'second', 'prediction']], ignore_index=True, sort=False)
        
        i += 100000
        

elif results['model'] == 'nn':
    torch.cuda.empty_cache()
    save_cols = x.columns
    x, y = torch.from_numpy(x.values).to(device), torch.from_numpy(y.values).to(device)
    train_set = [(x[i], y[i]) for i in range(len(y))]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    
    loss_function = nn.CrossEntropyLoss()
    discriminator = RH.Discriminator(x.shape[1], results['transform'], results['drop'], 9).to(device)
    optim = torch.optim.Adam(discriminator.parameters(), lr=results['lr_nn'])
    
    for epoch in range(results['epochs']):
        for i, (inputs, targets) in enumerate(train_loader):
            discriminator.zero_grad()
            yhat = discriminator(inputs.float())
            loss = loss_function(yhat, targets.long())
            loss.backward()
            optim.step()
    
    discriminator.eval()
    
    i = 0
    while i <= len(audio):
        print(f'{i}:{i+100000} / {len(audio)}')
        
        audio_x = audio.iloc[i:i+100000]
        audio_x = RH.shift(audio_x, 'id', results['lags'], results['leads'], exclude=['second'] + exclude + ['2014-Saturday'])
        audio_x = audio_x.dropna()
        
        torch_x = torch.from_numpy(audio_x.drop([col for col in audio_x.columns if col not in save_cols], axis=1).values).float()
        audio_x['prediction'] = np.argmax(discriminator(torch_x.to(device)).cpu().detach().numpy(), axis=1)
        
        predictions = predictions.append(audio_x[['id', 'second', 'prediction']], ignore_index=True, sort=False)
        
        i+=100000
    
predictions = predictions.merge(mapped, how='left', left_on='prediction', right_on='y')[['id', 'second', 'speaker']]
predictions['id'] = predictions['id'].astype(int)
predictions['second'] = predictions['second'].astype(int)
    
conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
con = urllib.parse.quote_plus(conn_str)

engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')

predictions.to_sql(name='Speaker', con=engine, schema='prediction', if_exists='replace', index=False)
