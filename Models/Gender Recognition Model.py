from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from pure_sklearn.map import convert_estimator
from helpers.cross_validation import CV
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from helpers import arbitraryNN
from functools import partial
from ramsey import ramsey
import pyodbc as sql
import pandas as pd
import numpy as np
import warnings
import urllib
import pickle
import skopt
import torch
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
warnings.filterwarnings('error', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
device = torch.device('cuda:0')

local, optimize = True, False
username = 'zangorth'
password = open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\password.txt', 'r').read()

#############
# Read Data #
#############
if local:
    connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                         'Server=ZANGORTH\HOMEBASE;DATABASE=HomeBase;' +
                         'Trusted_Connection=yes;')
else:
    connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                         'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                         f'UID={username};PWD={password};')
con = sql.connect(connection_string)
query = open('Queries\\GenderTraining.txt').read()
panda = pd.read_sql(query, con)
con.close()


##############
# Clean Data #
##############
panda['y'] = panda['gender'].astype('category').cat.codes
y = panda['y']

mapped = panda[['y', 'gender']].drop_duplicates().reset_index(drop=True)
mapped = mapped.loc[mapped.y != -1].sort_values('y').reset_index(drop=True)

x = panda.drop(['channel', 'id', 'second', 'gender', 'y'], axis=1)
save_cols = x.columns

scaler = preprocessing.StandardScaler().fit(x)
x = pd.DataFrame(scaler.transform(x), columns=save_cols)

pca = PCA(4)
pca = pca.fit(x)
x_pca = np.argmax(pca.transform(x), axis=1)

km = KMeans(6)
km = km.fit(x)
x_km = np.argmax(km.transform(x), axis=1)

channels = pd.get_dummies(panda['channel'], prefix='channel')

x = panda[['channel', 'id']].merge(x, left_index=True, right_index=True)
x = x.merge(channels, how='left', left_index=True, right_index=True)

x['pca'] = x_pca
x['km'] = x_km

####################
# Optimize Network #
####################
starts, calls = 20, 100
kwargs_out = None

max_shift = 5
space = [
    skopt.space.Categorical(['nn'], name='model'),
    skopt.space.Integer(0, max_shift, name='lags'),
    skopt.space.Integer(0, max_shift, name='leads'),
    skopt.space.Integer(0, 1, name='pca'),
    skopt.space.Integer(0, 1, name='km'),
    skopt.space.Integer(1, 5, name='layers'),
    skopt.space.Integer(1, 100, name='epochs'),
    skopt.space.Real(0.0001, 0.2, name='drop', prior='log-uniform'),
    skopt.space.Real(0.00001, 0.04, name='lr_nn', prior='log-uniform'),
    skopt.space.Integer(32, 1000, name='batch_size')
    ]

space = space + [skopt.space.Integer(2**2, 2**10, name=f'neuron_{i}') for i in range(5)]

i = 0
@skopt.utils.use_named_args(space)
def net(model, lags, leads, km, pca, over=False, **kwargs):
    lx = ramsey.shift(x, ['channel', 'id'], lags, leads)
    lx = lx.loc[y != -1].drop(['channel', 'id'], axis=1).dropna()
    ly = y.loc[lx.index]
    
    lx, ly = lx.reset_index(drop=True), ly.reset_index(drop=True)
    
    if not pca:
        lx = lx.drop('pca', axis=1)
        
    if not km:
        lx = lx.drop('km', axis=1)
    
    neurons = None if 'neuron_0' not in kwargs else [kwargs[key] for key in kwargs if 'neuron' in key]
    
    if model == 'nn':
        discriminator = arbitraryNN.Discriminator(drop=kwargs['drop'], neurons=neurons, lr_nn=kwargs['lr_nn'],
                                                  epochs=kwargs['epochs'], layers=kwargs['layers'], batch_size=int(kwargs['batch_size']))
        
    elif model == 'logit':
        discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
    
    f1 = partial(f1_score, average='macro')
    validator = CV(discriminator, f1, lower_bound=True)
    out = validator.cv(lx, ly, over=True)
    
    global i
    i += 1
    
    global kwargs_out
    kwargs_out = kwargs
    
    print(f'({i}/{calls}) {model}: {round(out, 3)}')
    return (- 1.0 * out)
        
if optimize:
    result = skopt.forest_minimize(net, space, acq_func='PI', n_initial_points=starts, n_calls=calls)
    
    print(f'Max F1: {result.fun}')
    
    results = {'model': result.x[0], 'lags': result.x[1], 'leads': result.x[2], 
               'pca': result.x[3], 'km': result.x[4]}
    
    i = 5
    for key in kwargs_out:
        results[key] = result.x[i]
        i += 1
    
    results['neurons'] =[results[key] for key in results if 'neuron' in key]
    
    print(results)
    
    pickle.dump(results, open('gender_results.pkl', 'wb'))
else:
    results = pickle.load(open('gender_results.pkl', 'rb'))
    
####################
# Validate Network #
####################
x = ramsey.shift(x, ['channel', 'id'], results['lags'], results['leads'])
x = x.loc[y != -1].drop(['channel', 'id'], axis=1).dropna()
y = y.loc[x.index]

if results['pca'] == 0:
    x = x.drop('pca', axis=1)
    
if results['km'] == 0:
    x = x.drop('km', axis=1)

x, y = x.reset_index(drop=True), y.reset_index(drop=True)

if results['model'] == 'nn':
    discriminator = arbitraryNN.Discriminator(drop=results['drop'], neurons=results['neurons'], lr_nn=results['lr_nn'],
                                              epochs=results['epochs'], layers=results['layers'], batch_size=int(results['batch_size']))
    
elif results['model'] == 'logit':
    discriminator = LogisticRegression(max_iter=500, fit_intercept=False)

f1 = partial(f1_score, average=None)
validator = CV(discriminator, f1)
avg = pd.DataFrame(validator.cv(x, y, over=True, full=True), columns=mapped['gender'])

f1 = partial(f1_score, average='macro')
validator = CV(discriminator, f1)

print(f'F1 Macro: {validator.cv(x, y, cv=100, over=True)}\n')
print(avg.mean())

###############
# Predictions #
###############
if results['model'] == 'nn':
    discriminator = arbitraryNN.Discriminator(drop=results['drop'], neurons=results['neurons'], lr_nn=results['lr_nn'],
                                              epochs=results['epochs'], layers=results['layers'], batch_size=int(results['batch_size']))
    
elif results['model'] == 'logit':
    discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
    
discriminator.fit(x, y)

if local:
    connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                         'Server=ZANGORTH\HOMEBASE;DATABASE=HomeBase;' +
                         'Trusted_Connection=yes;')
else:
    connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                         'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                         f'UID={username};PWD={password};')
con = sql.connect(connection_string)
query = '''SELECT channel, id FROM ramsey.metadata'''
iterations = pd.read_sql(query, con)

for i in range(len(iterations)):
    query = f'''
    SELECT DATEDIFF(MONTH, publish_date, GETDATE()) AS 'age', audio.*
    FROM ramsey.audio
    LEFT JOIN ramsey.metadata
        ON audio.channel = metadata.channel
        AND audio.id = metadata.id
    WHERE audio.channel = '{iterations['channel'][i]}' AND audio.id = {iterations['id'][i]}
    '''
    
    data = pd.read_sql(query, con)
    
    data_x = data.drop(['channel', 'id', 'second'], axis=1)
    save_cols = data_x.columns
    
    data_x = pd.DataFrame(scaler.transform(data_x), columns=save_cols)
    
    data_x_pca = np.argmax(pca.transform(data_x), axis=1)
    data_x_km = np.argmax(km.transform(data_x), axis=1)
    
    data_x = data[['channel', 'id']].merge(data_x, left_index=True, right_index=True)
    
    for channel in channels.columns:
        data_x[channel] = 1 if iterations['channel'][i] in channel else 0 
    
    data_x['pca'] = data_x_pca
    data_x['km'] = data_x_km
    
    data_x = ramsey.shift(data_x, ['channel', 'id'], results['lags'], results['leads'])
    data_x = data_x.drop(['channel', 'id'], axis=1).dropna()
    
    preds = pd.DataFrame({'y': discriminator.predict(data_x)}, index=data_x.index)
    preds = preds.merge(mapped, how='left', on='y').drop('y', axis=1)
    
    new = data[['channel', 'id', 'second']].merge(preds, left_index=True, right_index=True)
    
    
    
    new = pd.DataFrame({'channel': iterations['channel'][i], ''
                        
                        discriminator.predict(data_x)
    
        
        
    
    


if results['model'] == 'logit':
    discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
    discriminator.fit(x, y)
    
    discriminator = convert_estimator(discriminator)
    
    preds = []
    i = 0
    while i <= len(audio):
        print(f'{i}:{i+100000} / {len(audio)}')
        
        audio_x = audio.iloc[i:i+100000]
        audio_x = RH.shift(audio_x, 'id', results['lags'], results['leads'])
        audio_x = audio_x.dropna()
        
        audio_x['prediction'] = discriminator.predict(audio_x.drop([col for col in audio_x.columns if col not in x.columns], axis=1).values.tolist())
        predictions = predictions.append(audio_x[['id', 'second', 'prediction']], ignore_index=True, sort=False)
        
        i += 100000
        

elif results['model'] == 'nn':
    discriminator = arbitraryNN.Discriminator(drop=results['drop'], neurons=results['neurons'], lr_nn=results['lr_nn'],
                                              epochs=results['epochs'], layers=results['layers'], batch_size=int(results['batch_size']))
    
    discriminator.fit(x, y)
    
    i = 0
    while i <= len(audio):
        print(f'{i}:{i+100000} / {len(audio)}')
        
        audio_x = audio.iloc[i:i+100000]
        audio_x = RH.shift(audio_x, 'id', results['lags'], results['leads'])
        audio_x = audio_x.dropna()
        
        audio_x['prediction'] = discriminator.predict(audio_x.drop([col for col in audio_x.columns if col not in x.columns], axis=1))
        predictions = predictions.append(audio_x[['id', 'second', 'prediction']], ignore_index=True, sort=False)
        
        i+=100000
    
predictions = predictions.merge(mapped, how='left', left_on='prediction', right_on='y')[['id', 'second', 'gender']]
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

predictions.to_sql(name='Gender', con=engine, schema='prediction', if_exists='replace', index=False)