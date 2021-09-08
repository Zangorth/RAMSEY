from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from pure_sklearn.map import convert_estimator
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from functools import partial
from skopt import plots
import pyodbc as sql
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
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Helpers')
from cross_validation import CV
import ramsey_helpers as RH
import arbitraryNN

full = False
include_gender = True
optimize = True

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
                  
query = open('Queries\\SpeakerTraining.txt').read()

train_audio = pd.read_sql(query, con)

if full:
    query = open('Queries\\FullAudio.txt').read()
    audio = pd.read_sql(query, con)
    audio = audio.dropna().reset_index(drop=True)
    
if include_gender:
    query = 'SELECT id, [second], gender FROM RAMSEY.prediction.Gender'
    gender = pd.read_sql(query, con)
    gender = gender.merge(pd.get_dummies(gender['gender']), left_index=True, right_index=True, how='left')
    gender = gender.drop('gender', axis=1)
    

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

x = train_audio[['id', 'second']].merge(x, left_index=True, right_index=True)
x = x.merge(years, left_index=True, right_index=True)
x = x.merge(days, left_index=True, right_index=True)
x = x.merge(interaction, left_index=True, right_index=True)
x['pca'] = pca
x['km'] = km

if include_gender:
    x = x.merge(gender, on=['id', 'second'], how='left')

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
    
    if include_gender:
        audio = audio.merge(gender, on=['id', 'second'], how='left')
    
    audio = audio.dropna().reset_index(drop=True)

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
    skopt.space.Integer(0, 1, name='gender'),
    skopt.space.Integer(1, 5, name='layers'),
    skopt.space.Integer(1, 100, name='epochs'),
    skopt.space.Real(0.0001, 0.2, name='drop', prior='log-uniform'),
    skopt.space.Real(0.00001, 0.04, name='lr_nn', prior='log-uniform'),
    skopt.space.Integer(32, 1000, name='batch_size')
    ]

space = space + [skopt.space.Integer(2**2, 2**10, name=f'neuron_{i}') for i in range(5)]

i = 0
@skopt.utils.use_named_args(space)
def net(model, lags, leads, km, pca, gender, over=False, **kwargs):
    lx = RH.shift(x, 'id', lags, leads)
    lx = lx.loc[y != -1].drop('id', axis=1).dropna()
    ly = y.loc[lx.index]
    
    lx, ly = lx.reset_index(drop=True), ly.reset_index(drop=True)
    
    if not pca:
        lx = lx.drop('pca', axis=1)
        
    if not km:
        lx = lx.drop('km', axis=1)
        
    if include_gender and not gender:
        lx = lx.drop(['M', 'N', 'W'], axis=1)
    
    neurons = None if 'neuron_0' not in kwargs else [kwargs[key] for key in kwargs if 'neuron' in key]
    
    if model == 'nn':
        discriminator = arbitraryNN.Discriminator(drop=kwargs['drop'], neurons=neurons, lr_nn=kwargs['lr_nn'],
                                                  epochs=kwargs['epochs'], layers=kwargs['layers'], batch_size=int(kwargs['batch_size']))
        
    elif model == 'logit':
        discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
    
    f1 = partial(f1_score, average='macro')
    validator = CV(discriminator, f1)
    out = validator.cv(lx, ly, over=True)
    
    global i
    i += 1
    
    global kwargs_out
    kwargs_out = kwargs
    
    print(f'({i}/{calls}) {model}: {round(out, 3)}')
    return (- 1.0 * out)
        
if optimize:
    result = skopt.forest_minimize(net, space, acq_func='PI', n_initial_points=starts, n_calls=calls)
    
    plt.figure(figsize=(50, 50))
    plots.plot_evaluations(result)
    plt.savefig('SpeakerResults.png', bbox_inches='tight')
    
    print(f'Max F1: {result.fun}')
    
    results = {'model': result.x[0], 'lags': result.x[1], 'leads': result.x[2], 
               'pca': result.x[3], 'km': result.x[4], 'gender': result.x[5]}
    
    i = 6
    for key in kwargs_out:
        results[key] = result.x[i]
        i += 1
    
    results['neurons'] =[results[key] for key in results if 'neuron' in key]
    
    print(results)
    
    pickle.dump(results, open('speaker_results.pkl', 'wb'))
else:
    results = pickle.load(open('speaker_results.pkl', 'rb'))


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
    
if include_gender and results['gender'] == 0:
    x = x.drop(['M', 'N', 'W'], axis=1)

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

print(f'F1 Macro: {validator.cv(x, y, over=True)}\n')
print(avg.mean())

###############
# Predictions #
###############
predictions = pd.DataFrame(columns=['id', 'second', 'rediction'])

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
    discriminator = arbitraryNN.Discriminator(drop=results['drop'], neurons=results['neurons'], lr_nn=results['lr_nn'],
                                              epochs=results['epochs'], layers=results['layers'], batch_size=int(results['batch_size']))
    
    discriminator.fit(x, y)
    
    i = 0
    while i <= len(audio):
        print(f'{i}:{i+100000} / {len(audio)}')
        
        audio_x = audio.iloc[i:i+100000]
        audio_x = RH.shift(audio_x, 'id', results['lags'], results['leads'], exclude=['second'] + exclude + ['2014-Saturday'])
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

predictions.to_sql(name='Speaker', con=engine, schema='prediction', if_exists='replace', index=False)
