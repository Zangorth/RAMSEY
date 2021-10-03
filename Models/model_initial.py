from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from helpers.cross_validation import CV
from sklearn.decomposition import PCA
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
import pickle
import skopt
import torch
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
warnings.filterwarnings('error', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
device = torch.device('cuda:0')

local, optimize = True, True
model = 'speaker'

#############
# Read Data #
#############
connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                     'Server=ZANGORTH;DATABASE=HomeBase;' +
                     'Trusted_Connection=yes;')
con = sql.connect(connection_string)
query = open('Queries\\training_initial.txt').read().format(model)
panda = pd.read_sql(query, con)
con.close()

##############
# Clean Data #
##############
panda['y'] = panda[model].astype('category').cat.codes
y = panda['y']
slices = panda['slice']

mapped = panda[['y', model]].drop_duplicates().reset_index(drop=True)
mapped = mapped.loc[mapped.y != -1].sort_values('y').reset_index(drop=True)

x = panda.drop(['channel', 'publish_date', 'random_id', 'second', model, 'y', 'slice'], axis=1)

scaler = preprocessing.StandardScaler().fit(x)
pickle.dump(scaler, open('Pickles/scaler.pkl', 'wb'))
x = pd.DataFrame(scaler.transform(x), columns=x.columns)

pca = PCA(4).fit(x)
pickle.dump(pca, open('Pickles/pca.pkl', 'wb'))
x_pca = np.argmax(pca.transform(x), axis=1)

km = KMeans(6).fit(x)
pickle.dump(km, open('Pickles/km.pkl', 'wb'))
x_km = np.argmax(km.transform(x), axis=1)

############
# Pipeline #
############
def pipeline(dataframe, lags, leads, channels):
    scaler = pickle.load(open('Pickles/scaler.pkl', 'rb'))
    pca = pickle.load(open('Pickles/pca.pkl', 'rb'))
    km = pickle.load(open('Pickles/km.pkl', 'rb'))
    
    x = dataframe.drop(['channel', 'publish_date', 'random_id', 'second'], axis=1)
    
    x = pd.DataFrame(scaler.transform(x), columns=x.columns)
    x_pca = np.argmax(pca.transform(x), axis=1)
    x_km = np.argmax(km.transform(x), axis=1)
    
    x = dataframe[['channel', 'publish_date', 'random_id']].merge(x, how='left', left_index=True, right_index=True)
    
    x = ramsey.shift(x, group=['channel', 'publish_date', 'random_id'], lags=lags, leads=leads)
        
    for channel in channels:
        x[f'channel_{channel}'] = np.where(x['channel'] == channel, 1, 0)
    
    x['pca'] = x_pca
    x['km'] = x_km    
    
    return x

###################
# Custom Splitter #
###################
def ramsey_split(x, y, slices):
    ramsey_x = x.loc[slices == 'ramsey-all-all']
    ramsey_y = y.loc[ramsey_x.index]
    
    x_train, x_test, y_train, y_test = split(ramsey_x, ramsey_y, test_size=0.05, stratify=ramsey_y)
    x_train = x.loc[~x.index.isin(x_test.index)]
    y_train = y.loc[~y.index.isin(x_test.index)]
    
    return [x_train, x_test, y_train, y_test]

####################
# Optimize Network #
####################
x = panda.drop([model, 'y', 'slice'], axis=1)

starts, calls = 20, 50
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
    lx = pipeline(x, lags, leads, set(x['channel']))
    lx = lx.loc[y != -1].drop(['channel', 'publish_date', 'random_id'], axis=1).dropna()
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
    splitter = partial(split, test_size=0.15, stratify=ly)
    validator = CV(discriminator, f1, splitter, lower_bound=True)
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
    
    pickle.dump(results, open(f'Pickles/{model}_results.pkl', 'wb'))
else:
    results = pickle.load(open(f'Pickles/{model}_results.pkl', 'rb'))
    
####################
# Validate Network #
####################
x = pipeline(x, results['lags'], results['leads'], set(panda['channel']))
x = x.loc[y != -1].drop(['channel', 'publish_date', 'random_id'], axis=1).dropna()
y = y.loc[x.index]
slices = slices.loc[x.index]

if results['pca'] == 0:
    x = x.drop('pca', axis=1)
    
if results['km'] == 0:
    x = x.drop('km', axis=1)

x, y, slices = x.reset_index(drop=True), y.reset_index(drop=True), slices.reset_index(drop=True)

if results['model'] == 'nn':
    discriminator = arbitraryNN.Discriminator(drop=results['drop'], neurons=results['neurons'], lr_nn=results['lr_nn'],
                                              epochs=results['epochs'], layers=results['layers'], batch_size=int(results['batch_size']))
    
elif results['model'] == 'logit':
    discriminator = LogisticRegression(max_iter=500, fit_intercept=False)

f1 = partial(f1_score, average=None)
splitter = partial(ramsey_split, slices=slices)
validator = CV(discriminator, f1, splitter)
avg = pd.DataFrame(validator.cv(x, y, over=True, full=True), columns=mapped[model])

print(f'F1 Macro: {avg.mean().mean()}\n')
print(f'{avg.mean()}\n')

###############
# Predictions #
###############
if results['model'] == 'nn':
    discriminator = arbitraryNN.Discriminator(drop=results['drop'], neurons=results['neurons'], lr_nn=results['lr_nn'],
                                              epochs=results['epochs'], layers=results['layers'], batch_size=int(results['batch_size']))
    
elif results['model'] == 'logit':
    discriminator = LogisticRegression(max_iter=500, fit_intercept=False)
    
discriminator.fit(x, y)

connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                     'Server=ZANGORTH;DATABASE=HomeBase;' +
                     'Trusted_Connection=yes;')
con = sql.connect(connection_string)
csr = con.cursor()
query = '''SELECT channel, publish_date, random_id FROM ramsey.metadata'''
iterations = pd.read_sql(query, con)
csr.execute(open('Queries\\create_initial.txt').read().replace('model', model))
csr.commit()

for i in range(len(iterations)):
    print(f'{i+1}/{len(iterations)}')
    query = f'''
    SELECT DATEDIFF(MONTH, metadata.publish_date, GETDATE()) AS 'age', audio.*
    FROM ramsey.audio
    LEFT JOIN ramsey.metadata
        ON audio.channel = metadata.channel
        AND audio.publish_date = metadata.publish_date
        AND audio.random_id = metadata.random_id
    WHERE audio.channel = '{iterations['channel'][i]}' 
        AND audio.publish_date = '{iterations['publish_date'][i]}'
        AND audio.random_id = {iterations['random_id'][i]}
    '''
    
    data = pd.read_sql(query, con)
    data_x = pipeline(data, results['lags'], results['leads'], set(panda['channel']))
    data_x = data_x.drop(['channel', 'publish_date', 'random_id'], axis=1).dropna()
    
    preds = pd.DataFrame({'y': discriminator.predict(data_x)}, index=data_x.index)
    preds = preds.merge(mapped, how='left', on='y').drop('y', axis=1)
    
    new = data[['channel', 'publish_date', 'random_id', 'second']].merge(preds, left_index=True, right_index=True)

    ramsey.upload(new, 'ramsey', model)
    
con.close()