from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from skopt import plots
import pyodbc as sql
from torch import nn
import pandas as pd
import numpy as np
import skopt
import torch

class Discriminator(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(193, a),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(a, b),
            nn.ReLU(),
            nn.Linear(b, 8),
            nn.Softmax(dim=1)
            )
        
        
    def forward(self, x):
        output = self.model(x)
        return output





con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT * 
FROM RAMSEY.dbo.AudioTraining
WHERE speaker NOT IN ('MIXED', 'NONE')
'''

panda = pd.read_sql(query, con)
con.close()

panda['y'] = panda['speaker'].astype('category').cat.codes
mapped = panda[['y', 'speaker']].drop_duplicates().reset_index(drop=True)
mapped = mapped.sort_values('y')

kf = StratifiedKFold()
y = panda['y']

x = panda.drop(['id', 'cut', 'speaker', 'y'], axis=1)
x = preprocessing.StandardScaler().fit_transform(x)


space = [
    skopt.space.Integer(1, 30, name='epochs'),
    skopt.space.Integer(2**2, 2**10, name='a'),
    skopt.space.Integer(2**2, 2**10, name='b'),
    skopt.space.Real(0.001, 0.5, name='lr', prior='log-uniform')
    ]

@skopt.utils.use_named_args(space)
def net(epochs, a, b, lr):
    
    f1 = []
    for train_index, test_index in kf.split(x, y):
        x_train = torch.from_numpy(x[train_index])
        x_test = torch.from_numpy(x[test_index])
        
        y_train = torch.from_numpy(y[train_index].to_numpy())
        y_test = torch.from_numpy(y[test_index].to_numpy())
        
        train_set = [(x_train[i], y_train[i]) for i in range(len(y_train))]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    
        loss_function = nn.CrossEntropyLoss()
        discriminator = Discriminator(a, b)
        optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                discriminator.zero_grad()
                yhat = discriminator(inputs.float())
                loss = loss_function(yhat, targets.long())
                loss.backward()
                optim.step()
                
        test_hat = discriminator(x_test.float())    
        f1.append(f1_score(y_test, np.argmax(test_hat.detach().numpy(), axis=1), average='micro'))
    
    print(np.mean(f1))
    return (- 1.0 * np.mean(f1))
        

result = skopt.forest_minimize(net, space, acq_func='PI')

print(f'Max F1: {result.fun}')
print(f'Parameters: {result.x}')
plots.plot_evaluations(result)


f1 = []
for train_index, test_index in kf.split(x, y):
    x_train = torch.from_numpy(x[train_index])
    x_test = torch.from_numpy(x[test_index])
    
    y_train = torch.from_numpy(y[train_index].to_numpy())
    y_test = torch.from_numpy(y[test_index].to_numpy())
    
    train_set = [(x_train[i], y_train[i]) for i in range(len(y_train))]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    discriminator = Discriminator(result.x[1], result.x[2])
    optim = torch.optim.Adam(discriminator.parameters(), lr=result.x[3])

    for epoch in range(result.x[0]):
        for i, (inputs, targets) in enumerate(train_loader):
            discriminator.zero_grad()
            yhat = discriminator(inputs.float())
            loss = loss_function(yhat, targets.long())
            loss.backward()
            optim.step()
            
    test_hat = discriminator(x_test.float())    
    f1.append(f1_score(y_test, np.argmax(test_hat.detach().numpy(), axis=1), average='micro'))
    
    print(f'Accuracy: {accuracy_score(y_test, np.argmax(test_hat.detach().numpy(), axis=1))}')
    mapped['f1'] = f1_score(y_test, np.argmax(test_hat.detach().numpy(), axis=1), average=None)
    mapped['recall'] = recall_score(y_test, np.argmax(test_hat.detach().numpy(), axis=1), average=None)
    print(mapped)
    print('')

print(np.mean(f1))




train_set = [(x[i], y[i]) for i in range(len(y))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

loss_function = nn.CrossEntropyLoss()
discriminator = Discriminator(result.x[1], result.x[2])
optim = torch.optim.Adam(discriminator.parameters(), lr=result.x[3])

for epoch in range(result.x[0]):
    for i, (inputs, targets) in enumerate(train_loader):
        discriminator.zero_grad()
        yhat = discriminator(inputs.float())
        loss = loss_function(yhat, targets.long())
        loss.backward()
        optim.step()


torch.save(discriminator, 'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Voice Recognition.pt')










        
