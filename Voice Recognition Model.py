from sklearn import preprocessing
import pyodbc as sql
from torch import nn
import pandas as pd
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(193, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
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

y = torch.from_numpy(panda['y'].to_numpy())
x = panda.drop(['id', 'cut', 'speaker', 'y'], axis=1)

x = preprocessing.StandardScaler().fit_transform(x)
x = torch.from_numpy(x)

train_set = [(x[i], y[i]) for i in range(len(y))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)


loss_function = nn.CrossEntropyLoss()
discriminator = Discriminator()
optim = torch.optim.Adam(discriminator.parameters(), lr=0.01)

for epoch in range(5):
    print(epoch)
    for i, (inputs, targets) in enumerate(train_loader):
        discriminator.zero_grad()
        yhat = discriminator(inputs.float())    
        loss = loss_function(yhat, targets.long())
        loss.backward()
        optim.step()
        
    test_hat = discriminator(x.float())
    