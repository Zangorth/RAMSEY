from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pyodbc as sql
import pandas as pd
import json
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\googledrive')

scope = ['https://www.googleapis.com/auth/drive.metadata.readonly',
         'https://www.googleapis.com/auth/drive.readonly']

creds = json.load(open('token.json', 'rb'))
creds = Credentials.from_authorized_user_info(creds, scope)

username = 'zangorth'
password = open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\password.txt', 'r').read()

if creds.expired:
    creds.refresh(Request())

gdrive = build('drive', 'v3', credentials=creds)

personalities = {'ao': '1ZQSou6B0kTcBFXcjPIxzGnj0FhRWvY-5',
                 'coleman': '1G5DYgtLxIB1Z6F8DUEjU5YvB853OWND3',
                 'cruze': '18x-UtzsxmUpbLIU32sj6RVEwFoF7JBCm', 
                 'deloney': '1SM9w7kJt7Wo_xWcGfT2X0IaBatd4Lo9-',
                 'kamel': '1lyR-ZBL1r07ymiEEy_eVinZA_skkbQb3',
                 'ramsey': '1LBMXAuaaYJtIrV9lcfsBn8Mp2XVPsIzA',
                 'wright': '1nac6lxzHzUaYVZwsfaXsZE4nk3R4tBOg'}

channel, identity, drive_link = [], [], []

for folder in personalities:
    pageToken, stop = None, False
    while not stop:
        request = gdrive.files().list(q=f"'{personalities[folder]}' in parents",
                                      pageSize=1000, pageToken=pageToken,
                                      fields='nextPageToken, files(id, name)').execute()
    
        files = request.get('files', [])
        
        channel.extend([folder for i in range(len(files))])
        identity.extend([files[i]['name'].split()[0] for i in range(len(files))])
        drive_link.extend([files[i]['id'] for i in range(len(files))])
        
        pageToken = request.get('nextPageToken', None)
        stop = True if pageToken is None else False

panda = pd.DataFrame({'channel': channel, 'id': identity, 'drive': drive_link})

panda = panda.loc[panda['id'].apply(lambda x: x.isnumeric())]
panda['channel'] = panda['channel'].astype(str)
panda['id'] = panda['id'].astype(int)
panda['drive'] = panda['drive'].astype(str)
panda = panda.reset_index(drop=True)

connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                     'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                     f'UID={username};PWD={password}')

con = sql.connect(connection_string)

collected = pd.read_sql('SELECT channel, id, drive AS found FROM ramsey.metadata', con)
panda = panda.merge(collected, on=['channel', 'id'], how='left')
panda = panda.loc[panda['found'].isnull()].reset_index(drop=True)

csr = con.cursor()

for i in range(len(panda)):
    print(i/len(panda))
    query = f'''
    UPDATE ramsey.metadata
    SET drive = '{panda["drive"][i]}'
    WHERE channel = '{panda["channel"][i]}' AND id = {panda["id"][i]}
    '''
    
    csr.execute(query)

csr.commit()
con.close()