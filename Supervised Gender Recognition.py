from sqlalchemy import create_engine
from pydub import AudioSegment
import ramsey_helpers as RH
import pyodbc as sql
import pandas as pd
import urllib

#############
# Read Data #
#############
supervise = True

con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
                  
query = f'''
SELECT {'base.id' if supervise else "CONVERT(INT, base.id) AS 'id'"},
    base.[second], speaker, link{', gender' if not supervise else ''}
FROM {'RAMSEY.dbo.AudioCoding' if supervise else 'RAMSEY.prediction.Gender'} AS base
LEFT JOIN RAMSEY.dbo.metadata
    ON base.id = metadata.id
LEFT JOIN RAMSEY.prediction.Speaker
    ON base.id = Speaker.id
    AND base.[second] = Speaker.[second]
'''
                  
panda = pd.read_sql(query, con)


query = '''
SELECT DISTINCT id, second, 1 AS 'found'
FROM RAMSEY.training.Gender
'''

checked = pd.read_sql(query, con)

con.close()

#######################
# Supervised Learning #
#######################
sam_size = 505

samples = panda.merge(checked, how='left', on=['id', 'second'])
samples = samples.loc[samples.found.isnull()].reset_index(drop=True)
samples = samples.sample(sam_size).reset_index(drop=True)
    

for i in range(len(samples)):
    sample = samples['id'][i]
    second = samples['second'][i]
    speaker = samples['speaker'][i]
    link = samples['link'][i]
    prediction = '' if supervise else f'{samples["gender"][i]}? '
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    lead = sound[second*1000-3000: second*1000+3000]
    sound = sound[second*1000:second*1000+1000]
    
    gender = RH.train_audio(sound, lead, second, link, speaker, i, len(samples))
    
    upload = pd.DataFrame({'id': [sample], 'second': [second], 'gender': [gender], 'source': 'random'})
                
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    upload.to_sql(name='Gender', con=engine, schema='training', if_exists='append', index=False)