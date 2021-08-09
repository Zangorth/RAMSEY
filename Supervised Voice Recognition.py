from sqlalchemy import create_engine
from pydub.playback import play
from pydub import AudioSegment
import pyodbc as sql
import pandas as pd
import urllib

#############
# Read Data #
#############
con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT AudioCoding.id, second, metadata.link
FROM RAMSEY.dbo.AudioCoding
LEFT JOIN RAMSEY.DBO.metadata
    ON AudioCoding.id = metadata.id
'''
panda = pd.read_sql(query, con)

con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT *
FROM RAMSEY.dbo.predictions
'''
semi = pd.read_sql(query, con)



query = '''
SELECT DISTINCT id, second, 1 AS 'found'
FROM RAMSEY.dbo.AudioTraining
'''

checked = pd.read_sql(query, con)

con.close()



#######################
# Supervised Learning #
#######################
samples = panda.sample(4000).reset_index(drop=True)
    

for i in range(len(samples)):
    sample = samples['id'][i]
    second = samples['second'][i]
    
    upload = pd.DataFrame(columns=['id', 'second', 'speaker', 'source'])
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    lead = sound[second*1000-5000: second*1000+5000]
    sound = sound[second*1000:second*1000+1000]
        
    speaker = 0
    
    while speaker == 0:
        play(sound)

        speaker = input(f'({i+1}/{len(samples)}) Speaker: ')
        speaker = 0 if speaker == '0' else speaker

        if str(speaker).lower() == 'lead':
            play(lead)
            speaker = 0
            
        elif str(speaker).lower() == 'show':
            print(samples.link[i])
            speaker == 0
    
    
    upload = pd.DataFrame({'id': [sample], 'second': [second], 'speaker': [speaker], 'source': 'supervised'})
    
                
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    upload.to_sql(name='AudioTraining', con=engine, schema='dbo', if_exists='append', index=False)



##############################
# Semi - Supervised Learning #
##############################
upload = pd.DataFrame(columns=['id', 'second', 'label', 'speaker', 'source'])

samples = semi.groupby('speaker', group_keys=False).apply(lambda x: x.sample(min(len(x), 100)))
samples = samples.reset_index(drop=True)
samples = samples.merge(checked, how='left', on=['id', 'second'])
samples = samples.loc[samples.found.isnull(), ['id', 'second', 'speaker']].reset_index(drop=True)

for i in range(len(samples)):
    sample = int(samples['id'][i])
    second = samples['second'][i]
    label = samples['speaker'][i]
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    lead = sound[second*1000-5000: second*1000+1000]
    sound = sound[second*1000:second*1000+1000]
    
    speaker = 0
        
    while speaker == 0:
        play(sound)

        speaker = input(f'({len(upload)+1}/{len(samples)}) Speaker {label}? ')
        speaker = 0 if speaker == '0' else speaker

        if speaker == '':
            speaker = label
        elif str(speaker).lower() == 'lead':
            play(lead)
            speaker = 0
            
    upload = upload.append(pd.DataFrame({'id': [sample], 'second': [second], 
                                         'label': label, 'speaker': [speaker], 'source': 'semi'}),
                           ignore_index=True, sort=False)
  
upload = upload[['id', 'second', 'speaker', 'source']]
upload['source'] = 'semi'

conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
con = urllib.parse.quote_plus(conn_str)
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
upload.to_sql(name='AudioTraining', con=engine, schema='dbo', if_exists='append', index=False)
