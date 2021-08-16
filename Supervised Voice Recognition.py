from sqlalchemy import create_engine
from pydub import AudioSegment
import ramsey_helpers as RH
import pyodbc as sql
import pandas as pd
import urllib

#############
# Read Data #
#############
test = False
supervise = False

con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
                  
query = f'''
SELECT {'base.id' if supervise else "CONVERT(INT, base.id) AS 'id'"},
    [second], link{', speaker' if not supervise else ''}
FROM {'RAMSEY.dbo.AudioCoding' if supervise else 'RAMSEY.prediction.Speaker'} AS base
LEFT JOIN RAMSEY.dbo.metadata
    ON base.id = metadata.id
--WHERE YEAR(publish_date) = 2021
'''
                  
panda = pd.read_sql(query, con)


query = '''
SELECT DISTINCT id, second, 1 AS 'found'
FROM RAMSEY.training.Speaker
'''

checked = pd.read_sql(query, con)

con.close()

#######################
# Supervised Learning #
#######################
sam_size = 300

if not test:
    samples = panda.merge(checked, how='left', on=['id', 'second'])
    samples = samples.loc[samples.found.isnull()].reset_index(drop=True)
samples = samples.sample(sam_size).reset_index(drop=True)
    

for i in range(len(samples)):
    sample = samples['id'][i]
    second = samples['second'][i]
    link = samples['link'][i]
    prediction = '' if supervise else f'{samples["speaker"][i]}? '
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    lead = sound[second*1000-3000: second*1000+3000]
    sound = sound[second*1000:second*1000+1000]
    
    speaker = RH.train_audio(sound, lead, second, link, prediction, i, len(samples))
    
    upload = pd.DataFrame({'id': [sample], 'second': [second], 'speaker': [speaker], 'source': 'test'})
                
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    upload.to_sql(name='Speaker', con=engine, schema='training', if_exists='append', index=False)
