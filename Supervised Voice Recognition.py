from sqlalchemy import create_engine
from pydub.playback import play
from pydub import AudioSegment
import pyodbc as sql
import pandas as pd
import urllib

#############
# Read Data #
#############
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
sam_size = 15

samples = panda.merge(checked, how='left', on=['id', 'second'])
samples = samples.loc[samples.found.isnull()].reset_index(drop=True)
samples = samples.sample(sam_size).reset_index(drop=True)
    

for i in range(len(samples)):
    sample = samples['id'][i]
    second = samples['second'][i]
    prediction = '' if supervise else f'{samples["speaker"][i]}? '
    
    upload = pd.DataFrame(columns=['id', 'second', 'speaker', 'source'])
        
    speaker = 0
    
    while speaker == 0:
        sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
        lead = sound[second*1000-3000: second*1000+3000]
        sound = sound[second*1000:second*1000+1000]
        
        play(sound)

        speaker = input(f'({i+1}/{len(samples)}) Speaker: {prediction}')
        speaker = speaker.upper()
        speaker = 0 if speaker == '0' else speaker

        if str(speaker).lower() == '' and not supervise:
            speaker = samples['speaker'][i]

        elif str(speaker).lower() == 'lead':
            play(lead)
            speaker = 0
            
        elif str(speaker).lower() == 'show':
            print(samples.link[i])
            print(second)
            speaker = 0
    
    
    upload = pd.DataFrame({'id': [sample], 'second': [second], 'speaker': [speaker], 'source': 'random'})
                
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    upload.to_sql(name='SpeakerTraining', con=engine, schema='training', if_exists='append', index=False)
