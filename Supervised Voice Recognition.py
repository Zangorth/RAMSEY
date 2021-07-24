from sqlalchemy import create_engine
from pydub.playback import play
from pydub import AudioSegment
import pyodbc as sql
import pandas as pd
import numpy as np
import urllib


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
query = '''
SELECT *
FROM RAMSEY.dbo.AudioCoding
WHERE id IN (4, 5, 8, 59, 9, 19, 21, 35, 315, 375, 336, 337)
ORDER BY id, second
'''
panda = pd.read_sql(query, con)
panda['speaker'] = np.nan
con.close()

# Identify videos in which each personality is a cohost so we can get a good sample
    # for all of their voices. Hogan and wright are particularly hard since she doesn't
    # cohost very often and he isn't a host anymore. 
#coleman = [4, 5, 21, 46, 53, 59, 83, 114, 307, 358, 359]
#deloney = [8, 59, 80, 114, 124, 130, 139, 149, 372]
#wright = [9, 19, 20, 103, 106, 261, 276, 321]
#ao = [21, 35, 71, 77, 83, 102, 115, 124, 130, 446, 487]
#cruze = [315, 375, 378, 384, 389, 391, 550]
#hogan = [336, 337, 339, 343, 347, 367, 405, 430, 454, 458, 601, 623, 682, 689, 693, 722]
#samples = list(set(coleman + deloney + wright + ao + cruze + hogan))
samples = [4, 5, 8, 59, 9, 19, 21, 35, 315, 375, 336, 337]
samples.sort()

for sample in samples:
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    
    for cut in panda.loc[panda['id'] == sample, 'second'].sort_values():
        print(f'ID: {sample} | Second: {cut}')
        
        label = sound[cut*1000:cut*1000+1000]
        full = sound[cut*1000-2000:cut*1000+2000]
        
        speaker = 0
        
        while speaker == 0:
            play(label)
            
            speaker = input('Speaker: ')
            speaker = 0 if speaker == '0' else speaker
            
            if speaker == 'full':
                play(full)
                speaker = 0
        
        panda.loc[(panda.id == sample) & (panda.second == cut), 'speaker'] = speaker
        print('')
            
    
    

        
# Upload the identified samples to SQL
conn_str = (
    r'Driver={SQL Server};'
    r'Server=ZANGORTH\HOMEBASE;'
    r'Database=RAMSEY;'
    r'Trusted_Connection=yes;'
)
con = urllib.parse.quote_plus(conn_str)

engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')

panda.to_sql(name='AudioTraining', con=engine, schema='dbo', if_exists='replace', index=False)