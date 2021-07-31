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
                  
query = 'SELECT id, second FROM RAMSEY.dbo.AudioCoding'
panda = pd.read_sql(query, con)
con.close()


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = '''
SELECT *
FROM RAMSEY.dbo.predictions
WHERE id IN (232, 5059, 4188, 5061, 1372, 321, 1359, 391)
'''
semi = pd.read_sql(query, con)
con.close()

#######################
# Supervised Learning #
#######################
# Identify videos in which each personality is a cohost so we can get a good sample
    # for all of their voices. Hogan and wright are particularly hard since she doesn't
    # cohost very often and he isn't a host anymore. 
#coleman = [21, 53, 59, 83, 114, 307, 358, 359]
#deloney = [8, 59, 80, 114, 124, 130, 139, 149, 372]
#wright = [103, 261, 276, 321]
#ao = [21, 35, 71, 77, 83, 102, 115, 124, 130, 446, 487]
#cruze = [315, 375, 378, 384, 389, 391, 550]
#hogan = [336, 337, 339, 343, 347, 367, 405, 430, 454, 458, 601, 623, 682, 689, 693, 722]
#samples = list(set(coleman + deloney + wright + ao + cruze + hogan))

full = True

if full:
    samples = pd.DataFrame({'id': [21, 53, 59, 83, 103, 114, 261, 276, 315, 321, 336, 337, 339, 375, 378],
                            'second': 0})
    start = 0

else:
    samples = panda.sample(120).sort_values(['id', 'second']).reset_index(drop=True)
    start = 10000
    

for i in range(len(samples)):
    sample = samples['id'][i]
    second = samples['second'][i]
    
    upload = pd.DataFrame(columns=['id', 'second', 'speaker'])
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    sound = sound if full else sound[second*1000-10000:second*1000+20000]
    
    for cut in range(start, len(sound), 1000):
        print(f'ID: {sample} | Second: {(cut-10000)/1000}')
        
        label = sound[cut:cut+1000]
        lead = sound[cut-10000:cut+1000]
        
        speaker = 0
        
        while speaker == 0:
            play(label)

            speaker = input('Speaker: ')
            speaker = 0 if speaker == '0' else speaker

            if speaker == 'lead':
                play(lead)
                speaker = 0
        
        
        upload = upload.append(pd.DataFrame({'id': [sample], 'second': [second], 
                                             'speaker': [speaker], 'source': 'supervised'}),
                               ignore_index=True, sort=False)
        
        second += 1
                
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

for sample in set(semi['id']):
    supervision = semi.loc[semi.id == sample].sort_values('second').reset_index(drop=True)
    second = supervision.second.min()
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    sound = sound[second*1000:]
    
    for cut in range(0, len(sound), 1000):
        label = supervision.loc[supervision.second == second, 'speaker'].item()
        
        play(sound[cut:cut+1000])
        speaker = input(f'Speaker {label}? ')
        
        if speaker != '':
            supervision.loc[supervision.second == second, 'speaker'] = speaker
            print(f'({second}/{len(sound)/1000}) Updated: Speaker = {speaker}')
        
        second += 1
        
    supervision['source'] = 'semi'
    supervision = supervision[['id', 'second', 'speaker', 'source']]
    
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    supervision.to_sql(name='AudioTraining', con=engine, schema='dbo', if_exists='append', index=False)










































