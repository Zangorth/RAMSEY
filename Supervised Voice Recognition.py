from sqlalchemy import create_engine
from pydub.playback import play
from pydub import AudioSegment
import pyodbc as sql
import pandas as pd
import numpy as np
import librosa
import urllib


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = 'SELECT * FROM RAMSEY.dbo.metadata'
panda = pd.read_sql(query, con)
con.close()

# Identify videos in which each personality is a cohost so we can get a good sample
    # for all of their voices. Hogan and wright are particularly hard since she doesn't
    # cohost very often and he isn't a host anymore. 
coleman = [4, 5, 21, 46, 53, 59, 83, 114, 307, 358, 359]
deloney = [8, 59, 80, 114, 124, 130, 139, 149, 372]
wright = [9, 19, 20, 103, 106, 261, 276, 321]
ao = [21, 35, 71, 77, 83, 102, 115, 124, 130, 446, 487]
cruze = [315, 375, 378, 384, 389, 391, 550]
hogan = [336, 337, 339, 343, 347, 367, 405, 430, 454, 458, 601, 623, 682, 689, 693, 722]

samples = list(set(coleman + deloney + wright + ao + cruze + hogan))

panda = pd.DataFrame(columns = ['id', 'cut', 'speaker'] + [i for i in range(193)])

# For each sample we start five seconds in listen to three seconds, record who the speaker is,
    # and then skip fifteen seconds ahead until the end of the recording. This probably provides
    # more samples than we'll need, but that's fine, no harm. 
for sample in samples:
    print(f'Video: {samples.index(sample)+1}/{len(samples)+1}')
    print('')
    
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')
    cut = 5000
    
    while cut+15000 < len(sound):
        out = sound[cut:cut+3000]
        out.export(f'C:\\Users\\Samuel\\Audio\\Audio Segment\\{sample} {cut}.wav', format='wav')
        
        speaker = 0
        
        while speaker == 0:
            play(out)
            
            speaker = input('Speaker: ')
            speaker = 0 if speaker == '0' else speaker
        
        y, rate = librosa.load(f'C:\\Users\\Samuel\\Audio\\Audio Segment\\{sample} {cut}.wav', res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
        
        features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
        features = [float(f) for f in features]
        features = [sample, cut, speaker] + features
        
        app = pd.DataFrame(features).T
        app.columns = ['id', 'cut', 'speaker'] + [i for i in range(193)]
        
        panda = panda.append(app, ignore_index=True, sort=False)
        
        cut = cut + 15000
        
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