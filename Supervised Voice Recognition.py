from pydub.playback import play
from pydub import AudioSegment
from random import randint
import pyodbc as sql
import pandas as pd
import numpy as np
import librosa


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = 'SELECT * FROM RAMSEY.dbo.metadata'
panda = pd.read_sql(query, con)
con.close()

panda = pd.DataFrame(columns = ['id', 'cut', 'speaker'] + [i for i in range(193)])

samples = panda.sample(2, random_state=52)['id'].tolist()

for sample in samples:
    sound = AudioSegment.from_file(f'C:\\Users\\Samuel\\Audio\\Audio Full\\{sample}.mp3')

    for i in range(0, 5):
        cut = randint(0, len(sound)-3000)
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
        features = [sample, cut, speaker] + features
        
        app = pd.DataFrame(features).T
        app.columns = ['id', 'cut', 'speaker'] + [i for i in range(193)]
        
        panda = panda.append(app, ignore_index=True, sort=False)