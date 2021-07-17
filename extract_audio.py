from sqlalchemy import create_engine
from pydub import AudioSegment
import pandas as pd
import numpy as np
import warnings
import librosa
import urllib

warnings.filterwarnings('ignore')

def extract_audio(file):
    sound = AudioSegment.from_file(file)
    first, cut = True, 0
    
    while cut <= len(sound):
        out = sound[cut:cut+3000]
        
        try:
            y, rate = librosa.load(out.export(format='wav'), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
            stft = np.abs(librosa.stft(y))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
            
            features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
            features = [float(f) for f in features]
            features = [int(file.replace('.mp3', '')), cut/1000] + features
            
            if first:
                temp = pd.DataFrame(features).T
                temp.columns = (['id', 'second'] + [f'mfccs_{i}' for i in range(len(mfccs))] + 
                                [f'chroma_{i}' for i in range(len(chroma))] + [f'mel_{i}' for i in range(len(mel))] +
                                [f'contrast_{i}' for i in range(len(contrast))] + [f'tonnetz_{i}' for i in range(len(tonnetz))])
                
                panda = temp.copy()
                first = False
                
            else:
                temp = pd.DataFrame(features).T
                temp.columns = (['id', 'second'] + [f'mfccs_{i}' for i in range(len(mfccs))] + 
                                [f'chroma_{i}' for i in range(len(chroma))] + [f'mel_{i}' for i in range(len(mel))] +
                                [f'contrast_{i}' for i in range(len(contrast))] + [f'tonnetz_{i}' for i in range(len(tonnetz))])
                panda = panda.append(temp, ignore_index=True, sort=False)
                
        except ValueError:
            pass
        
        cut += 1000
        
    panda['id'] = panda['id'].astype(int)
        
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    
    panda.to_sql(name='AudioCoding', con=engine, schema='dbo', if_exists='append', index=False)
    
    return None