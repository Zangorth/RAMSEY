from youtube_transcript_api import YouTubeTranscriptApi
from sqlalchemy import create_engine
from pydub.playback import play
from pydub import AudioSegment
import multiprocessing as mp
from pytube import Playlist
from pytube import YouTube
from random import randint
from time import sleep
from tqdm import tqdm
import pyodbc as sql
import pandas as pd
import numpy as np
import warnings
import librosa
import urllib
import pickle
import string
import os

class ParameterError(Exception):
    pass

class NoTranscriptFound(Exception):
    pass

##################
# Audio Training #
##################
def train_audio(sound, lead, second, link, prediction='', iterator='', size=''):
    status = '' if iterator == '' and size == '' else f'{iterator}/{size}'
    string_pred = f' {prediction}?' if prediction != '' else ''

    train = 0
    while train == 0:
        play(sound)

        train = input(f'{status} Label{string_pred}: ')
        train = train.upper()
        train = 0 if train == '0' else train

        if str(train).lower() == '':
            train = prediction

        elif str(train).lower() == 'lead':
            train = 0
            play(lead)

        elif str(train).lower() == 'show':
            train = 0
            timestamp = f'{int(round(second/60, 0))}:{second % 60}'
            print(f'{timestamp} - {link}')

        else:
            pass

    return train

##############
# Lags/Leads #
##############
def shift(x, group, lags, leads, exclude = []):
    out = x.copy()
    x = out[[col for col in out.columns if col not in exclude]]
    
    for i in range(lags):
        lag = x.groupby(group).shift(i)
        lag.columns = [f'{col}_lag{i}' for col in lag.columns]
        
        out = out.merge(lag, left_index=True, right_index=True)
        
    for i in range(leads):
        lead = x.groupby(group).shift(-i)
        lead.columns = [f'{col}_lead{i}' for col in lead.columns]
        
        out = out.merge(lead, left_index=True, right_index=True)
        
    return out


###############
# Data Scrape #
###############
class NewData():
    def __init__(self, link, audio_location, transcript_location):
        self.link = link
        self.audio_location = audio_location
        self.transcript_location = transcript_location
        
    def scrape(self):
        con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                      Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                      Trusted_Connection=yes;''')
                      
        query = 'SELECT * FROM RAMSEY.dbo.metadata'
        collected = pd.read_sql(query, con)
        
        self.i = collected.id.max() + 1
        
        yt = YouTube(self.link)
        
        if yt.channel_id != 'UC7eBNeDW1GQf2NJQ6G6gAxw':
            return 'Please only upload videos from The Ramsey Show - Highlights' 
        
        if self.link in list(collected['link']):
            return 'Link already exists in database'
        
        name = yt.streams[0].title
        name = name.translate(str.maketrans('', '', string.punctuation)).lower()
        
        keywords = ('|'.join(yt.keywords)).replace("'", "''").lower()
        
        con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                          Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                          Trusted_Connection=yes;''')
                         
        csr = con.cursor()
        query = f'''
        INSERT INTO RAMSEY.dbo.metadata 
        VALUES ({self.i}, '{name}', '{self.link}', '{yt.publish_date.strftime("%Y/%m/%d")}', '{keywords}', {yt.length}, {yt.rating}, {yt.views})'''
        
        csr.execute(query)
        csr.commit()
        con.close()
        
        yt.streams.filter(only_audio=True).first().download(f'{self.audio_location}/{name}')
        
        current_name = os.listdir(f'{self.audio_location}/{name}')[0]
        os.rename(f'{self.audio_location}\\{name}\\{current_name}', f'{self.audio_location}\\{self.i}.mp3')
        os.rmdir(f'{self.audio_location}\\{name}')
        
        self.file = f'{self.audio_location}\\{self.i}.mp3'
        
        transcript_link = self.link.split('?v=')[-1]
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(transcript_link)
            pickle.dump(transcript, open(f'{self.transcript_location}\\{self.i}.pkl', 'wb'))
        except NoTranscriptFound:
            pass
        except Exception:
            pass
        
        con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                          Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                          Trusted_Connection=yes;''')
                          
        query = f'SELECT * FROM RAMSEY.dbo.metadata WHERE id = {self.i}'
        new = pd.read_sql(query, con)
        con.close()
        
        return new

    def iterables(self):
        con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
        query = 'SELECT DISTINCT id FROM RAMSEY.dbo.AudioCoding'
        completed = pd.read_sql(query, con)
        completed = completed['id'].tolist()
        self.columns = pd.read_sql('Select TOP 1 * FROM RAMSEY.dbo.AudioCoding', con).columns
        con.close()
        
        if self.i in completed:
            return 'Audio Already Encoded'
        
        sound = AudioSegment.from_file(self.file)
        iterables = [[int(self.file.replace(f'{self.audio_location}\\', '').replace('.mp3', '')), 
                      cut, sound[cut*1000:cut*1000+1000]] for cut in range(int(round(len(sound)/1000, 0)))]
        
        return iterables
    
    def encode_audio(self, sound):
        warnings.filterwarnings('ignore')
        try:
            file = sound[0]
            cut = sound[1]
            sound = sound[2]
            
            y, rate = librosa.load(sound.export(format='wav'), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
            stft = np.abs(librosa.stft(y))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
            
            features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
            features = [float(f) for f in features]
            features = [file, cut] + features
            
        except ValueError:
            features = []
            
        except ParameterError:
            features = []
        
        except Exception:
            features = []
    
        return features
    
    def upload(self, dataframe):
        dataframe.columns = self.columns
        
        conn_str = (
            r'Driver={SQL Server};'
            r'Server=ZANGORTH\HOMEBASE;'
            r'Database=RAMSEY;'
            r'Trusted_Connection=yes;'
        )
        con = urllib.parse.quote_plus(conn_str)
        
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
        
        dataframe.to_sql(name='AudioCoding', con=engine, schema='dbo', if_exists='append', index=False)
        
        return dataframe
        
        
        





































