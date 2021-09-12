from youtube_transcript_api import YouTubeTranscriptApi
from sqlalchemy import create_engine
from pydub import AudioSegment
from datetime import datetime
from pytube import YouTube
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

###############
# Data Scrape #
###############
class Scrape():
    def __init__(self, link, audio_location, transcript_location):
        self.link = link
        self.audio_location, self.transcript_location = audio_location, transcript_location
        
        self.personalities = {'UC7eBNeDW1GQf2NJQ6G6gAxw': 'ramsey',
                              'UC4HiMKM8WLcNbt9ae_XNRNQ': 'deloney',
                              'UC0tVfiyBpMOQLA3FAanPGJA': 'coleman',
                              'UCaW51g-nmLfq703TPZC7Gsg': 'ao',
                              'UCt59W0ScV709iwy2h-oiulQ': 'cruze',
                              'UC1CHQyZ5-MTJzuSCvSVw_qg': 'wright',
                              'UCKFrkFOwmiXMuZtQJXuG5OQ': 'kamel'}
        
        return None

    def metadata(self):
        connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                             'Server=ZANGORTH;DATABASE=HomeBase;' +
                             'Trusted_Connection=yes;') 
        con = sql.connect(connection_string)
        query = 'SELECT * FROM [ramsey].[metadata]'
        collected = pd.read_sql(query, con)
        self.columns = pd.read_sql('SELECT TOP 1 * FROM [ramsey].[audio]', con).columns
        con.close()
        
        self.yt = YouTube(self.link)
        
        if self.yt.channel_id not in [personality for personality in self.personalities]:
            return 'Please only upload videos from the Ramsey Personality Channels'
        
        if self.link in list(collected['link']):
            return 'Link already exists in database'
        
        self.personality = self.personalities[self.yt.channel_id]
        
        name = self.yt.title
        name = name.translate(str.maketrans('', '', string.punctuation)).lower()
        self.name = name
        
        keywords = ('|'.join(self.yt.keywords)).replace("'", "''").lower()
        self.publish_date = self.yt.publish_date.strftime("%Y-%m-%d")
        self.random_id = int(round(np.random.uniform()*1000000, 0))
        
        if (datetime.now() - self.yt.publish_date).days < 7:
            return 'Videos are only recorded after having been published at least 7 days'
        
        out = pd.DataFrame({'channel': self.personality,
                            'publish_date': [self.publish_date],
                            'random_id': [self.random_id],
                            'title': [name], 'link': [self.link],
                            'keywords': [keywords], 'seconds': [self.yt.length], 
                            'rating': [self.yt.rating], 'view_count': [self.yt.views],
                            'upload_date': [datetime.now().strftime("%Y-%m-%d")]})
            
        return out
        
    def audio(self):
        os.chdir(self.audio_location)
        self.yt.streams.filter(only_audio=True).first().download()
        current_name = [f for f in os.listdir() if '.mp4' in f][0]
        self.file = f'{self.personality} {str(self.publish_date)} {self.random_id}.mp3'
        
        try:
            os.rename(current_name, self.file)
        except FileExistsError:
            os.remove(self.file)
            os.rename(current_name, self.file)
        
        return None
        
        
    def transcript(self):
        transcript_link = self.link.split('?v=')[-1]
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(transcript_link)
            pickle.dump(transcript, open(f'{self.transcript_location}\\{self.file.replace(".mp3", ".pkl")}', 'wb'))
        except NoTranscriptFound:
            pass
        except Exception:
            pass
        
        return None
    
    def iterables(self):
        sound = AudioSegment.from_file(self.file)
        iterables = [[cut, sound[cut*1000:cut*1000+1000]] for cut in range(int(round(len(sound)/1000, 0)))]
        
        return iterables
    
    def encode_audio(self, sound):
        warnings.filterwarnings('ignore')
        
        second = sound[0]
        sound = sound[1]
        
        try:
            y, rate = librosa.load(sound.export(format='wav'), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0)
            stft = np.abs(librosa.stft(y))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y, sr=rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=rate).T,axis=0)
            
            features = list(mfccs) + list(chroma) + list(mel) + list(contrast) + list(tonnetz)
            features = [float(f) for f in features]
            features = [self.personality, self.publish_date, self.random_id, second] + features
            
            features = pd.DataFrame([features], columns=self.columns)
            
        except ValueError:
            features = pd.DataFrame(columns=self.columns, index=[0])
            features.iloc[0, 0:3] = [self.personality, self.publish_date, self.random_id, second]
            
        except ParameterError:
            features = pd.DataFrame(columns=self.columns, index=[0])
            features.iloc[0, 0:3] = [self.personality, self.publish_date, self.random_id, second]
        
        except Exception:
            features = pd.DataFrame(columns=self.columns, index=[0])
            features.iloc[0, 0:3] = [self.personality, self.publish_date, self.random_id, second]
        
        return features

###############
# Upload Data #
###############
def upload(dataframe, schema, table, exists='append'):
    conn_str = (
            r'Driver={ODBC Driver 17 for SQL Server};'
            r'Server=ZANGORTH;'
            r'Database=HomeBase;'
            'Trusted_Connection=yes;'
        )
    con = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    dataframe.to_sql(name=table, con=engine, schema=schema, if_exists=exists, index=False)
    
    return None


##############
# Index Data #
##############

def reindex(schema, table, index, username, password, alter = {'channel': 'VARCHAR(10)'}):
    if 'guest' not in username:
        connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                              'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                              f'UID={username};PWD={password}')
        azure = sql.connect(connection_string)
        csr = azure.cursor()
        
        for key in alter:
            query = f'''
            ALTER TABLE {schema}.{table}
            ALTER COLUMN {key} {alter[key]}
            '''
            
            csr.execute(query)
        
        query = f'''
        CREATE CLUSTERED INDEX IX_{table}
        ON {schema}.{table}({', '.join(index)})
        '''
        
        csr.execute(query)
        csr.commit()
        azure.close()
        
    return None
        
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