from youtube_transcript_api import YouTubeTranscriptApi
from sqlalchemy import create_engine
from pydub import AudioSegment
from datetime import datetime
from pytube import YouTube
import streamlit as st
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
    def __init__(self, link, username, password, audio_location, transcript_location):
        self.link = link
        self.audio_location, self.transcript_location = audio_location, transcript_location
        self.username, self.password = username, password
        
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
                             'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                             f'UID={self.username};PWD={self.password}')                             
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
        self.i = collected.loc[collected['channel'] == self.personality, 'id'].max() + 1 if len(collected.loc[collected['channel'] == self.personality, 'id']) > 0 else 0
        
        name = self.yt.title
        name = name.translate(str.maketrans('', '', string.punctuation)).lower()
        self.name = name
        
        keywords = ('|'.join(self.yt.keywords)).replace("'", "''").lower()
        
        if (datetime.now() - self.yt.publish_date).days < 7:
            return 'Videos are only recorded after having been published at least 7 days'
        
        upload = pd.DataFrame({'channel': self.personality, 'id': [self.i],
                               'title': [name], 'link': [self.link],
                               'publish_date': [self.yt.publish_date.strftime("%Y/%m/%d")],
                               'keywords': [keywords], 'seconds': [self.yt.length], 'rating': [self.yt.rating], 'view_count': [self.yt.views]})
            
        return upload
        
    def audio(self):
        os.chdir(self.audio_location)
        self.yt.streams.filter(only_audio=True).first().download()
        current_name = [f for f in os.listdir() if '.mp4' in f][0]
        self.file = f'{self.i} {self.personality}.mp3'
        
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
            pickle.dump(transcript, open(f'{self.transcript_location}\\{self.i} {self.personality}.pkl', 'wb'))
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
            features = [self.personality, self.i, second] + features
            
            features = pd.DataFrame([features], columns=self.columns)
            
        except ValueError:
            features = pd.DataFrame(columns=self.columns, index=[0])
            features.iloc[0, 0:2] = [self.personality, self.i, second]
            
        except ParameterError:
            features = pd.DataFrame(columns=self.columns, index=[0])
            features.iloc[0, 0:2] = [self.personality, self.i, second]
        
        except Exception:
            features = pd.DataFrame(columns=self.columns, index=[0])
            features.iloc[0, 0:2] = [self.personality, self.i, second]
        
        return features

###############
# Upload Data #
###############
def upload(dataframe, schema, table, username, password, exists='append'):
    if 'guest' not in username:
        conn_str = (
                r'Driver={ODBC Driver 17 for SQL Server};'
                r'Server=zangorth.database.windows.net;'
                r'Database=HomeBase;'
                f'UID={username};'
                f'PWD={password};'
            )
        azure = urllib.parse.quote_plus(conn_str)
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={azure}')
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

###########################
# Streamlit: Data Collect #
###########################
def data_collect(video_link, username, password, audio_location, transcript_location, verbose=True, parallel=False):
    with st.spinner('Downloading Audio'):
        new = Scrape(video_link, username, password, audio_location, transcript_location)
        metadata = new.metadata()
        new.audio()
        new.transcript()
    
    if type(metadata) == str:
        st.write(metadata)
        return []
    
    elif type(metadata) == pd.core.frame.DataFrame:
        if verbose:
            st.write('Meta Data')
            st.dataframe(metadata, width=5000)
        
        with st.spinner('Generating Audio Slices'):
            iterables = new.iterables()
                
        iteration = st.empty()
        i, pb = 0, st.progress(0)
        audio_coding = []
        for sound in iterables:
            audio_coding.append(new.encode_audio(sound))
            
            iteration.text(f'Encoding Audio - Seconds Processed: {i+1}/{len(iterables)}')
            pb.progress((i+1)/len(iterables))
            
            i += 1
                
        iteration.empty()
        pb.empty()
    
        audio_coding = pd.concat(audio_coding)
        
        if 'guest' in username and verbose:
            st.write('Data from Guest Profiles will not be uploaded')
        
        elif username == 'zangorth':
            with st.spinner('Uploading Metadata'):
                metadata['seconds'] = audio_coding['second'].max()
                upload(metadata, 'ramsey', 'metadata', username, password)
            
            with st.spinner('Uploading Audio Data'):
                upload(audio_coding, 'ramsey', 'audio', username, password)
        
        if verbose:
            st.write('')
            st.write('Audio Code')
            st.dataframe(audio_coding, width=5000)
        
    return [metadata, audio_coding]
        
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