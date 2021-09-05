from pytube import Playlist
from ramsey import ramsey
import pyodbc as sql
import pandas as pd
import numpy as np
import warnings
import librosa
import ray # ray requires scipy 1.6.3 to work

###################
# Define Function #
###################
@ray.remote
def encode_audio(sound):
    warnings.filterwarnings('ignore')
    
    speaker = sound[0]
    index = sound[1]
    second = sound[2]
    sound = sound[3]
    
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
        features = [speaker, index, second] + features
        
        features = pd.DataFrame([features])
        
    except ValueError:
        features = pd.DataFrame(index=[0])
        features.iloc[0, 0:3] = [speaker, index, second]
    
    except Exception:
        features = pd.DataFrame(index=[0])
        features.iloc[0, 0:3] = [speaker, index, second]
    
    return features


###############
# Scrape Data #
###############
personality = 'coleman'
username = 'zangorth'
password = open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\password.txt', 'r').read()

personalities = {'ramsey': 'https://www.youtube.com/watch?v=0JUw1agDjoA&list=UU7eBNeDW1GQf2NJQ6G6gAxw&index=2',
                 'deloney': 'https://www.youtube.com/watch?v=_wWc1Tc19qA&list=UU4HiMKM8WLcNbt9ae_XNRNQ&index=2',
                 'coleman': 'https://www.youtube.com/watch?v=aKRSyxnE3C4&list=UU0tVfiyBpMOQLA3FAanPGJA&index=2',
                 'ao': 'https://www.youtube.com/watch?v=adTnzyz7deI&list=UUaW51g-nmLfq703TPZC7Gsg&index=2',
                 'cruze': 'https://www.youtube.com/watch?v=PvwDX69CsAQ&list=UUt59W0ScV709iwy2h-oiulQ&index=2',
                 'wright': 'https://www.youtube.com/watch?v=bdXVQGZtYy4&list=UU1CHQyZ5-MTJzuSCvSVw_qg&index=2',
                 'kamel': 'https://www.youtube.com/watch?v=u5ufvVsaW4M&list=UUKFrkFOwmiXMuZtQJXuG5OQ&index=2'}

audio_location = f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Audio Full\\{personality}'
transcript_location = f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Transcript\\{personality}'

videos = Playlist(personalities[personality]).video_urls
videos = list(videos)

connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                     'Server=zangorth.database.windows.net;DATABASE=HomeBase;' +
                     f'UID={username};PWD={password}')
con = sql.connect(connection_string)
query = 'SELECT * FROM ramsey.metadata'
collected = pd.read_sql(query, con)
columns = pd.read_sql('SELECT TOP 1 * FROM ramsey.audio', con).columns
con.close()

videos = list(set(videos) - set(collected['link']))
video_link = videos[0]

ray.init(num_cpus=16)
i = 1
for video_link in videos:
    print(f'{i}/{len(videos)}')
    i += 1
    
    new = ramsey.Scrape(video_link, username, password, audio_location, transcript_location)
    metadata = new.metadata()
    new.audio()
    new.transcript()
    
    if type(metadata) != str:
        iterables = new.iterables()
        iterables = [[personality, metadata.id.item(), sound[0], sound[1]] for sound in iterables]
        
        audio_coding = ray.get([encode_audio.remote(sound) for sound in iterables])
        audio_coding = pd.concat(audio_coding)
        audio_coding.columns = columns

        metadata['seconds'] = audio_coding['second'].max()
        
        ramsey.upload(metadata, 'ramsey', 'metadata', username, password)
        ramsey.upload(audio_coding, 'ramsey', 'audio', username, password)
        
ray.shutdown()