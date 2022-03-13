from pytube import Playlist
from ramsey import ramsey
import pyodbc as sql
import pandas as pd
import numpy as np
import warnings
import librosa
import ray

###################
# Define Function #
###################
@ray.remote
def encode_audio(sound):
    warnings.filterwarnings('ignore')
    
    channel = sound[0]
    publish_date = sound[1]
    random_id = sound[2]
    second = sound[3]
    sound = sound[4]
    
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
        features = [channel, publish_date, random_id, second] + features
        
        features = pd.DataFrame([features])
    
    except Exception:
        features = pd.DataFrame(index=[0])
        features.iloc[0, 0:3] = [channel, publish_date, random_id, second]
    
    return features


###############
# Scrape Data #
###############
personalities = {'ramsey': 'https://www.youtube.com/watch?v=j3NNJxYb9Sg&list=UU7eBNeDW1GQf2NJQ6G6gAxw',
                 'deloney': 'https://www.youtube.com/watch?v=ud3krvLD1bo&list=UU4HiMKM8WLcNbt9ae_XNRNQ',
                 'coleman': 'https://www.youtube.com/watch?v=PRNZ199_ax4&list=UU0tVfiyBpMOQLA3FAanPGJA',
                 'ao': 'https://www.youtube.com/watch?v=iE2yBe21-eA&list=PLPIXh_zvJ-4AEupWXKvlPQF2NFDQceoro',
                 'cruze': 'https://www.youtube.com/watch?v=-XU-xOg3pHM&list=UUt59W0ScV709iwy2h-oiulQ',
                 'wright': 'https://www.youtube.com/watch?v=zhp-1T1na8Q&list=UU1CHQyZ5-MTJzuSCvSVw_qg',
                 'kamel': 'https://www.youtube.com/watch?v=NXCsCMSWfiI&list=UUKFrkFOwmiXMuZtQJXuG5OQ'}

for personality in list(personalities.keys()):
    audio_location = f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Audio Full\\{personality}'
    transcript_location = f'C:\\Users\\Samuel\\Google Drive\\Portfolio\\Ramsey\\Audio\\Transcript\\{personality}'
    
    videos = Playlist(personalities[personality]).video_urls
    videos = list(videos)
    
    connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                         'Server=ZANGORTH;DATABASE=HomeBase;' +
                         'Trusted_Connection=yes;')
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
        print(f'{personality}: {i}/{len(videos)}')
        i += 1
        
        try:
            new = ramsey.Scrape(video_link, audio_location, transcript_location)
            metadata = new.metadata()
            new.audio()
            new.transcript()
            
        except Exception:
            metadata = ''
        
        if type(metadata) != str:
            iterables = new.iterables()
            iterables = [[personality, metadata.publish_date.item(), metadata.random_id.item(), sound[0], sound[1]] for sound in iterables]
            
            audio_coding = ray.get([encode_audio.remote(sound) for sound in iterables])
            audio_coding = pd.concat(audio_coding)
            audio_coding.columns = columns
    
            metadata['seconds'] = audio_coding['second'].max()
            
            ramsey.upload(metadata, 'ramsey', 'metadata')
            ramsey.upload(audio_coding, 'ramsey', 'audio')
            
    ray.shutdown()